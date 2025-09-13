import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
# from methods.saito import saito
from methods.hunemohr import hunemohr
# from methods.tanaka import tanaka
from methods.schneider import schneider
from pathlib import Path
import os
import re
from datetime import datetime
import json
import shutil

# Constants
DATA_LOCO = Path("test_images")
KVP_PAIRS = [(70, 100), (70, 120), (70, 140), (80, 100), (80, 120), (80, 140)]
SERIES_RE = re.compile(
    r'^(?:degraded-)?(.+)-(\d+(?:\.\d+)?)-(\d+)$', re.IGNORECASE)
NOISY_SERIES_RE = re.compile(
    r'^(?:degraded-)?(.+)-(\d+(?:\.\d+)?)-(\d+)-(\d+(?:\.\d+)?)$', re.IGNORECASE)
VAR = [0.01, 0.05, 0.1]
MAX_SLICES = 10

# Process uploaded folder of series
def process_upload(series_path, out_root="test_images"):

    series_path = Path(series_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for root, subdirs, files in os.walk(series_path):
        root = Path(root)

        # Skip writing anything for the top if it directly contains files you don't intend to process.
        for filename in files:
            if filename.startswith('.') or filename == '.DS_Store':
                continue

            src_path = root / filename

            # Build output directory under the fixed local folder "test_images"
            for var in VAR:
                subfolder_name = root.name
                out_dir = out_root / f"degraded-{subfolder_name}-{var}"
                out_dir.mkdir(parents=True, exist_ok=True)

                degrade_image(src_path, out_dir, var)


def degrade_image(file: str | Path, out_dir: str | Path, var):
    """
    Read a DICOM, add Gaussian noise, and saves it as a DICOM:
      out_dir / (stem + ".dcm")
    """
    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dicom_data = pydicom.dcmread(str(file))
    image = dicom_data.pixel_array.astype(np.float32)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = cv2.resize(image, (512, 512))
    image = image.flatten()
    image = np.expand_dims(image, axis=0)

    x, y = image.shape
    mean = 0
    sigma = np.sqrt(var)
    n = np.random.normal(loc=mean, scale=sigma, size=(x, y))
    degraded_image = image + n

    # Save as png
    out_path = out_dir / f"{file.stem}.png"
    degraded_2d = degraded_image.reshape(512, 512)
    degraded_u8 = (np.clip(degraded_2d, 0.0, 1.0) * 255.0).astype(np.uint8)

    ok = cv2.imwrite(str(out_path), degraded_u8)

    # Convert from png to DICOM
    img = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)

    rows, cols = img.shape
    pixel_array = img.astype(np.uint16)

    # Create FileDataset
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out_path_dcm = out_dir / f"{file.stem}.dcm"

    ds = FileDataset(str(out_path_dcm), {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    # Set required DICOM tags
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "OT"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.RescaleSlope = dicom_data.RescaleSlope
    ds.RescaleIntercept = dicom_data.RescaleIntercept

    # Set image-specific data
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned
    ds.PixelData = pixel_array.tobytes()

    # Set timestamp
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')

    # Save DICOM file
    ds.save_as(out_path_dcm)
    print(f"DICOM saved to {out_path_dcm}")

    # Delete temporary PNG
    try:
        os.remove(out_path)
        print(f"Successfuly deleted PNG: {out_path}")
    except Exception as e:
        print(f"Failed to delete PNG: {e}")


def index_series_by_kvp(root, noisy=False):
    '''
    Walks root directory and returns:
      {(prefix, thickness): {kvp: pathToSeries}}  for clean
      {(prefix, thickness, noise): {kvp: pathToSeries}} for noisy
    '''
    index = {}

    for paths, subdirs, files in os.walk(root):
        base = Path(paths).name
        m = NOISY_SERIES_RE.match(base) if noisy else SERIES_RE.match(base)
        if not m:
            continue

        if noisy:
            prefix, thickness, kvp_str, noise_str = m.groups()
            kvp = int(kvp_str)
            noise = float(noise_str)

            key = (prefix, thickness, noise)
        else:
            prefix, thickness, kvp_str = m.groups()
            kvp = int(kvp_str)

            key = (prefix, thickness)

        index.setdefault(key, {})
        index[key][kvp] = Path(paths)

    return index


def get_sorted_dicoms(directory: Path):
    """
    Returns a sorted list of DICOM file paths in the given directory.
    """
    return sorted([p for p in directory.glob("*.dcm") if p.is_file()])


def run_methods(i, clean_high_file, clean_low_file, noisy_high_file, noisy_low_file, method_name, method_fn, prefix, thickness, kvp_low, kvp_high, noise_level, radii, phantom_type):
    result = []
    try:
        print(
            f"  → Running {kvp_low}/{kvp_high} pair index {i} with {method_name}")

        clean_result = method_fn(
            clean_high_file, clean_low_file, phantom_type, radii)
        noisy_result = method_fn(
            noisy_high_file, noisy_low_file, phantom_type, radii)

        result.append({
            "phantom": prefix,
            "thickness": thickness,
            "kvp_pair": (kvp_low, kvp_high),
            "method": method_name,
            "pair_index": i,
            "clean": clean_result,
            "noisy": noisy_result,
            "noise_level": noise_level
        })
    except Exception as e:
        print(
            f"    [!] Error running {method_name} on pair index {i} ({kvp_low}/{kvp_high}): {e}")
    return result


def test(series_clean, series_noisy_root, phantom_type, radii):
    clean_idx = index_series_by_kvp(series_clean)
    results = []

    noisy_idx = index_series_by_kvp(series_noisy_root, noisy=True)

    if not noisy_idx:
        print("❌ No noisy series found.")
        return

    for clean_key in clean_idx:
        prefix, thickness = clean_key
        print(
            f"\n=== Starting TEST: phantom={prefix}, thickness={thickness}, radii={radii} ===")

        for noisy_key in sorted(noisy_idx.keys(), key=lambda x: x[2]):
            noisy_prefix, noisy_thickness, noise_level = noisy_key

            if (noisy_prefix != prefix or noisy_thickness != thickness):
                continue

            print(f"\n🔊 Noise level: {noise_level}")
            print(f"--- Clean: {clean_key} | Noisy: {noisy_key} ---")

            for kvp_low, kvp_high in KVP_PAIRS:
                clean_low_dir = clean_idx[clean_key].get(kvp_low)
                clean_high_dir = clean_idx[clean_key].get(kvp_high)
                noisy_low_dir = noisy_idx[noisy_key].get(kvp_low)
                noisy_high_dir = noisy_idx[noisy_key].get(kvp_high)

                if not all([clean_low_dir, clean_high_dir, noisy_low_dir, noisy_high_dir]):
                    print(
                        f"Skipping {kvp_low}/{kvp_high} due to missing directories")
                    continue

                clean_low_files = get_sorted_dicoms(clean_low_dir)
                clean_high_files = get_sorted_dicoms(clean_high_dir)
                noisy_low_files = get_sorted_dicoms(noisy_low_dir)
                noisy_high_files = get_sorted_dicoms(noisy_high_dir)

                pair_count = min(len(clean_low_files), len(clean_high_files),
                                 len(noisy_low_files), len(noisy_high_files),
                                 MAX_SLICES)

                if pair_count == 0:
                    print(f"  No matching file count in {kvp_low}/{kvp_high}")
                    continue

                for i in range(pair_count):
                    clean_low_file = clean_low_files[i]
                    clean_high_file = clean_high_files[i]
                    noisy_low_file = noisy_low_files[i]
                    noisy_high_file = noisy_high_files[i]

                    for method_name, method_fn in [("hunemohr", hunemohr)]:
                        method_results = run_methods(
                            i,
                            clean_high_file,
                            clean_low_file,
                            noisy_high_file,
                            noisy_low_file,
                            method_name,
                            method_fn,
                            prefix,
                            thickness,
                            kvp_low,
                            kvp_high,
                            noise_level,
                            radii,
                            phantom_type
                        )
                        if method_results:
                            results.extend(method_results)

    # Save output
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_path = Path(f"results/Hunemohr/Hunemohr-{phantom_type}-8/hunemohr_comparison_{radii}_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

if __name__ == "__main__":
    print("trying")
    process_upload("/Users/royaparsa/Desktop/Head-8/")

    series_clean = "/Users/royaparsa/Desktop/Head-8/"
    series_noisy = "/Users/royaparsa/Desktop/noise-benchmark/test_images"

    phantom_type = "Head"
    radii_ratio = [0.25, 0.5, 0.75, 1]

    print("Indexing series...")
    clean_idx = index_series_by_kvp(series_clean)
    noisy_idx = index_series_by_kvp(series_noisy)

    print("CLEAN keys found:", list(clean_idx.keys()))
    print("NOISY keys found:", list(noisy_idx.keys()))
    print()

    # Now run the test harness
    for radi in radii_ratio:
        test(series_clean, series_noisy, phantom_type, radi)
    
    shutil.rmtree("test_images")