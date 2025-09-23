import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from methods.hunemohr import hunemohr
from methods.tanaka import tanaka
from methods.saito import saito
from pathlib import Path
import os
from datetime import datetime
import json
import shutil

# ===== TwinBeam constants =====
DATA_LOCO = Path("test_images")
TWINBEAM_HIGH_DIRNAME = "twinbeamh"
TWINBEAM_LOW_DIRNAME = "twinbeaml"
TWINBEAM_THICKNESS = "2.0"
TWINBEAM_KVP = 120

# noise variances to apply
VAR = [0.01, 0.05, 0.1]
MAX_SLICES = 10

# --------- Degrade utilities ---------
def process_upload(series_path, out_root="test_images"):
    """
    Walk the given series_path and degrade every DICOM found into
    out_root/degraded-<subfolder>-<var>/file.dcm

    TwinBeam note: we expect two subfolders: twinbeamh, twinbeaml
    """
    series_path = Path(series_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(series_path):
        root = Path(root)

        # only process dicoms
        for filename in files:
            if filename.startswith('.') or filename == '.DS_Store':
                continue
            src_path = root / filename
            # write each variance into its own output subfolder
            for var in VAR:
                subfolder_name = root.name  
                out_dir = out_root / f"degraded-{subfolder_name}-{var}"
                out_dir.mkdir(parents=True, exist_ok=True)
                degrade_image(src_path, out_dir, var)


def degrade_image(file: str | Path, out_dir: str | Path, var: float):
    """
    Read a DICOM, add Gaussian noise, and save it as a DICOM in out_dir.
    """
    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dicom_data = pydicom.dcmread(str(file))
    image = dicom_data.pixel_array.astype(np.float32)

    # normalize and resize to a consistent frame, then flatten for noise application
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    image = cv2.resize(image, (512, 512))
    image = image.flatten()[None, :]  # shape (1, N)

    x, y = image.shape
    n = np.random.normal(loc=0.0, scale=np.sqrt(var), size=(x, y))
    degraded_image = image + n

    # Save temp PNG
    out_path_png = out_dir / f"{file.stem}.png"
    degraded_2d = degraded_image.reshape(512, 512)
    degraded_u8 = (np.clip(degraded_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path_png), degraded_u8)

    # Convert PNG → DICOM (secondary capture)
    img = cv2.imread(str(out_path_png), cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    pixel_array = img.astype(np.uint16)

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out_path_dcm = out_dir / f"{file.stem}.dcm"
    ds = FileDataset(str(out_path_dcm), {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    # minimal tags
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "OT"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # preserve slope/intercept if present
    if hasattr(dicom_data, "RescaleSlope"):
        ds.RescaleSlope = dicom_data.RescaleSlope
    if hasattr(dicom_data, "RescaleIntercept"):
        ds.RescaleIntercept = dicom_data.RescaleIntercept

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()

    # timestamp
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')

    ds.save_as(out_path_dcm)

    # cleanup png
    try:
        os.remove(out_path_png)
    except Exception:
        pass


def get_sorted_dicoms(directory: Path):
    return sorted([p for p in Path(directory).glob("*.dcm") if p.is_file()])

# --------- TwinBeam indexers ---------
def index_twinbeam_clean(root: str | Path):
    """
    Return a dict:
      { ('twinbeam', '2.0'): {'high': Path, 'low': Path} }
    from clean directories named exactly 'twinbeamh' and 'twinbeaml'.
    """
    root = Path(root)
    high_dir = root / TWINBEAM_HIGH_DIRNAME
    low_dir = root / TWINBEAM_LOW_DIRNAME
    index = {}
    if high_dir.is_dir() and low_dir.is_dir():
        index[("twinbeam", TWINBEAM_THICKNESS)] = {
            "high": high_dir,
            "low":  low_dir
        }
    return index


def index_twinbeam_noisy(root: str | Path):
    """
    Return a dict:
      { ('twinbeam', '2.0', noise): {'high': Path, 'low': Path} }
    from degraded directories named:
      degraded-twinbeamh-<var>, degraded-twinbeaml-<var>
    """
    root = Path(root)
    index = {}
    for var in VAR:
        high_dir = root / f"degraded-{TWINBEAM_HIGH_DIRNAME}-{var}"
        low_dir = root / f"degraded-{TWINBEAM_LOW_DIRNAME}-{var}"
        if high_dir.is_dir() and low_dir.is_dir():
            index[("twinbeam", TWINBEAM_THICKNESS, float(var))] = {
                "high": high_dir,
                "low":  low_dir
            }
    return index

# --------- Runner ---------
def run_methods(i, clean_high_file, clean_low_file, noisy_high_file, noisy_low_file,
                method_name, method_fn, prefix, thickness, kvp_low, kvp_high,
                noise_level, radii, phantom_type):
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


def test_twinbeam(series_clean_root, series_noisy_root, phantom_type, radii):
    clean_idx = index_twinbeam_clean(series_clean_root)
    noisy_idx = index_twinbeam_noisy(series_noisy_root)

    if not clean_idx:
        print("❌ No clean TwinBeam series found (need 'twinbeamh' and 'twinbeaml').")
        return
    if not noisy_idx:
        print("❌ No noisy TwinBeam series found under test_images.")
        return

    results = []
    for (prefix, thickness), pair_dirs in clean_idx.items():
        print(
            f"\n=== Starting TwinBeam TEST: phantom={prefix}, thickness={thickness}, radii={radii} ===")

        clean_high_dir = pair_dirs["high"]
        clean_low_dir = pair_dirs["low"]
        clean_high_files = get_sorted_dicoms(clean_high_dir)
        clean_low_files = get_sorted_dicoms(clean_low_dir)

        for (_, _, noise_level), noisy_pair_dirs in sorted(noisy_idx.items(), key=lambda x: x[0][2]):
            print(f"\n🔊 Noise level: {noise_level}")
            noisy_high_dir = noisy_pair_dirs["high"]
            noisy_low_dir = noisy_pair_dirs["low"]

            noisy_high_files = get_sorted_dicoms(noisy_high_dir)
            noisy_low_files = get_sorted_dicoms(noisy_low_dir)

            pair_count = min(len(clean_low_files), len(clean_high_files),
                             len(noisy_low_files), len(noisy_high_files), MAX_SLICES)
            if pair_count == 0:
                print("  No matching file count between clean/noisy TwinBeam sets.")
                continue

            for i in range(pair_count):
                clean_low_file = clean_low_files[i]
                clean_high_file = clean_high_files[i]
                noisy_low_file = noisy_low_files[i]
                noisy_high_file = noisy_high_files[i]

                for method_name, method_fn in [
                    ("hunemohr", hunemohr),
                    ("saito", saito),
                    ("tanaka", tanaka),
                ]:
                    res = run_methods(
                        i,
                        clean_high_file, clean_low_file,
                        noisy_high_file, noisy_low_file,
                        method_name, method_fn,
                        prefix, thickness,
                        TWINBEAM_KVP, TWINBEAM_KVP, 
                        noise_level, radii, phantom_type
                    )
                    if res:
                        results.extend(res)

    # Save output
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out_dir = Path(f"results/Hunemohr/Hunemohr-{phantom_type}-TwinBeam")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / \
        f"hunemohr_twinbeam_comparison_{radii}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ TwinBeam results saved to {out_path}")


# --------- Main ---------
if __name__ == "__main__":
    series_clean_root = "/Users/royaparsa/Desktop/TwinBeam-Head-2mm/"
    series_noisy_root = str(DATA_LOCO)

    phantom_type = "Body"         
    radii_ratio = [0.25, 0.5, 0.75, 1.0]

    print("Degrading TwinBeam series...")
    process_upload(series_clean_root)

    print("Indexing TwinBeam series...")
    clean_idx_dbg = index_twinbeam_clean(series_clean_root)
    noisy_idx_dbg = index_twinbeam_noisy(series_noisy_root)
    print("CLEAN keys found:", list(clean_idx_dbg.keys()))
    print("NOISY keys found:", list(noisy_idx_dbg.keys()))
    print()

    for r in radii_ratio:
        test_twinbeam(series_clean_root, series_noisy_root, phantom_type, r)

    shutil.rmtree(DATA_LOCO, ignore_errors=True)
