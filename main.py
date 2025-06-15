from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
import os
from pathlib import Path
from typing import Literal
import gdown
import tarfile


# ----------------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
UTILS_DIR = REPO_ROOT / "utils"
YOLO_DIR = REPO_ROOT / "yolov5"

# Paths
TRAIN_YOLO = UTILS_DIR / "train_yolo.py"
GEN_LABELS = UTILS_DIR / "generate_yolo_labels.py"
CROP_ALL   = UTILS_DIR / "crop_all.py"

# Default data locations
CCPD_ROOT             = REPO_ROOT / "data" / "CCPD2019"
CROPS_ROOT            = REPO_ROOT / "data" / "CCPD2019_crops"
GINUZZO_CHECKPOINT    = REPO_ROOT / "pdlpr" / "best.pt"
BASELINE_CHECKPOINT   = REPO_ROOT / "baseline" / "ocr_model.pth"

# ----------------------------------------------------------------------------
#  Small helpers
# ----------------------------------------------------------------------------

def ask_yes_no(prompt: str, default: bool | None = None) -> bool:
    suffix = " [Y/n]" if default else " [y/N]" if default is False else " [y/n]"
    while True:
        resp = input(f"{prompt}{suffix}: ").strip().lower()
        if not resp and default is not None:
            return default
        if resp in {"y", "yes"}:
            return True
        if resp in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def run_step(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None):
    print("\nRunning:", " ".join(map(str, cmd)))
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Command failed with exit-code {exc.returncode}. Aborting pipeline.")
        sys.exit(exc.returncode)

# ----------------------------------------------------------------------------
#  Pipeline steps
# ----------------------------------------------------------------------------

# Calls the yolo labels generator
def step_generate_labels(args):
    if not GEN_LABELS.exists():
        print("⚠️  generate_yolo_labels.py not found under utils/. Skipping.")
        return
    cmd = [sys.executable, str(GEN_LABELS),
           "--repo_root", str(REPO_ROOT),
           ]
    run_step(cmd)

# Calls the cropper
def step_crop_images(args):
    if not CROP_ALL.exists():
        print("⚠️  crop_all.py not found under utils/. Skipping.")
        return
    cmd = [sys.executable, str(CROP_ALL),
           "--repo_root", str(REPO_ROOT)
           ]
    run_step(cmd)

# model choice
def choose_model() -> Literal["baseline", "pdlpr"]:
    while True:
        choice = input("Choose model - [b]aseline or [p]dlpr: ").strip().lower()
        if choice in {"b", "baseline"}:
            return "baseline"
        if choice in {"p", "pdlpr"}:
            return "pdlpr"
        print("Please enter 'b' for baseline or 'p' for PDLPR.")

# Does inference
def step_inference(model_choice: Literal["baseline", "pdlpr"], args):
    """Run inference for either the pdlpr model or the baseline OCR-CTC."""
    # 1) Determina la cartella da analizzare
    if args.crop_dir:
        data_dir = pathlib.Path(args.crop_dir)
    elif args.subset:
        data_dir = CROPS_ROOT / f"{args.subset}_crops"
    else:
        data_dir = CROPS_ROOT
    print(f"📂 Running inference on: {data_dir}")

    # 2) PDLPR ───────────────────────────────────────────────────────────
    if model_choice == "pdlpr":
        ckpt = GINUZZO_CHECKPOINT
        if not ckpt.exists():
            print("❌ PDLPR checkpoint not found - aborting inference.")
            return
        print(f"🔍 PDLPR Inference Check:"
              f"\n  → data_root: {data_dir}"
              f"\n  → weights:   {ckpt}")

        cmd = [
            sys.executable, "-m", "pdlpr.eval",
            "--data_root",  str(data_dir),
            "--weights",    str(ckpt),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        run_step(cmd, cwd=REPO_ROOT, env=env)

    # 3) BASELINE ────────────────────────────────────────────────
    else:
        ckpt = BASELINE_CHECKPOINT
        if not ckpt.exists():
            print("⚠️  baseline checkpoint not found - aborting inference.")
            return

        print(f"🔍 Baseline Inference Check:"
            f"\n  → data_root: {data_dir}"
            f"\n  → weights:   {ckpt}")

        evaluate_py = REPO_ROOT / "baseline" / "evaluate.py"
        if not evaluate_py.exists():
            print("⚠️  baseline/evaluate.py non trovato - impossibile eseguire l'inferenza.")
            return

        # usa getattr: se il flag --batch non esiste prende 64
        batch_sz = getattr(args, "batch", 64)

        cmd = [
            sys.executable, str(evaluate_py),
            "--data_root", str(data_dir),
            "--weights",   str(ckpt),
            "--batch",     str(batch_sz),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        run_step(cmd, cwd=REPO_ROOT, env=env)


def step_download_dataset():
    GDRIVE_ID = "1pbHQFfrkHmHNe1qjDWEth3FfyJ9WHDDc"
    dataset_dir = REPO_ROOT / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / "CCPD2019.tar.xz"
    final_dataset_path = dataset_dir / "CCPD2019"

    if final_dataset_path.exists():
        print("✔️ Dataset already exists, skipping download and extraction.")
        return

    # --- Step 1: Download the dataset  ---
    print(f"⬇️ Downloading CCPD dataset from Google Drive into {archive_path}")
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, str(archive_path), quiet=False)

    if not archive_path.exists():
        print(f"❌ Archive file not found at {archive_path}. Please download it first.")
        return

    # --- Step 2: Extract the .tar.xz archive ---
    print(f"📦 Extracting {archive_path} to {dataset_dir}...")
    try:
        with tarfile.open(archive_path, "r:xz") as txz:
            txz.extractall(path=dataset_dir)
        print("✅ Extraction complete.")
    except tarfile.ReadError as e:
        print(f"❌ Error reading the archive: {e}")
        if archive_path.exists():
            os.remove(archive_path) # Clean up corrupted file
        return
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return

    # --- Step 3: Clean up the archive file ---
    if archive_path.exists():
        os.remove(archive_path)
        print(f"🧹 Removed archive file: {archive_path}")

    print(f"✅ Dataset is ready in {final_dataset_path}")


# Calls the base sub-dataset splitter
def step_split_base(args):
    splitter = UTILS_DIR / "base_splitter.py"
    if not splitter.exists():
        print(f"⚠️  base_splitter.py not found under {UTILS_DIR}, skipping split.")
        return
    
    source_directory = CCPD_ROOT / "ccpd_base"
    validation_directory = CCPD_ROOT / "ccpd_base_val"

    print("Splitting base into train/val (10% validation)…")
    cmd = [
        sys.executable,
        str(splitter),
        "--src_dir", str(source_directory),
        "--val_dir", str(validation_directory),
        "--val_ratio", "0.1" # You can also make this a script argument if needed
    ]
    run_step(cmd)

def step_train_model(model_choice: Literal["baseline", "pdlpr"], args):
    """Invoke the appropriate training script for the selected model."""
    # ------------------------------------------------------------------ PDLPR
    if model_choice == "pdlpr":
        cmd = [
            sys.executable, "-m", "pdlpr.train",
            "--data_root",  str(CROPS_ROOT / "ccpd_base_crops"),
            "--train_root", str(CROPS_ROOT / "ccpd_base_crops"),
            "--val_root",   str(CROPS_ROOT / "ccpd_base_val_crops"),
        ]
        env = os.environ.copy()
        run_step(cmd, cwd=REPO_ROOT, env=env)

    # ---------------------------------------------------------------- BASELINE
    else:
        baseline_train = REPO_ROOT / "baseline" / "train.py"
        if not baseline_train.exists():
            print("⚠️  baseline/train.py non trovato → skip")
            return

        cmd = [
            sys.executable, str(baseline_train),
            "--data_root",  str(CROPS_ROOT / "ccpd_base_crops"),
            "--train_root", str(CROPS_ROOT / "ccpd_base_crops"),
            "--val_root",   str(CROPS_ROOT / "ccpd_base_val_crops"),
        ]
        env = os.environ.copy()
        run_step(cmd, cwd=REPO_ROOT, env=env)


# Handles the sub-datasets
def choose_subset() -> str:
    """Interactively pick a CCPD subset suffix for inference."""
    options = [
        "ccpd_base_val", "ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn",
        "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ]
    print("Select sub-dataset for inference:")
    for idx, name in enumerate(options, start=1):
        print(f"  {idx}. {name}")
    while True:
        resp = input(f"Enter number (1-{len(options)}): ").strip()
        if resp.isdigit():
            i = int(resp)
            if 1 <= i <= len(options):
                return options[i-1]
        print("Please enter a valid number.")


# ----------------------------------------------------------------------------
#  Main CLI
# ----------------------------------------------------------------------------

# Help provided, also each arguments explained in the repo
def parse_args():
    p = argparse.ArgumentParser(description="Run full licence-plate pipeline interactively")
    p.add_argument("--non_interactive", action="store_true", help="Run all steps without prompts (use flags)")
    p.add_argument("--clone_yolo", action="store_true", help="Clone the YOLO repository before running other steps")
    p.add_argument("--train", action="store_true", help="Force YOLO training step")
    p.add_argument("--gen_labels", action="store_true", help="Force YOLO label generation step")
    p.add_argument("--crop", action="store_true", help="Force crop step")
    p.add_argument("--model", choices=["baseline", "pdlpr"], help="Model to use for inference in non-interactive mode")
    p.add_argument("--subset", choices=[
        "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn",
        "ccpd_np", "ccpd_rotate", "ccpd_tilt", "ccpd_weather"
    ], help="Run inference on a specific CCPD subset (suffix)")
    p.add_argument("--crop_dir", type=Path, help="Path to existing 48x144 crops (overrides default)")
    p.add_argument("--download", action="store_true", help="Download and extract CCPD dataset")

    return p.parse_args()


def main():
    args = parse_args()
    interactive = not args.non_interactive

    # 1. Download dataset
    if args.download or (interactive and ask_yes_no("Download and extract CCPD dataset?", default=False)):
        step_download_dataset()
        step_split_base(args)

    # 2. Generate YOLO labels
    if args.gen_labels or (interactive and ask_yes_no("Generate YOLO labels?", default=False)):
        step_generate_labels(args)

    # 3. Crop plate images
    if args.crop or (interactive and ask_yes_no("Crop plate images to 48x144?", default=False)):
        step_crop_images(args)

    # 4. Choose model
    if args.model:
        model_choice = args.model
    elif interactive:
        model_choice = choose_model()
    else:
        print("Invalid ")
        sys.exit(1)

    # 5. Ask whether to train the chosen model
    if interactive and ask_yes_no(f"Do you want to train the {model_choice} model?", default=False):
        step_train_model(model_choice, args)

    # 6. Ask whether to run inference
    if interactive and ask_yes_no("Do you want to run inference with the chosen model?", default=True):
        # choose subset to run on
        if interactive:
            args.subset = choose_subset()
        step_inference(model_choice, args)

    print("✔️  All done. Exiting.")
    return


if __name__ == "__main__":
    main() 

