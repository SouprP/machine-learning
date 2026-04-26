import cv2
import numpy as np
import shutil
import random
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
TARGET_SIZE = (32, 32)
TARGET_N = 1000  # Samples per class after augmentation
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==========================================
# Preprocessing Helpers
# ==========================================
def correct_exposure(img_rgb: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to 
    the L channel of the LAB colour space. Handles uneven lighting 
    region-by-region, which is much better than global equalisation.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def normalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Per-channel min-max stretch to [0, 255].
    Ensures consistent intensity range across images taken in different conditions.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        out[:, :, c] = (ch - mn) / (mx - mn) * 255.0 if mx > mn else ch
    return out.astype(np.uint8)

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. BGR -> RGB
      2. Resize to TARGET_SIZE
      3. CLAHE exposure correction
      4. Histogram normalisation
    Returns a uint8 RGB image.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img_rgb = correct_exposure(img_rgb)
    img_rgb = normalize_histogram(img_rgb)
    return img_rgb

def augment_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Lightweight augmentation used when a class has fewer than TARGET_N images.
    All ops are NumPy/OpenCV only - no extra dependencies.
    """
    img = img_rgb.copy()

    # Random horizontal flip
    if random.random() < 0.5:
        img = img[:, ::-1, :].copy()

    # Brightness jitter ±25%
    factor = random.uniform(0.75, 1.25)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Random crop + resize (simulates zoom / translation)
    H, W = img.shape[:2]
    crop = int(H * random.uniform(0.05, 0.15))
    if crop > 0:
        y1 = random.randint(0, crop)
        x1 = random.randint(0, crop)
        patch = img[y1 : H - crop + y1, x1 : W - crop + x1]
        img = cv2.resize(patch, (W, H), interpolation=cv2.INTER_AREA)

    return img

# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == "__main__":
    in_dir = Path("./input/filtered")
    out_dir = Path("./input/processed")

    if not in_dir.exists():
        print(f"[Error] Directory not found: {in_dir}")
        print("Please run the filtering script first.")
        exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting image processing & augmentation pipeline...")
    print(f"Targeting {TARGET_N} samples per class.\n")

    # --- 1. Copy CSV files ---
    for csv_file in ["Train.csv", "Test.csv"]:
        src_csv = in_dir / csv_file
        dst_csv = out_dir / csv_file
        if src_csv.exists():
            shutil.copy2(src_csv, dst_csv)
            print(f"[Info] Copied metadata: {csv_file}")

    # --- 2. Process Test Images (No Augmentation) ---
    test_dir_in = in_dir / "Test"
    test_paths = list(test_dir_in.rglob("*.png")) if test_dir_in.exists() else []
    
    if test_paths:
        print(f"\n[Processing] Extracting {len(test_paths)} Test Images...")
        for img_path in test_paths:
            dst = out_dir / img_path.relative_to(in_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            raw = cv2.imread(str(img_path))
            if raw is not None:
                processed = preprocess_image(raw)
                cv2.imwrite(str(dst), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    # --- 3. Process Train Images (Preprocess + Augment) ---
    train_dir_in = in_dir / "Train"
    if not train_dir_in.exists():
        print("\n[Warning] No Train/ folder found in filtered directory. Skipping training processing.")
    else:
        class_folders = sorted([d for d in train_dir_in.iterdir() if d.is_dir()])
        print(f"\n[Processing] Balancing & Augmenting {len(class_folders)} Class Folders...")

        for class_folder in class_folders:
            class_id = class_folder.name
            dst_class = out_dir / "Train" / class_id
            dst_class.mkdir(parents=True, exist_ok=True)

            img_paths = list(class_folder.rglob("*.png"))
            processed_imgs = []

            # Preprocess originals
            for img_path in img_paths:
                raw = cv2.imread(str(img_path))
                if raw is None:
                    continue
                processed = preprocess_image(raw)
                processed_imgs.append(processed)

            n_orig = len(processed_imgs)
            if n_orig == 0:
                continue

            # Augment if fewer than TARGET_N
            aug_count = 0
            while len(processed_imgs) < TARGET_N:
                src = random.choice(processed_imgs[:n_orig])
                processed_imgs.append(augment_image(src))
                aug_count += 1

            # Trim to exactly TARGET_N and shuffle
            random.shuffle(processed_imgs)
            processed_imgs = processed_imgs[:TARGET_N]

            # Save all to disk
            for i, img_rgb in enumerate(processed_imgs):
                out_path = dst_class / f"{i:05d}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            print(f" -> Class {class_id:>3s}: {n_orig:>4} original + {aug_count:>4} augmented = {len(processed_imgs)} total")

    print(f"\n[Success] Processed dataset ready at: {out_dir.resolve()}")