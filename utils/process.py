import cv2
from pathlib import Path
import shutil

TARGET_SIZE = (32, 32) 

if __name__ == "__main__":
    in_dir = Path("./input/filtered")
    out_dir = Path("./input/processed")
    
    if not in_dir.exists():
        print(f"Error: Input directory {in_dir} does not exist. Run filter.py first.")
        exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # copy csv files
    for csv_file in ["Train.csv", "Test.csv"]:
        src_csv = in_dir / csv_file
        dst_csv = out_dir / csv_file
        if src_csv.exists():
            shutil.copy2(src_csv, dst_csv)
            print(f"Copied {csv_file}")

    # gather all files
    image_paths = list(in_dir.rglob("*.png"))
    total_images = len(image_paths)
    
    print(f"Found {total_images} images to process. Resizing...")
    
    processed_count = 0
    for img_path in image_paths:
        # determine where this file should go in the output directory
        relative_path = img_path.relative_to(in_dir)
        dst_path = out_dir / relative_path
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # read, reisze and write to new loc
        img = cv2.imread(str(img_path))
        
        if img is not None:
            resized_img = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(str(dst_path), resized_img)
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"  -> Processed {processed_count}/{total_images} images...")
        else:
            print(f"  -> Warning: Could not read image {img_path}")

    print(f"\nSuccess! {processed_count} images resized and saved to: {out_dir.resolve()}")