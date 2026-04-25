from pathlib import Path
import shutil
import pandas as pd

from .data import NEEDED_CLASSES

# NEEDED_CLASSES = [1, 12, 13, 14, 15, 17, 25, 27, 33, 40]

if __name__ == "__main__":
    # base paths
    base_dir = Path("./input/raw")
    out_dir = Path("./input/filtered")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv_path = base_dir / "Train.csv"
    test_csv_path = base_dir / "Test.csv"
    
    if train_csv_path.exists():
        print("Filtering Train.csv...")
        df_train = pd.read_csv(train_csv_path)
        df_train_filtered = df_train[df_train["ClassId"].isin(NEEDED_CLASSES)]
        df_train_filtered.to_csv(out_dir / "Train.csv", index=False)
    else:
        print(f"Warning: {train_csv_path} not found.")
        df_train_filtered = pd.DataFrame()
        
    if test_csv_path.exists():
        print("Filtering Test.csv...")
        df_test = pd.read_csv(test_csv_path)
        df_test_filtered = df_test[df_test["ClassId"].isin(NEEDED_CLASSES)]
        df_test_filtered.to_csv(out_dir / "Test.csv", index=False)
    else:
        print(f"Warning: {test_csv_path} not found.")
        df_test_filtered = pd.DataFrame()

    # filter and  copy training files
    print("Copying required Train folders...")
    for class_id in NEEDED_CLASSES:
        src_folder = base_dir / "Train" / str(class_id)
        dst_folder = out_dir / "Train" / str(class_id)
        
        if src_folder.exists():
            shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        else:
            print(f"  -> Missing Train folder: {src_folder}")

    # filter and copy validation / test files
    print("Copying required Test files...")
    if not df_test_filtered.empty:
        for _, row in df_test_filtered.iterrows():
            # path usually somethign like 'Test/00000.png'
            src_file = base_dir / row["Path"]
            dst_file = out_dir / row["Path"]
            
            if src_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            else:
                print(f"  -> Missing Test file: {src_file}")

    print(f"\nSuccess! All filtered data has been written to: {out_dir.resolve()}")