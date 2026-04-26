import cv2
import numpy as np
import pandas as pd # You might not even need this anymore!
from pathlib import Path

from .preprocess import TARGET_SIZE

DATA_DIR = Path("./input/processed")

NEEDED_CLASSES = [1, 12, 13, 14, 15, 17, 25, 27, 33, 40]

CLASS_NAMES = {
    1: "Speed limit (30km/h)", 
    12: "Priority road", 
    13: "Yield", 
    14: "Stop",
    15: "No vehicles", 
    17: "No entry", 
    25: "Road work", 
    27: "Pedestrians",
    33: "Turn right ahead", 
    40: "Roundabout mandatory"
}

# create a dictionary mapping: {1: 0, 12: 1, 13: 2, ..., 40: 9}
LABEL_MAP = {original_id: new_id for new_id, original_id in enumerate(NEEDED_CLASSES)}

def load_train_data():
    """
    Loads training images and labels by scanning the processed directories.
    Returns:
        tuple: (train_images, train_labels) as numpy arrays.
    """
    images = []
    labels = []
    train_dir = DATA_DIR / "Train"
    
    print("Loading Train data...")
    for class_id in NEEDED_CLASSES:
        class_folder = train_dir / str(class_id)
        
        if not class_folder.exists():
            print(f"  -> Warning: Folder {class_folder} not found, skipping.")
            continue
            
        # grab all common image files in this class folder dynamically
        for img_path in class_folder.glob("*.*"):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
                
            img = cv2.imread(str(img_path))
            
            if img is not None:
                # resize to correct size
                img = cv2.resize(img, TARGET_SIZE)
                # opencv reads in BGR by default, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(LABEL_MAP[class_id])
            else:
                print(f"  -> Warning: Could not read {img_path}")
                
    return np.array(images), np.array(labels)

def load_test_data():
    """
    Loads testing images and labels using Test.csv since test images 
    are not sorted into class folders.
    """
    images = []
    labels = []
    
    csv_path = DATA_DIR / "Test.csv"
    
    if not csv_path.exists():
        print(f"  -> Error: Could not find {csv_path}")
        return np.array([]), np.array([])
        
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        original_class = row["ClassId"]
        
        # skip this image if it's not one of the 10 classes we care about
        if original_class not in NEEDED_CLASSES:
            continue
            
        img_path = DATA_DIR / row["Path"]
        
        # if not img_path.exists():
        #      img_path = DATA_DIR / "Test" / Path(row["Path"]).name
             
        img = cv2.imread(str(img_path))
        
        if img is not None:
            # resize and convert, same thing as earlier
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images.append(img)
            labels.append(LABEL_MAP[original_class])
        else:
            print(f"  -> Warning: Could not read {img_path}")
            
    return np.array(images), np.array(labels)

def load_data():
    """
    Wrapper function to load both train and test data,
    works similarly to the tensorflow counterpart.
    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    train_data = load_train_data()
    test_data = load_test_data()
    
    return train_data, test_data

if __name__ == "__main__":
    pass