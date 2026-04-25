import cv2
import numpy as np
import pandas as pd
from pathlib import Path

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
    Loads training images and labels from the processed directory.
    Returns:
        tuple: (train_images, train_labels) as numpy arrays.
    """
    csv_path = DATA_DIR / "Train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run process.py first.")
        
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = DATA_DIR / row["Path"]
        img = cv2.imread(str(img_path))
        
        if img is not None:
            # opencv reads in BGR by default, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            # map the classes to the 0-9 range
            mapped_label = LABEL_MAP[row["ClassId"]]
            labels.append(mapped_label)
            
    return np.array(images), np.array(labels)

def load_test_data():
    """
    Loads testing images and labels from the processed directory.
    Returns:
        tuple: (test_images, test_labels) as numpy arrays.
    """
    csv_path = DATA_DIR / "Test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Run process.py first.")
        
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = DATA_DIR / row["Path"]
        img = cv2.imread(str(img_path))
        
        if img is not None:
            # convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            
            # map the classes to the 0-9 range
            mapped_label = LABEL_MAP[row["ClassId"]]
            labels.append(mapped_label)
            
    return np.array(images), np.array(labels)

def load_data():
    """
    Wrapper function to load both train and test data,
    works simillarly to the tensorflow counterpart.
    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    train_data = load_train_data()
    test_data = load_test_data()
    
    return train_data, test_data

# no need for this probably
if __name__ == "__main__":
    pass