import numpy as np
import matplotlib.pyplot as plt
import random
from utils.data import load_test_data, NEEDED_CLASSES, CLASS_NAMES

# ============================================================
# CONFIGURATION BLOCK - SWAP MODELS HERE
# ============================================================
# from core.cnn.model import NumPyCNN as ActiveModel
# from core.cnn_tensor.model import TensorFlowCNN as ActiveModel
# from core.random_forest.model import RandomForestModel as ActiveModel
# from core.mlp.model import MLPModel as ActiveModel
from core.knn.model import KNNModel as ActiveModel

# .pkl or .keras
MODEL_SAVE_NAME = "gtsrb_model_knn.pkl" 
# ============================================================


def main():
    print("--- Step 1: Loading test data and model ---")
    x_test, y_test = load_test_data()
    x_test_norm = x_test.astype('float32') / 255.0

    model = ActiveModel()
    model.load(f"./saved_models/{MODEL_SAVE_NAME}")

    print("\n--- Step 2: Testing---")
    num_samples = 10
    random_indices = random.sample(range(len(x_test)), num_samples)
    
    sample_images = x_test_norm[random_indices]
    sample_labels_true = y_test[random_indices]
    
    raw_predictions = model.predict(sample_images)
    sample_labels_pred = np.argmax(raw_predictions, axis=1)

    print("\n--- Step 3: Results ---")
    plt.figure(figsize=(20, 4))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_test[random_indices[i]])
        plt.axis('off')
        
        true_class_id = NEEDED_CLASSES[sample_labels_true[i]]
        pred_class_id = NEEDED_CLASSES[sample_labels_pred[i]]
        true_name = CLASS_NAMES[true_class_id]
        pred_name = CLASS_NAMES[pred_class_id]
        
        color = "green" if true_class_id == pred_class_id else "red"
        plt.title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=10)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()