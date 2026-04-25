import os
import numpy as np
from utils.data import load_data

# ============================================================
# CONFIGURATION BLOCK - SWAP MODELS HERE
# ============================================================
from core.cnn.model import NumPyCNN as ActiveModel
MODEL_SAVE_NAME = "gtsrb_numpy.pkl"
# ============================================================

def main():
    print("--- Step 1: Loading data ---")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    print(f"\n--- Step 2: Model Setup ({ActiveModel.__name__}) ---")
    model = ActiveModel()
    model.build(input_shape=(32, 32, 3), num_classes=10)

    print("\n--- Step 3: Training ---")
    model.train(x_train, y_train, epochs=5, batch_size=32, learning_rate=0.01)

    print("\n--- Step 4: Evaluation ---")
    model.evaluate(x_test, y_test)

    print("\n--- Step 5: Saving Model ---")
    os.makedirs("./saved_models", exist_ok=True)
    model.save(f"./saved_models/{MODEL_SAVE_NAME}")

if __name__ == "__main__":
    main()