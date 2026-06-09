import os
import time
import json
import numpy as np
import pandas as pd
from utils.data import load_data, load_test_data

from core.cnn.model import NumPyCNN
from core.cnn_tensor.model import TensorFlowCNN
from core.random_forest.model import RandomForestModel
from core.mlp.model import MLPModel
from core.knn.model import KNNModel

from utils.metrics import compute_metrics

# ============================================================
# CONFIGURATION
# ============================================================
MODELS = [
    NumPyCNN, 
    TensorFlowCNN, 
    RandomForestModel, 
    MLPModel, 
    KNNModel
]

SHARED_BUILD_ARGS = {
    "input_shape": (32, 32, 3), 
    "num_classes": 10,
    "n_pca": 150,
    "n_estimators": 50,  # reduced to speed up testing
    "k": 7,
    "hidden_layer_sizes": (128, 64) # simplified for quick tests
}

SHARED_TRAIN_ARGS = {
    "epochs": 3, 
    "batch_size": 32, 
    "learning_rate": 0.01
}

def main():
    print("--- Step 1: Loading data ---")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    x_train_norm = x_train.astype('float32') /  255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    classes = np.unique(y_train)

    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    results = []

    for ModelClass in MODELS:
        model = ModelClass()
        print(f"\n=============================================")
        print(f"Running Experiment for {model.name}")
        print(f"=============================================")
        
        # Build Model
        model.build(**SHARED_BUILD_ARGS)

        # Train Model
        start_time = time.time()
        
        # TensorFlowCNN and NumPyCNN have slightly different signatures maybe, but we can pass kwargs
        try:
            model.train(x_train_norm, y_train, **SHARED_TRAIN_ARGS)
        except Exception as e:
            print(f"Error training {model.name}: {e}")
            continue
            
        train_time = time.time() - start_time
        
        # Test Model & get metrics
        start_infer = time.time()
        try:
            y_proba = model.predict(x_test_norm)
            
            # Sklearn base predict returns probabilities if predict_proba is used
            # For TF CNN / np CNN it returns raw logits or softmax outputs
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                y_pred = np.argmax(y_proba, axis=1)
            else:
                y_pred = y_proba
                y_proba = None
                
        except Exception as e:
            print(f"Error predicting {model.name}: {e}")
            continue
            
        infer_time = time.time() - start_infer
        
        # Calculate complex metrics (using the metrics util)
        metrics = compute_metrics(y_test, y_pred, y_proba, train_time, infer_time, classes=classes)
        
        # Save model
        save_path = f"./saved_models/{model.name.lower().replace(' ', '_')}.pkl"
        try:
            if hasattr(model, 'model') and model.name == "TensorFlow_CNN":
                save_path = save_path.replace('.pkl', '.keras')
            model.save(save_path)
        except Exception as e:
            print(f"Error saving {model.name}: {e}")

        # Add generic model identifiers
        metrics["model_name"] = model.name
        metrics["save_path"] = save_path
        
        results.append(metrics)
        
        # Save incremental results to CSV
        df = pd.DataFrame(results)
        df.to_csv("./results/all_models_metrics.csv", index=False)
        print(f"\n>>> Saved metrics for {model.name} to ./results/all_models_metrics.csv")

    print("\n--- ALL EXPERIMENTS COMPLETED ---")
    print("Results summarize:")
    print(pd.DataFrame(results)[['model_name', 'accuracy', 'f1', 'train_time_sec', 'inference_time_sec']])

if __name__ == "__main__":
    main()
