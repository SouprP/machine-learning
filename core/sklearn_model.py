import pickle
import numpy as np
from core.base_model import BaseModel

class SklearnBaseModel(BaseModel):
    """
    A generic base class for all scikit-learn models.
    Handles flattening (H, W, C) images to 1D arrays and standardizing
    the train/evaluate/save/load loops using sklearn Pipelines.
    """
    def __init__(self, name="Sklearn_Model"):
        super().__init__(name=name)
        self.model = None  # Will be defined as a Pipeline in subclass build()

    def train(self, x_train, y_train, *args, **kwargs):
        print(f"[{self.name}] Training pipeline ...")
        # Flatten (N, H, W, C) to (N, Features)
        X_flat = x_train.reshape(len(x_train), -1)
        
        self.model.fit(X_flat, y_train)
        
        preds = self.model.predict(X_flat)
        acc = np.mean(preds == y_train)
        print(f"[{self.name}] Train accuracy: {acc:.4f}")

    def predict(self, x):
        X_flat = x.reshape(len(x), -1)
        return self.model.predict_proba(X_flat)

    def evaluate(self, x_test, y_test, *args, **kwargs):
        X_flat = x_test.reshape(len(x_test), -1)
        preds = self.model.predict(X_flat)
        acc = np.mean(preds == y_test)
        print(f"[{self.name}] Test Accuracy: {acc:.4f}")
        return acc

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[{self.name}] Pipeline saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        print(f"[{self.name}] Pipeline loaded from {filepath}")