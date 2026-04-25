from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    Every custom model must inherit from this and implement these methods.
    """
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None

    @abstractmethod
    def build(self, *args, **kwargs):
        """Define the model architecture or initialize the algorithm."""
        pass

    @abstractmethod
    def train(self, x_train, y_train, *args, **kwargs):
        """Train the model on the provided training data."""
        pass

    @abstractmethod
    def predict(self, x):
        """Generate predictions for the given input data."""
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test, *args, **kwargs):
        """Evaluate the model's performance on the test set."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the model weights/architecture to disk."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the model weights/architecture from disk."""
        pass