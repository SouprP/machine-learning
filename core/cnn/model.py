import numpy as np
import pickle
from core.base_model import BaseModel

from core.cnn.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization

# ==========================================
# 1. Base Sequential Engine
# ==========================================
class Sequential(BaseModel):
    def __init__(self, layers=None, name="Sequential_Engine"):
        super().__init__(name=name)
        self.layers = layers if layers is not None else []
        
    def add(self, layer):
        self.layers.append(layer)

    def train(self, x_train, y_train, epochs=5, batch_size=32, learning_rate=0.01):
        x_train_np = np.transpose(x_train, (0, 3, 1, 2)) 
        num_samples = x_train_np.shape[0]
        
        print(f"[{self.name}] Starting Training...")
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for i in range(0, num_samples, batch_size):
                x_batch = x_train_np[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # forward Pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output, training=True)
                    
                # output is ALREADY softmaxed because of the final Dense activation
                predictions = output 
                
                pred_labels = np.argmax(predictions, axis=1)
                correct += np.sum(pred_labels == y_batch)
                
                batch_loss = -np.log(predictions[np.arange(len(y_batch)), y_batch] + 1e-8)
                total_loss += np.sum(batch_loss)
                
                # backward Pass (combining softmax + cross entropy derivatives)
                grad = predictions.copy()
                grad[np.arange(len(y_batch)), y_batch] -= 1
                grad /= len(y_batch)
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
                    
            avg_loss = total_loss / num_samples
            accuracy = correct / num_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    def predict(self, x):
        x_np = np.transpose(x, (0, 3, 1, 2))
        output = x_np
        for layer in self.layers:
            output = layer.forward(output, training=False)
        return output

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        pred_labels = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_labels == y_test)
        print(f"[{self.name}] Test Accuracy: {accuracy:.4f}")
        return accuracy

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.layers, f)
        print(f"[{self.name}] Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.layers = pickle.load(f)
        print(f"[{self.name}] Model loaded from {filepath}")

# ==========================================
# CNN using NumPy
# ==========================================
class NumPyCNN(Sequential):
    def __init__(self, name="NumPy_CNN"):
        super().__init__(name=name)

    def build(self, input_shape=(32, 32, 3), num_classes=10):
        
        self.layers = [
            # Block 1: feature extraction
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Block 2: deeper feature extraction
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Block 3: even deeper
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # classification head
            Flatten(),
            Dense(128, activation='relu'),

            # randomness to inputs 
            Dropout(0.5), 
            Dense(num_classes, activation='softmax')
        ]
        
        print(f"[{self.name}] Network successfully built.")