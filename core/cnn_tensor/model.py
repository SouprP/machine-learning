import tensorflow as tf
from core.base_model import BaseModel

class TensorFlowCNN(BaseModel):
    def __init__(self, name="TensorFlow_CNN"):
        super().__init__(name=name)
        
    def build(self, input_shape=(32, 32, 3), num_classes=10, learning_rate=0.0001, **kwargs):
        """
        Builds and compiles the CNN architecture for TensorFlow.
        """
        self.model = tf.keras.Sequential([
            # Block 1: feature extraction
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Block 2: deeper feature extraction
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Block 3: even deeper
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            # classification head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),

            # randomness to inputs 
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # sparse_categorical_crossentropy is used because our labels are integers,
        # not one-hot encoded vectors.
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()

    def train(self, x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, **kwargs):
        """Trains the model with the given dataset."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build() first.")
            
        print(f"[{self.name}] Starting training...")
        
        # early stopping to halt training if the model stops improving
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping]
        )
        return history

    def predict(self, x):
        """Returns raw predictions (probabilities per class)."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build() first.")
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        """Evaluates model performance on unseen test data."""
        if self.model is None:
            raise ValueError("Model is not built yet. Call build() first.")
            
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"[{self.name}] Test Loss: {loss:.4f} - Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def save(self, filepath):
        """Saves the entire model (architecture, weights, optimizer state)."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"[{self.name}] Model saved to {filepath}")

    def load(self, filepath):
        """Loads a saved model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"[{self.name}] Model loaded from {filepath}")