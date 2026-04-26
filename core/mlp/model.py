from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from core.sklearn_model import SklearnBaseModel

class MLPModel(SklearnBaseModel):
    def __init__(self, name="MLP"):
        super().__init__(name=name)

    def build(self, hidden_layer_sizes=(512, 256, 128), n_pca=150, **kwargs):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_pca, random_state=42)),
            ('clf', MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                verbose=True
            ))
        ])
        print(f"[{self.name}] Built pipeline (layers={hidden_layer_sizes}, pca={n_pca})")

    def train(self, x_train, y_train, *args, **kwargs):
        # Call the parent class train to do the standard fitting
        super().train(x_train, y_train, *args, **kwargs)
        
        # Add our custom MLP readout!
        mlp_step = self.model.named_steps['clf']
        if hasattr(mlp_step, "loss_curve_"):
            n_iter = len(mlp_step.loss_curve_)
            final_loss = mlp_step.loss_curve_[-1]
            print(f"  [MLP] Converged in {n_iter} iterations, final loss={final_loss:.4f}")