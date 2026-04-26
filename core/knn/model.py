from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from core.sklearn_model import SklearnBaseModel

class KNNModel(SklearnBaseModel):
    def __init__(self, name="kNN"):
        super().__init__(name=name)

    def build(self, k=7, n_pca=150, **kwargs):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_pca, random_state=42)),
            ('clf', KNeighborsClassifier(
                n_neighbors=k, 
                metric="euclidean", 
                weights="distance", 
                n_jobs=-1
            ))
        ])
        print(f"[{self.name}] Built pipeline (k={k}, pca={n_pca})")