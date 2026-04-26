from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from core.sklearn_model import SklearnBaseModel

class RandomForestModel(SklearnBaseModel):
    def __init__(self, name="Random_Forest"):
        super().__init__(name=name)

    def build(self, n_estimators=300, n_pca=150, **kwargs):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_pca, random_state=42)),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators, 
                min_samples_leaf=2, 
                n_jobs=-1, 
                random_state=42
            ))
        ])
        print(f"[{self.name}] Built pipeline (trees={n_estimators}, pca={n_pca})")