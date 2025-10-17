import numpy as np
from sklearn.linear_model import Ridge

class RidgeReadout:
    """
    Simple ridge regression readout with scikit-learn API.
    """
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, random_state: int = 0):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_
