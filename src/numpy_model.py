import numpy as np

class numpy_model_reg():
    
    def __init__(self):
        self.type = "regression"
        self.name = "Numpy Model Class"
        self.beta = None
    
    def fit(self, X, y=None):
        X_aug = np.c_[np.ones(len(X)), X.values]
        XtX_inv = np.linalg.inv(X_aug.T @ X_aug)
        self.beta = XtX_inv @ (X_aug.T @ y.values)
    
    def predict(self, X):
        X_aug = np.c_[np.ones(len(X)), X.values]
        return X_aug @ self.beta