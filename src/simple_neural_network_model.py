from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor

class simple_MLP_regressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(32, 16), max_iter=300, learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.model = None
        self.type = "regression"
        self.name = "Simple MLP Regressor"
        
    def fit(self, X, y):
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            random_state=42
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
