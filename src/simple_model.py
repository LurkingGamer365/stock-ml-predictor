import numpy as np

class simple_model_class():
    
    def __init__(self):
        self.type = "classification"
        self.name = "Simple Model"
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        
        close = X['Close']
        ma_5 = X['MA_5']
        volatility_10 = X['Volatility_10']
        return np.where((close > ma_5) & (volatility_10 > 5), 1, -1)