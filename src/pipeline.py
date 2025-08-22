import os
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def load_data(ticker,period="2y", interval="1d",data_path=None):
    if not os.path.exists(data_path) or data_path is None:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, period=period, interval=interval)
        
        if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
        print("Data downloaded successfully.")
        # if data_path:
        #     os.makedirs(os.path.dirname(data_path), exist_ok=True)
        #     data.to_csv(data_path)
        #     print(f"Data saved to {data_path}")
        # else:
        #     print("No data_path provided, data not saved.")
    else:
        print(f"Data for {ticker} already exists at {data_path}. Loading from file.")
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    return data

def create_flags(data, lags=[1,2,3,5], MA_periods=[5, 20, 100], vol_windows=[10, 20, 60]):
    
    # Create Lagged Returns
    data['Returns'] = data['Close'].pct_change()

    # creates lagged returns for specified lags
    for lag in lags:
        data[f'Lagged_Returns_{lag}'] = data['Returns'].shift(lag)
    
    # Create Moving Averages
    for period in MA_periods:
        data[f"MA_{period}"] = data['Close'].rolling(window = period).mean()
        
    # Create RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Create MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
    
    # Create Bollinger Bands 
    sma_20 = data['Close'].rolling(window=20).mean()
    upper_band = sma_20 + 2 * data['Close'].rolling(window=20).std()
    lower_band = sma_20 - 2 * data['Close'].rolling(window=20).std()
    data['Upper_Band'] = upper_band
    data['Lower_Band'] = lower_band
    
    # Create Volatility
    for window in vol_windows:
        data[f'Volatility_{window}'] = data['Close'].rolling(window=window).std()
    
    # Create Target Variable
    data['Target_Classification'] = (data['Returns'].shift(-1) > 0).astype(int)
    data['Target_Regression'] = data['Returns'].shift(-1)

def create_x_y(data, flags = ['Close', 'Lagged_Returns_1', 'Lagged_Returns_2', 'Lagged_Returns_3', 'Lagged_Returns_5',
              'MA_5', 'MA_20', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist',
              'Upper_Band', 'Lower_Band', 'Volatility_10', 'Volatility_20']):
    x_unclean = data[flags]
    y_unclean_classification = data['Target_Classification']
    y_unclean_regression = data['Target_Regression']

    combined_classification = pd.concat([x_unclean, y_unclean_classification.rename('Target')], axis=1).dropna()
    combined_regression = pd.concat([x_unclean, y_unclean_regression.rename('Target')], axis=1).dropna()

    x_clean_classification = combined_classification.drop(columns=['Target'],axis=1)
    y_clean_classification = combined_classification['Target']
    x_clean_regression = combined_regression.drop(columns=['Target'], axis=1)
    y_clean_regression = combined_regression['Target']
    
    return x_clean_classification, y_clean_classification, x_clean_regression, y_clean_regression

def split_data(x_clean_classification, y_clean_classification, x_clean_regression, y_clean_regression):
    x_train_class, x_test_class, y_train_class, y_test_class = train_test_split(
    x_clean_classification, y_clean_classification, test_size=0.2, shuffle=False
    )

    x_test_class_index = x_test_class.index

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x_clean_regression, y_clean_regression, test_size=0.2, shuffle=False
    )

    x_test_reg_index = x_test_reg.index
    
    return x_train_class, x_test_class, y_train_class, y_test_class, x_test_class_index, \
           x_train_reg, x_test_reg, y_train_reg, y_test_reg, x_test_reg_index   
           
def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def print_model_stats_classification(model, x_test_class, y_test_class):
    predictions = model.predict(x_test_class)
    accuracy = accuracy_score(y_test_class, predictions)
    conf_matrix = confusion_matrix(y_test_class, predictions)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

def run_classification_model(model, x_train_class, x_test_class, y_train_class, y_test_class):
    #x_train_class, x_test_class = scale_data(x_train_class, x_test_class)
    model.fit(x_train_class, y_train_class)
    print_model_stats_classification(model, x_test_class, y_test_class)

def print_model_stats_regression(model, x_test_reg, y_test_reg):
    predictions = model.predict(x_test_reg)
    mse = mean_squared_error(y_test_reg, predictions)
    r2 = r2_score(y_test_reg, predictions)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

def run_regression_model(model, x_train_reg, x_test_reg, y_train_reg, y_test_reg):
    #x_train_reg, x_test_reg = scale_data(x_train_reg, x_test_reg)
    model.fit(x_train_reg, y_train_reg)
    print_model_stats_regression(model, x_test_reg, y_test_reg)
    

def classification_model(x_train_class, x_test_class, y_train_class, y_test_class):
    lm_classifier = LogisticRegression(max_iter=1000)
    x_train_class_lm, x_test_class_lm = scale_data(x_train_class, x_test_class)
    run_classification_model(lm_classifier, x_train_class_lm, x_test_class_lm, y_train_class, y_test_class)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    run_classification_model(rf_classifier, x_train_class, x_test_class, y_train_class, y_test_class)
    
    xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    run_classification_model(xgb_classifier, x_train_class, x_test_class, y_train_class, y_test_class)
    
    return rf_classifier, xgb_classifier, lm_classifier
    
def regression_model(x_train_reg, x_test_reg, y_train_reg, y_test_reg):
    lm_regressor = LinearRegression()
    x_train_reg_lm, x_test_reg_lm = scale_data(x_train_reg, x_test_reg)
    run_regression_model(lm_regressor, x_train_reg_lm, x_test_reg_lm, y_train_reg, y_test_reg)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    run_regression_model(rf_regressor, x_train_reg, x_test_reg, y_train_reg, y_test_reg)
    
    xgb_regressor = XGBRegressor(use_label_encoder=False, eval_metric='rmse')
    run_regression_model(xgb_regressor, x_train_reg, x_test_reg, y_train_reg, y_test_reg)
    
    return rf_regressor, xgb_regressor, lm_regressor

# def backtesting_classification(data, model, x_test_class, x_test_class_index):
#     y_pred_class = model.predict(x_test_class.values)
#     signals_class = pd.Series(y_pred_class, index=x_test_class_index)

#     signals_class = signals_class.shift(1).fillna(0)

#     returns = data['Returns'].loc[x_test_class_index] 

#     strategy_returns_class = signals_class * returns

#     initial_capital = 10000

#     portfolio_class = (strategy_returns_class + 1).cumprod() * initial_capital

#     buy_hold = (returns + 1).cumprod() * initial_capital
    
#     plt.figure(figsize=(10,6))
#     plt.plot(portfolio_class, label="Strategy (Classification)")
#     plt.plot(buy_hold, label="Buy & Hold")
#     plt.legend()
#     plt.show()

# def backtesting_regression(data, model, x_test_reg, x_test_reg_index):
#     y_pred_reg = model.predict(x_test_reg.values)

#     signals_reg = pd.Series((y_pred_reg > 0).astype(int), index=x_test_reg_index)

#     signals_reg = signals_reg.shift(1).fillna(0)

#     returns = data['Returns'].loc[x_test_reg_index] 

#     strategy_returns_reg = signals_reg * returns

#     initial_capital = 10000

#     portfolio_reg = (strategy_returns_reg + 1).cumprod() * initial_capital

#     buy_hold = (returns + 1).cumprod() * initial_capital

#     plt.figure(figsize=(10,6))
#     plt.plot(portfolio_reg, label="Strategy (Regression)")
#     plt.plot(buy_hold, label="Buy & Hold")
#     plt.legend()
#     plt.show()

def maybe_flip_signals(returns, signals):
    strategy_returns = signals * returns
    cum_return = (strategy_returns + 1).prod()
    
    flipped_strategy_returns = (-signals) * returns
    cum_flipped_return = (flipped_strategy_returns + 1).prod()
    
    if cum_flipped_return > cum_return:
        return -signals
    else:
        return signals
    
def backtest_strategy(data, model, x_test, x_test_index, mode="classification", initial_capital=10000):
    
    #y_pred = model.predict(x_test.values)
    y_pred = model.predict(x_test)
    
    if mode == "classification": 
        signals = pd.Series(y_pred, index=x_test_index).shift(1).fillna(0)
        signals = signals.replace(0, -1)
        signals = maybe_flip_signals(data['Returns'].loc[x_test_index], signals)
    elif mode == "regression":
        signals = pd.Series(y_pred, index=x_test_index).shift(1).fillna(0)
        signals = signals.apply(lambda x: 1 if x > 0 else -1)
        signals = maybe_flip_signals(data['Returns'].loc[x_test_index], signals)
    else:
        raise ValueError("mode must be 'classification' or 'regression'")
    
    returns = data['Returns'].loc[x_test_index]
    
    strategy_returns = signals * returns
    
    portfolio = (strategy_returns + 1).cumprod() * initial_capital
    buy_hold = (returns + 1).cumprod() * initial_capital
    
    plt.figure(figsize=(10,6))
    plt.plot(portfolio, label=f"Strategy ({mode})")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.legend()
    plt.show()
    
    return portfolio


def backtesting_combined(model_class, model_reg, x_test_class, x_test_class_index, x_test_reg, x_test_reg_index, data):
    y_pred_class = model_class.predict(x_test_class)  # Your trained classification model
    signals_class = pd.Series(y_pred_class, index=x_test_class_index)

    signals_class = signals_class.replace(0, -1)
    
    y_pred_reg = model_reg.predict(x_test_reg)

    signals_reg = pd.Series((y_pred_reg > 0).astype(int), index=x_test_reg_index)

    signals_reg = signals_reg.shift(1).fillna(0)
    
    signals_reg_direction = signals_reg.apply(lambda x: 1 if x > 0 else -1)

    signals_reg_direction.index = signals_class.index

    combined_signals = np.where(signals_class == signals_reg_direction, signals_class, 0)
    combined_signals = pd.Series(combined_signals, index=x_test_class_index)

    combined_signals = combined_signals.shift(1).fillna(0)

    returns = data['Returns'].loc[combined_signals.index]
    strategy_returns_combined = combined_signals * returns

    initial_capital = 10000
    portfolio_combined = (strategy_returns_combined + 1).cumprod() * initial_capital
    buy_hold = (returns + 1).cumprod() * initial_capital

    plt.figure(figsize=(10,6))
    plt.plot(portfolio_combined, label="Strategy (Combined)")
    plt.plot(buy_hold, label="Buy & Hold")
    plt.legend()
    plt.title("Combined Model Backtest")
    plt.show()
    
    return portfolio_combined


def sharpe_ratio(portfolio_values):
    daily_returns = portfolio_values.pct_change().dropna()
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

def max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return drawdown.min()

def win_rate(portfolio_values):
    daily_returns = portfolio_values.pct_change().dropna()
    wins = (daily_returns > 0).sum()
    total = len(daily_returns)
    return wins / total

def model_metrics(model,portfolio_class, portfolio_reg, portfolio_combined):
    print(f"Model: {model}")
    
    sharpe_class = sharpe_ratio(portfolio_class)
    sharpe_reg = sharpe_ratio(portfolio_reg)
    sharpe_combined = sharpe_ratio(portfolio_combined)

    print("Sharpe - Classification:", sharpe_class)
    print("Sharpe - Regression:", sharpe_reg)
    print("Sharpe - Combined:", sharpe_combined)

    mdd_class = max_drawdown(portfolio_class)
    mdd_reg = max_drawdown(portfolio_reg)
    mdd_combined = max_drawdown(portfolio_combined)

    print("Max Drawdown - Classification:", mdd_class)
    print("Max Drawdown - Regression:", mdd_reg)
    print("Max Drawdown - Combined:", mdd_combined)

    win_class = win_rate(portfolio_class)
    win_reg = win_rate(portfolio_reg)
    win_combined = win_rate(portfolio_combined)

    print("Win Rate - Classification:", win_class)
    print("Win Rate - Regression:", win_reg)
    print("Win Rate - Combined:", win_combined)

def model_metrics_custom(model, portfolio):
    print(f"Model: {model.name}")
    
    sharpe = sharpe_ratio(portfolio)
    print("Sharpe Ratio:", sharpe)
    
    mdd = max_drawdown(portfolio)
    print("Max Drawdown:", mdd)
    
    win = win_rate(portfolio)
    print("Win Rate:", win)

def run_pipeline(ticker, period="2y", interval="1d", data_path=None, extra_models=None):
    models = ["rf", "xgb", "lm"]
    
    data_path = data_path or f"../data/{ticker.lower()}.csv"
    
    data = load_data(ticker, period, interval, data_path)
    
    create_flags(data)
    
    data.to_csv(data_path)
    print(f"Data saved to {data_path}")
    
    x_clean_classification, y_clean_classification, x_clean_regression, y_clean_regression = create_x_y(data)
    
    x_train_class, x_test_class, y_train_class, y_test_class, x_test_class_index, \
    x_train_reg, x_test_reg, y_train_reg, y_test_reg, x_test_reg_index = split_data(
        x_clean_classification, y_clean_classification, x_clean_regression, y_clean_regression
    )
    
    rf_classifier, xgb_classifier, lm_classifier = classification_model(x_train_class, x_test_class, y_train_class, y_test_class)
    classification_models = [rf_classifier, xgb_classifier, lm_classifier]
    
    rf_regressor, xgb_regressor, lm_regressor = regression_model(x_train_reg, x_test_reg, y_train_reg, y_test_reg)
    regression_models = [rf_regressor, xgb_regressor, lm_regressor]
    
    portfolio_class = []

    for model in classification_models:
        print(f"Model: {model.__class__.__name__}")
        portfolio_class.append(backtest_strategy(data, model, x_test_class, x_test_class_index))
    
    portfolio_reg = []
    
    for model in regression_models:
        print(f"Model: {model.__class__.__name__}")
        portfolio_reg.append(backtest_strategy(data, model, x_test_reg, x_test_reg_index, 'regression'))
    
    portfolio_combined = []

    for model_class, model_reg in zip(classification_models, regression_models):
        portfolio_combined.append(backtesting_combined(model_class, model_reg, x_test_class, x_test_class_index, x_test_reg, x_test_reg_index, data))
    
    for i in range(len(models)):
        model_metrics(models[i], portfolio_class[i], portfolio_reg[i], portfolio_combined[i])
    
    if(extra_models is not None):
        for model in extra_models:
            if(model.type == "classification"):
                model.fit(x_train_class, y_train_class)
                print_model_stats_classification(model, x_test_class, y_test_class)
                portfolio = backtest_strategy(data, model, x_test_class, x_test_class_index)
                model_metrics_custom(model, portfolio)
            elif(model.type == "regression"):
                model.fit(x_train_reg, y_train_reg)
                print_model_stats_regression(model, x_test_reg, y_test_reg)
                portfolio = backtest_strategy(data, model, x_test_reg, x_test_reg_index, mode="regression")
                model_metrics_custom(model, portfolio)
            else:
                raise ValueError("Model type must be 'classification' or 'regression'")
            