Stock-ML-Predictor

About

A Python machine learning pipeline for predicting short-term stock price movements and backtesting trading strategies. Supports both classification and regression approaches with multiple models, including Logistic Regression, Linear Regression, Random Forest, and XGBoost.

This project is designed as an introduction to quantitative finance and algorithmic trading using machine learning. The main goal is to learn the concepts and techniques behind stock price prediction, backtesting, and strategy evaluation. It is not intended for live trading or financial advice.

It was a way to:
Understand the workflow of a quantitative trading pipeline.
Experiment with different ML models for classification and regression.
Explore backtesting and key metrics like Sharpe ratio and drawdown.

Features
Data Handling: Download historical stock data using Yahoo Finance API or load from local CSV files.
Feature Engineering: Lagged returns and technical indicators.
Modeling: Train and test classification and regression models.
Backtesting: Evaluate strategies against a buy-and-hold benchmark.
Metrics: Compute Sharpe ratio, maximum drawdown, and win rate.
Combined Strategies: Merge classification and regression signals to improve robustness.

Results
Demonstrated on multiple stocks, including Apple, Google and Netflix.
Metrics tracked: Sharpe ratio, max drawdown, win rate.
Random Forest generally performs best across both classification and regression approaches.

Technologies
Python
pandas, numpy
scikit-learn
XGBoost
Matplotlib
Yahoo Finance API