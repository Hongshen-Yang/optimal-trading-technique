# Optimal Trading Technique

This repository contains the open-source code for the paper *Optimal Market-Neutral Currency Trading on the Cryptocurrency Platform*[^1] 

[^1]: Yang, H., & Malik, A. (2024). Optimal market-neutral multivariate pair trading on the cryptocurrency platform. International Journal of Financial Studies, 12(3), 77. https://doi.org/10.3390/ijfs12030077

## Overview
This code implements an optimal market-neutral currency trading strategy for cryptocurrencies. The approach involves:

* Building optimization models for portfolio management using Gurobi.
* Simulating trading with different risk and return scenarios.
* Evaluating strategies using various financial metrics.

## Core algorithm
![opti](https://github.com/user-attachments/assets/539e18c9-8c73-400c-9e92-7d98d0566e06)

## Key Parameters
* `CROSSING_MEAN`: Position closing threshold.
* `CROSSING_MAX`: Position opening threshold.
* `ORIG_AMOUNT`: Original investment amount.
* `RISK_FREE_RATE`: Risk-free rate for calculations.
* `TX_COST`: Transaction cost.
* `LAMBDA`: Risk aversion parameter.

## Key Functions
* `build_prob_cons`: Builds the optimization model.
* `simulate_trade`: Simulates trading scenarios.
* `arbitrage_trade`: Main function to execute the trading strategy.

## Data Source
The data source comes from the [Kraken OHLCVT dataset](https://support.kraken.com/hc/en-us/articles/360047124832-Downloadable-historical-OHLCVT-Open-High-Low-Close-Volume-Trades-data).

Generally, you can use any data source that meets that format of OHLCVT (Open, High, Low, Close, Volume, Trades).
