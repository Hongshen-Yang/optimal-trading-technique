# Optimal Trading Technique

This repository contains the open-source code for the paper *Optimal Market-Neutral Currency Trading on the Cryptocurrency Platform*[^1] 

[^1]: Yang, H., & Malik, A. (2024). Optimal market-neutral currency trading on the cryptocurrency platform (arXiv:2405.15461). arXiv. https://doi.org/10.48550/arXiv.2405.15461

## Overview
This code implements an optimal market-neutral currency trading strategy for cryptocurrencies. The approach involves:

* Building optimization models for portfolio management using Gurobi.
* Simulating trading with different risk and return scenarios.
* Evaluating strategies using various financial metrics.

## Core algorithm
![optimisation](https://github.com/user-attachments/assets/d26242d7-2ffa-44ed-b8a9-4267a1136186)

- $N$: set of all pairs
- $w_n$: weight vector for the n-th pair
- $\text{EP}_n$: expected profit vector for the n-th pair
- $\odot$: Hadamard product
- $\lambda$: risk aversion coefficient
- $\text{cov}_n$: covariance matrix for the n-th pair
- $w_{n,\text{long}}$: long weight for the n-th pair
- $w_{n,\text{short}}$: short weight for the n-th pair
- $\text{TW}_c$: trading weight for currency $c$
- $c$: currency
- $tc$: transaction cost
- $P_{c,t}$: price of currency $c$ at time $t$
- $T$: set of all time periods

## Key Parameters
* `CROSSING_MEAN`: Mean crossing threshold.
* `CROSSING_MAX`: Maximum crossing threshold.
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
