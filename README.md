# MDM3 Group 11 Trading

This repository contains Python code for performing Topological Data Analysis (TDA) on commodity futures data. The code calculates log-returns for selected commodities (e.g., GOLD and NATURAL GAS), constructs sliding-window point clouds, and computes persistent homology using the [ripser](https://github.com/scikit-tda/ripser.py) package. It then summarises the topology using persistence landscapes (via [giotto-tda](https://giotto-ai.github.io/giotto-tda/)) and computes L1 and L2 norms as well as Wasserstein distances between consecutive persistence diagrams. Spikes in these metrics are used to identify extreme events or potential regime shifts.

## Features

- **Data Preprocessing:** Reads a CSV file (`commodity_futures.csv`) containing commodity prices, parses dates, and calculates daily log-returns.
- **Sliding Window Analysis:** Constructs point clouds over a sliding window to capture the joint dynamics of the commodities.
- **Persistent Homology:** Computes persistence diagrams for dimension 1 using `ripser`.
- **Persistence Landscape:** Summarizes the topology via persistence landscapes and computes discrete L1 and L2 norms.
- **Wasserstein Distance:** Quantifies the change between consecutive windows by calculating the Wasserstein distance between persistence diagrams.
- **Visualisation:** Plots the evolution of the computed metrics over time, marking threshold levels for potential extreme events.

## Installation

Ensure you have Python 3.6 or above installed. Then, install the required packages using pip:

```bash
pip install numpy pandas matplotlib ripser persim giotto-tda
