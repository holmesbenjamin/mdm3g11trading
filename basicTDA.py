import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

def time_delay_embedding(
    series: np.ndarray, 
    embedding_dim: int = 3, 
    delay: int = 1
) -> np.ndarray:
    n_points = len(series)
    max_index = n_points - (embedding_dim - 1) * delay
    
    if max_index <= 0:
        raise ValueError("Time series too short for requested embedding_dim and delay.")
    embedded = np.zeros((max_index, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = series[i * delay : i * delay + max_index]
    return embedded

def main():
    df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    commodity_name = "NATURAL GAS"
    if commodity_name not in df.columns:
        raise ValueError(f"Commodity '{commodity_name}' not found in CSV columns.")
    series = df[commodity_name].dropna()
    data_array = series.values
    data_diff = np.diff(data_array) 
    
    embedding_dim = 3  # how many delayed copies
    delay = 50          # time lag between samples
    embedded_points = time_delay_embedding(data_diff, embedding_dim, delay)
    
    # persistent homology with Ripser
    results = ripser(embedded_points)
    diagrams = results['dgms']  
    
    plot_diagrams(diagrams, show=True)
    plt.title(f"Persistence Diagrams for {commodity_name} (Embedding Dim={embedding_dim}, Delay={delay})")

if __name__ == "__main__":
    main()
