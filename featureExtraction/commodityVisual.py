import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,                
    "font.family": "serif",            
    "font.serif": ["Computer Modern"], 
    "axes.labelsize": 20,              
    "xtick.labelsize": 20,             
    "ytick.labelsize": 20,             
    "legend.fontsize": 20,            
    "figure.titlesize": 0,             
    "axes.titlesize": 0,              
})

df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
commodities = df.columns
for commodity in commodities:
    if commodity == 'WTI CRUDE' or commodity == "BRENT CRUDE": #set commodities to compare
        if df[commodity].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df[commodity], label=commodity, color = 'black' ,marker=',', linestyle='-')
        ax.set_title(f"{commodity} Over Time")
        ax.set_xlabel("Year")
        plt.xticks(rotation=45)
        ax.set_ylabel("Price/Value")
        ax.xaxis.set_major_locator(mdates.YearLocator())           
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))   
        ax.grid(True)
        ax.legend()
    
    plt.show()
