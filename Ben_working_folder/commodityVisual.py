import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv("hugh_working_folder/commodity_futures_adjusted/commodity_futures_percentage_returns.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
commodities = df.columns
for commodity in commodities:
    if commodity == 'GOLD': #set commodities to compare
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
