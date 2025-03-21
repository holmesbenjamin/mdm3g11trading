import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
import seaborn as sns

rcParams.update({
    "text.usetex": False,  
    "font.family": "serif",
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 0,
    "axes.titlesize": 0,
})
df = pd.read_csv("commodity_futures.csv", parse_dates=["Date"])
df = df.dropna(subset=["BRENT CRUDE"])
df["BRENT CRUDE"] = pd.to_numeric(df["BRENT CRUDE"], errors="coerce")
df.set_index("Date", inplace=True)
plt.figure(figsize=(2, 6))
box = plt.boxplot(df['BRENT CRUDE'], vert=True, patch_artist=True)
for element in ['boxes', 'whiskers', 'caps', 'medians']:
    plt.setp(box[element], color='black')  
for patch in box['boxes']:
    patch.set(facecolor='lightgrey')  
plt.ylabel('Price')
plt.title('Brent Crude Prices Boxplot')

plt.show()
