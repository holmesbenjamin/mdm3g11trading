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
    "figure.titlesize": 20,
    "axes.titlesize": 0,
})
import os

output_dir = "images/results"
os.makedirs(output_dir, exist_ok=True)

def load_data(csv_path="datasets/results/FINALbacktest_results_summary_all.csv"):
    df = pd.read_csv(csv_path)
    return df

def extract_stage_number(stage_str):
    return int(stage_str.split()[-1])

def plot_accuracy_vs_stages(df, commodity):
    df_comm = df[df["Commodity"] == commodity]
    
    grp = df_comm.groupby(["Stage", "Mode"])["Mean Accuracy"].mean().reset_index()
    
    grp["Stage_num"] = grp["Stage"].apply(extract_stage_number)
    grp = grp.sort_values("Stage_num")
    
    plt.figure(figsize=(8, 6))
    
    for mode in grp["Mode"].unique():
        subset = grp[grp["Mode"] == mode]
        plt.plot(subset["Stage_num"], subset["Mean Accuracy"], marker="o", label=mode)
    
    plt.xticks(grp["Stage_num"].unique(), [f"Stage {int(x)}" for x in grp["Stage_num"].unique()])
    plt.xlabel("Stage")
    plt.ylabel("Mean Accuracy")
    plt.title(f"{commodity} - Accuracy vs Stages")
    plt.legend(title="Strategy")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"accuracy_vs_stages_{commodity.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_cum_returns_vs_stages(df, commodity):
    df_comm = df[df["Commodity"] == commodity]
    
    df_comm["Stage_num"] = df_comm["Stage"].apply(lambda x: int(x.split()[-1]))
    
    plt.figure(figsize=(8, 6))
    
    timeframes = ["daily", "monthly"]
    modes = ["buy_hold", "buy_short"]
    
    for timeframe in timeframes:
        for mode in modes:
            subset = df_comm[(df_comm["Timeframe"] == timeframe) & (df_comm["Mode"] == mode)]
            grp = subset.groupby(["Stage", "Stage_num"])["Cumulative Return"].mean().reset_index()
            grp = grp.sort_values("Stage_num")
            if grp.empty:
                continue  
            cum_returns = grp["Cumulative Return"]
            plt.plot(grp["Stage_num"], cum_returns, marker="o", label=f"{timeframe} {mode}")
    
    stages_sorted = sorted(df_comm["Stage_num"].unique())
    plt.xticks(stages_sorted, [f"Stage {int(x)}" for x in stages_sorted])
    plt.xlabel("Stage")
    plt.ylabel("Cumulative Sum of Returns")
    plt.title(f"{commodity} - Cumulative Returns vs Stages")
    plt.legend(title="Strategy")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"cum_returns_vs_stages_{commodity.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
def main():
    df = load_data()
    commodities = df["Commodity"].unique()
    for commodity in commodities:
        plot_accuracy_vs_stages(df, commodity)
        plot_cum_returns_vs_stages(df, commodity)
    print("All plots saved in the images/results directory.")

if __name__ == "__main__":
    main()