import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def plot_backtesting(history: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(history, label=history.columns)
    plt.title("Backtesting of Strategies")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.show();