# src/plotting.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curves(equity_dict: dict[str, pd.Series], title: str = "Equity Curves"):
    """
    equity_dict: { name -> equity Series }
    """
    plt.figure(figsize=(12, 6))
    for name, eq in equity_dict.items():
        plt.plot(eq.index, eq.values, label=name)

    plt.title(title)
    plt.ylabel("Equity (starting at 1.0)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdowns(equity_dict: dict[str, pd.Series], title: str = "Drawdowns"):
    """
    Plot drawdowns for each equity curve.
    Drawdown_t = equity_t / running_max_t - 1
    """
    plt.figure(figsize=(12, 6))
    for name, eq in equity_dict.items():
        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        plt.plot(drawdown.index, drawdown.values, label=name)

    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()