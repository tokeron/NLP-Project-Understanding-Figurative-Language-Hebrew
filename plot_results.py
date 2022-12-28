import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import argparse


def plot_results(df_path):
    # plot the results
    df = pd.read_csv(df_path)
    # keep only two first columns
    df = df.iloc[:, :2]
    # remove rows with nan
    df = df.dropna()
    # make a table of the results with matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    the_table.set_fontsize(24)
    the_table.scale(1, 3.8)

    cells = the_table.properties()["celld"]
    for i in range(0, int(len(cells) / 2)):
        cells[i, 0].set_text_props(ha="left")
        cells[i, 1].set_text_props(ha="left")

    for i in range (0, len(df.columns)):
        the_table.auto_set_column_width(i)

    # add title
    ax.set_title('Results on test set', fontsize=30)

    # tight layout
    plt.tight_layout()

    plt.show()




if __name__ == '__main__':
    df_path = '/home/tok/figurative-language/results.csv'
    plot_results(df_path)