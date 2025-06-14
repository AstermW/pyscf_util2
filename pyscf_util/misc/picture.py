from urllib import robotparser

# import seaborn
# import pandas
import matplotlib.pyplot as plt

# import numpy


def draw_extra_pic(
    x: list,
    y: list,
    legend: list,
    line_prop: list,
    xlabel: str = "$E_{pt}^{(2)}/E_H$",
    ylabel: str = "$E_{tot}/E_H$",
    title="",
    width=16,
    height=9,
    fontsize_xylabel=18,
    fontsize_xytick=18,
    fontsize_title=18,
    fontsize_legend=18,
    save_name=None,
):
    plt.figure(figsize=(width, height))
    for id, x in enumerate(x):
        plt.plot(
            x,
            y[id],
            marker=line_prop[id]["marker"],
            markersize=line_prop[id]["markersize"],
            linewidth=line_prop[id]["linewidth"],
            label=legend[id],
        )
    plt.xlabel(xlabel, fontsize=fontsize_xylabel)
    plt.ylabel(ylabel, fontsize=fontsize_xylabel)
    plt.xticks(fontsize=fontsize_xytick)
    plt.yticks(fontsize=fontsize_xytick)
    plt.title(title, fontsize=fontsize_title)
    plt.legend(fontsize=fontsize_legend)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
