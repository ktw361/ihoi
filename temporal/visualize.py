from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_summaries(homans) -> plt.figure:
    """ homans: list of HO_forwarder """
    l = len(homans)
    num_cols = 5
    num_rows = l // num_cols + 1
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        sharex=True, sharey=True, figsize=(20, 20))
    idx = 0
    for idx, ax in enumerate(axes.flat, start=0):
        homan = homans[idx]
        img = homan.render_summary()
        ax.imshow(img)
        ax.set_axis_off()
        idx += 1
        if idx >= l:
            break
    
    plt.tight_layout()
    return fig