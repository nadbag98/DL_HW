import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def plot_result(results: dict, value_to_plot: str):
    """
    Plots value of results for each network depth as a function of epoch
    :param results: Dictionary with keys that are network depths, and values that
                    are dictionaries of value name (i.e. "Train Loss") to list of values
    :param value_to_plot: Specific result to plot
    :return:
    """
    fig = go.Figure()
    for depth, value_dicts in results.items():
        value_to_epoch = value_dicts[value_to_plot]
        fig.add_trace(
            go.Scatter(x=np.arange(1, len(value_to_epoch)+1), y=value_to_epoch, mode='lines+markers', name="N="+str(depth))
        )
    fig.update_layout(
        xaxis_title="Epoch #",
        yaxis_title="Value of " + value_to_plot
    )
    # fig.show()
    fig.write_image(value_to_plot + ".png")
