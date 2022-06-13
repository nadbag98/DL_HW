import plotly.graph_objects as go
import numpy as np


def plot_result(vals_dict: dict, metric: str, path: str = "./results", title: str = ""):
    fig = go.Figure()
    for key, val in vals_dict.items():
        N = len(val)
        fig.add_trace(go.Scatter(x=np.arange(1, N + 1), y=val, mode='lines', name=key))

    fig.update_layout(
        xaxis_title="Epoch #",
        yaxis_title=metric.capitalize()
    )
    # fig.show()
    fig.write_image(path + '/' + title + ".png")
