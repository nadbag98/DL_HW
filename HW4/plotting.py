import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def plot_result(train_vals, test_vals, metric: str, path: str = "./results"):
    N = len(train_vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, N + 1), y=train_vals, mode='lines+markers', name="Train"))
    fig.add_trace(go.Scatter(x=np.arange(1, N + 1), y=test_vals, mode='lines+markers', name="Test"))

    fig.update_layout(
        xaxis_title="Epoch #",
        yaxis_title=metric.capitalize()
    )
    # fig.show()
    fig.write_image(path + '/train_and_test_' + metric + ".png")
