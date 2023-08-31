import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = go.Figure()
fig = make_subplots(rows=1, cols=3, subplot_titles=("a) brak nakładania się klas", "b) częściowe nakładanie się klas", "c) przykłady należą do tego samego rozkładu"))

for i, a in enumerate([0.8, 0.3, 0]):
    data1 = np.random.multivariate_normal(a*np.array([0, 5]), [[1, 0], [0, 1]], size=300)
    data2 = np.random.multivariate_normal(a*np.array([-5, -2.5]), [[1, 0], [0, 1]], size=300)
    data3 = np.random.multivariate_normal(a*np.array([5, -2.5]), [[1, 0], [0, 1]], size=300)

    fig.add_trace(go.Scatter(x=data1[:, 0], y=data1[:, 1],
                             mode='markers', name='klasa 1',
                             showlegend=i == 1, line_color='red'), 1, i+1)
    fig.add_trace(go.Scatter(x=data2[:, 0], y=data2[:, 1],
                             mode='markers', name='klasa 2',
                             showlegend=i == 1, line_color='blue'), 1, i+1)
    fig.add_trace(go.Scatter(x=data3[:, 0], y=data3[:, 1],
                             mode='markers', name='klasa 3',
                             showlegend=i == 1, line_color='green'), 1, i+1)
fig.update_layout(height=400, width=1200)
fig.write_image("plots/overlapping.jpg")

fig = go.Figure()

data1 = np.random.multivariate_normal(np.array([0, 5]), [[1, 0], [0, 1]], size=100)
data2 = np.random.multivariate_normal(np.array([-5, -2.5]), [[1, 0], [0, 1]], size=30)
data2 = np.concatenate([data2, np.random.multivariate_normal(np.array([-1, 4]), [[0.2, 0], [0, 0.2]], size=10)])
data2 = np.concatenate([data2, np.random.multivariate_normal(np.array([4, 0]), [[0.2, 0], [0, 0.2]], size=10)])
data3 = np.random.multivariate_normal(np.array([5, -2.5]), [[1, 0], [0, 1]], size=100)

fig.add_trace(go.Scatter(x=data1[:, 0], y=data1[:, 1],
                         mode='markers', name='klasa 1',
                         showlegend=i == 1, line_color='red'))
fig.add_trace(go.Scatter(x=data2[:, 0], y=data2[:, 1],
                         mode='markers', name='klasa 2',
                         showlegend=i == 1, line_color='blue'))
fig.add_trace(go.Scatter(x=data3[:, 0], y=data3[:, 1],
                         mode='markers', name='klasa 3',
                         showlegend=i == 1, line_color='green'))
fig.update_layout(height=400, width=400)
fig.write_image("plots/subconcepts.jpg")
