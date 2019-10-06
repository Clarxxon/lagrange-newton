import plotly
import plotly.graph_objects as go
import pandas as pd
import math
from tabulate import tabulate
import numpy as np


def func(x):
    return 2*math.sin(4*x)-math.cos(x)


LEFT_RANGE = -2
RIGHT_RANGE = 2
n = 6

x_arr = []
y_arr = []


for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, n):
    x_arr.append(i)
    y_arr.append(func(i))


coeffs = []


def l(x, x_table, n, i):
    tmp = 1
    for j in range(0, n):
        if(i != j):
            tmp *= (x-x_table[j])/(x_table[i]-x_table[j])
    return tmp


def lagrange(x, x_table, y_table):
    res = 0
    for i in range(0, n):
        res += y_table[i]*l(x, x_table, n, i)
    return res


# calc interpolated
interpolated = []
for i in np.linspace(-5, 5, 200):
    interpolated.append(lagrange(i, x_arr, y_arr))


# calc true
true_func = []
for i in np.linspace(-5, 5, 200):
    true_func.append(func(i))


# calc interpolated
interpolated_interval = []
for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, 100):
    interpolated_interval.append(lagrange(i, x_arr, y_arr))


# calc true
true_func_interval = []
for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, 100):
    true_func_interval.append(func(i))

# plot
fig = go.Figure()
x_long_array = np.linspace(LEFT_RANGE, RIGHT_RANGE, 100)
total_different = 0
for i in range(0, len(true_func_interval)):
    total_different += abs(true_func_interval[i]-interpolated_interval[i])

    fig.add_trace(go.Scatter(x=[x_long_array[i],x_long_array[i]] , y=[true_func_interval[i],interpolated_interval[i]],
                             mode='lines+markers',showlegend=False))


fig.add_trace(go.Scatter(x=[i for i in np.linspace(-5, 5, 200)], y=true_func,
                         mode='lines+markers',
                         name='2SIN(4x)-COS(X)'))
fig.add_trace(go.Scatter(x=[i for i in np.linspace(-5, 5, 200)], y=interpolated,
                         mode='lines+markers',
                         name='lagrange'))
fig.add_trace(go.Scatter(x=[x for x in x_arr if x is not None], y=[x for x in y_arr if x is not None],
                         mode='markers',
                         marker=dict(color='LightSkyBlue',size=20),
                         name='CONTROL POINTS'))
fig.update_yaxes(range=[-10, 10])
fig.update_layout(title_text="Метод Лагранжа, n="+str(n-1) +
                  ". Ошибка на интервале [-2,2] = "+str(total_different))
fig.show()
