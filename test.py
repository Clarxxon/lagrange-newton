import plotly
import plotly.graph_objects as go
import pandas as pd
import math
from tabulate import tabulate
import numpy as np


def func(x):
    return 2*math.sin(4*x)-math.cos(x)


def newton(x, d_array, x_arr):
    res = d_array[0]
    for i in range(1, len(d_array)):
        tmp = 1
        for j in range(0, i):
            tmp *= (x+x_arr[j]*-1)
        res += d_array[i]*tmp
    return res


LEFT_RANGE = -2
RIGHT_RANGE = 2
n = 6

x_arr = []
x_arr_no_none = []
y_arr = []


for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, n):
    x_arr.append(i)
    y_arr.append(func(i))
    x_arr_no_none.append(i)

    if(i != RIGHT_RANGE):
        x_arr.append(None)
        y_arr.append(None)

table = [x_arr, y_arr]
for i in range(2, n+1):
    table.append([None] * (2*n-1))


coeffs = [y_arr[0]]
for col in range(2, n+1):
    for row in np.arange(col-1, len(x_arr)-col+1, 2):
        row = int(row)

        y1 = table[col-1][row-1]
        y2 = table[col-1][row+1]

        x1 = table[0][row-col+1]
        x2 = table[0][row+col-1]

        table[col][row] = (y2-y1)/(x2-x1)

    coeffs.append(table[col][col-1])

print(tabulate(table, headers='keys', tablefmt='psql'))

# calc interpolated
interpolated = []
for i in np.linspace(-5, 5, 100):
    interpolated.append(newton(i, coeffs, x_arr_no_none))


# calc true
true_func = []
for i in np.linspace(-5, 5, 100):
    true_func.append(func(i))


# calc interpolated
interpolated_interval = []
for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, 100):
    interpolated_interval.append(newton(i, coeffs, x_arr_no_none))


# calc true
true_func_interval = []
for i in np.linspace(LEFT_RANGE, RIGHT_RANGE, 100):
    true_func_interval.append(func(i))

total_different = 0
for i in range(0, len(true_func_interval)):
    total_different += abs(true_func_interval[i]-interpolated_interval[i])

# plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=[i for i in np.linspace(-5, 5, 100)], y=true_func,
                         mode='lines+markers',
                         name='2SIN(4x)-COS(X)'))
fig.add_trace(go.Scatter(x=[i for i in np.linspace(-5, 5, 100)], y=interpolated,
                         mode='lines+markers',
                         name='newton'))
fig.add_trace(go.Scatter(x=[x for x in x_arr if x is not None], y=[x for x in y_arr if x is not None],
                         mode='markers',
                         marker=dict(color='LightSkyBlue',size=20),
                         name='CONTROL POINTS'))
fig.update_yaxes(range=[-10, 10])
fig.update_layout(title_text="Метод Ньютона, n="+str(n-1) +
                  ". Ошибка на интервале [-2,2] = "+str(total_different))
fig.show()
