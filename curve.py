import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

points = np.array([[0, 0], [.3, .9], [1, 1]])

data = np.linspace(0, 1, 50)

def get_interpolation_polynomial(points: np.ndarray):

    def poly(x):
        n = len(points)
        y = 0
        for i in range(n):
            prod = 1
            for j in range(n):
                if i == j: continue
                prod *= points[i, 1] * (x - points[j, 0])/(points[i, 0] - points[j, 0])
            y += prod
        return y
    
    return poly

poly = np.vectorize(get_interpolation_polynomial(points))

root = tk.Tk()
frame = tk.Frame(root)

fig = Figure()
ax = fig.add_subplot(111)

arr = np.array(points)
ax.scatter(arr[:, 0], poly(arr[:, 0]), marker='x')

graph = ax.plot(data, poly(data))
canvas = FigureCanvasTkAgg(fig, master=root)

canvas.draw()
canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
frame.pack()

root.mainloop()