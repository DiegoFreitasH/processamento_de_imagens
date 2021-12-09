import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class CurveEditor:

    def __init__(self, app=None) -> None:
        self.root = tk.Toplevel()
        self.app = app

        self.points_x = [0, 1]
        self.points_y = [0, 1]

        tk.Label(self.root, text='X:').grid(column=0, row=1)
        self.input_x = tk.Entry(self.root, width=5)
        self.input_x.grid(column=1, row=1)
        
        tk.Label(self.root, text='Y:').grid(column=2, row=1)
        self.input_y = tk.Entry(self.root, width=5)
        self.input_y.grid(column=3, row=1)

        self.draw_curve = tk.Button(
            self.root,
            text="Add Point",
            command=self.add_point
        )
        self.draw_curve.grid(column=1, row=2, columnspan=2)
        apply_controls = tk.Button(
            self.root,
            text="Apply",
            command=self.apply
        )
        apply_controls.grid(column=3, row=2, columnspan=2)
        reset_controls = tk.Button(
            self.root,
            text="Reset",
            command=self.reset
        )
        reset_controls.grid(column=5, row=2, columnspan=2)

        if self.app.color_mode == 'L':
            self.channels = ['Value']
        elif self.app.color_mode == 'RGB':
            self.channels = ['Value', 'Red', 'Green', 'Blue']

        tk.Label(self.root, text='Channel:').grid(row=1, column=4, columnspan=2)
        self.active_channel = tk.StringVar()
        channel_controls = tk.OptionMenu(
            self.root,
            self.active_channel,
            *self.channels
        )
        self.active_channel.set('Value')
        channel_controls.grid(row=1, column=6)
        self.poly = self.get_polynomial_interpolation()
        figure = plt.Figure(figsize=(5,4), dpi=100)
        ax = figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(figure, self.root)
        self.line, = ax.plot(self.points_x, self.points_y)
        self.chart.get_tk_widget().grid(column=0, row=10, columnspan=10,padx=20, pady=20)
        self.root.mainloop()

    def add_point(self):
        try:
            x = float(self.input_x.get())
            y = float(self.input_y.get())
            x = max(min(x, 1), 0)
            y = max(min(y, 1), 0)
            if x in self.points_x:
                i = self.points_x.index(x)
                self.points_y[i] = y
            else:
                self.points_x.append(x)
                self.points_y.append(y)
            self.poly = self.get_polynomial_interpolation()
            self.update_curve()
        except Exception as e:
            print(e)

    def remove_last_points(self):
        if len(self.points_x > 2):
            self.points_x.pop()
            self.points_y.pop()
        
    def update_curve(self):
        x = np.linspace(0,1,30)
        y = self.poly(x)
        
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.chart.draw()
    
    def get_polynomial_interpolation(self):
        px = np.array(self.points_x)
        py = np.array(self.points_y)
        sorted_index = np.argsort(px)

        px = px[sorted_index]
        py = py[sorted_index]
        def poly(x):
            y = 0
            for i in range(len(px)):
                prod = py[i]
                for j in range(len(px)):
                    if j == i: continue
                    prod *= (x - px[j]) / (px[i] - px[j])
                y += prod
            return np.clip(y, 0, 1)
        
        return np.vectorize(poly)
    
    def reset(self):
        self.points_x = [0,1]
        self.points_y = [0,1]
        self.poly = self.get_polynomial_interpolation()
        self.update_curve()

    def apply(self):
        if self.active_channel.get() == 'Value':
            self.app.modified_image_data = self.poly(self.app.modified_image_data)
            self.app.image_data = self.app.modified_image_data
            self.app.update_canvas(self.app.modified_image_data)
        elif self.app.color_mode == 'RGB':
            channel = self.channels.index(self.active_channel.get()) - 1
            self.app.modified_image_data[:,:,channel] = self.poly(self.app.modified_image_data[:,:,channel])
            self.app.image_data = self.app.modified_image_data
            self.app.update_canvas(self.app.modified_image_data)
        
        self.root.destroy()

if '__name__' == 'main': 
    CurveEditor()