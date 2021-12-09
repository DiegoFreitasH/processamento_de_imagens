import tkinter as tk
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def hsv2rgb(hsv):
    """ convert HSV to RGB color space

    :param hsv: np.ndarray
    :return: np.ndarray
    """

    hi = np.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb

class Histogram:

    def __init__(self, app=None) -> None:
        self.app = app
        
        self.root = tk.Toplevel()
        self.root.title('Histogram')
        self.image = app.modified_image_data
        self.color_mode = app.color_mode
        
        if self.color_mode == 'RGB':
            self.im_value = np.mean(self.image, axis=2)
        else:
            self.im_value = self.image

        tk.Label(self.root, text='Channels').grid(row=1, column=0, columnspan=8)
        
        if self.color_mode == 'RGB':
            self.colors = ['Value', 'Red', 'Green', 'Blue']
        elif self.color_mode == 'L':
            self.colors = ['Value']
        
        self.option = tk.StringVar()
        colors_controls = tk.OptionMenu(
            self.root,
            self.option,
            command=self.update_hist,
            *self.colors
        )
        self.option.set("Value")
        colors_controls.grid(column=1, row=1, columnspan=10)
        
        eq_controls = tk.Button(
            self.root,
            text='Equalize',
            command=self.equalize,
        )
        
        eq_controls.grid(row=11, column=0, columnspan=10, rowspan=6)
        
        figure = plt.figure(figsize=(5,4), dpi=100)
        
        self.ax = figure.add_subplot(111)
        self.chart = FigureCanvasTkAgg(figure, self.root)
        self.ax.hist(self.array_to_image(self.im_value).getdata())
        self.chart.get_tk_widget().grid(column=0, row=10, columnspan=10,padx=20, pady=20)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
    
    def update_hist(self, *event):
        opt = self.colors.index(self.option.get()) - 1
        if opt == -1:
            self.ax.cla()
            print('starting update')
            self.ax.hist(self.array_to_image(self.im_value).getdata())
            print('end update')
            self.chart.draw()
        else:
            self.ax.cla()
            self.ax.hist(self.array_to_image(self.image).getdata(opt), color=self.option.get())
            self.chart.draw()
    
    def array_to_image(self, array: np.ndarray) -> Image.Image:
            return Image.fromarray(np.uint8(np.clip(array * 255, 0, 255)))
    
    def to_bytes_matrix(self, image_data: np.ndarray) -> np.ndarray:
        return np.uint8(np.clip(image_data * 255, 0, 255))

    def equalize(self):
        if self.color_mode == 'L':
            pixel_matrix = self.to_bytes_matrix(self.image)
            image = Image.fromarray(pixel_matrix)
            
            hist = np.array(image.histogram())
            hist_prob = hist / (image.width * image.height)
            cum_hist = np.cumsum(hist_prob)
            
            color_lookup = np.uint8(np.clip(cum_hist * 255, 0, 255))
            self.image = np.array([[color_lookup[v]/255 for v in row] for row in pixel_matrix])
            self.im_value = self.image
        elif self.color_mode == 'RGB':
            hsv = rgb2hsv(self.image)
            pixel_matrix = self.to_bytes_matrix(hsv[:,:,2])
            image = Image.fromarray(pixel_matrix)
            
            hist = np.array(image.histogram())
            hist_prob = hist / (image.width * image.height)
            cum_hist = np.cumsum(hist_prob)
            
            color_lookup = np.uint8(np.clip(cum_hist * 255, 0, 255))
            equalized_v = np.array([[color_lookup[v]/255 for v in row] for row in pixel_matrix])
            hsv[:,:,2] = equalized_v
            self.image = hsv2rgb(hsv)
            self.im_value = np.mean(self.image, axis=2)
        
        if self.app != None:
            self.app.update_canvas(self.image)
        self.update_hist()

if __name__ == '__main__':
    Histogram()