import os
from matplotlib import scale
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import Scale, filedialog
from tkinter.constants import HORIZONTAL
from PIL import ImageTk, Image
from numpy.core.fromnumeric import cumsum

IMG_DIRECTORY = '~/UFC/processamento_imagens/processing_project/img'
root = tk.Tk()

class MainApp(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title('Controls')

        self.image_data = np.array([])
        self.modified_image_data = np.array([])

        open_file = tk.Button(
            parent,
            text='Open Image',
            padx=10, pady=5,
            fg='white', bg='#263D42',
            command=self.open_file
        )
        open_file.pack(padx=10, pady=5)
        
        tk.Label(parent, text='Brightness').pack()
        self.brightness = tk.DoubleVar()
        brightness_controls = Scale(
            parent, 
            from_=0,
            to=200,
            orient=HORIZONTAL,
            variable=self.brightness,
            command=self.apply_changes # Real time change
        )
        brightness_controls.set(100)
        brightness_controls.pack(padx=10, pady=5)

        self.is_negative = tk.BooleanVar()
        negative_controls = tk.Checkbutton(
            parent,
            text='Negative',
            variable=self.is_negative,
            onvalue=True,
            offvalue=False,
            command=self.apply_changes
        )
        negative_controls.pack()

        self.apply_log = tk.BooleanVar()
        log_control = tk.Checkbutton(
            parent,
            text='Log2',
            variable=self.apply_log,
            onvalue=True,
            offvalue=False,
            command=self.apply_changes
        )
        log_control.pack()

        tk.Label(parent, text='Gamma').pack()
        self.gamma_value = tk.DoubleVar()
        gamma_control = tk.Scale(
            parent,
            variable=self.gamma_value,
            from_=1,
            to=200,
            orient=HORIZONTAL,
            command=self.apply_changes
        )
        gamma_control.set(100)
        gamma_control.pack()

        equalize_controls = tk.Button(
            parent,
            text='Equalize',
            padx=10, pady=5,
            fg='white', bg='#263D42',
            command=self.equalize_histogram
        )
        equalize_controls.pack()

        hist_controls = tk.Button(
            parent,
            text='Show Histogram',
            padx=10, pady=5,
            fg='white', bg='#263D42',
            command=self.show_histogram
        )
        hist_controls.pack()

        reset = tk.Button(
            parent,
            text='Reset Modifications',
            padx=10, pady=5,
            fg='white', bg='#263D42',
            command=self.reset_modifications
        )
        reset.pack(padx=20, pady=5)

        self.image_displayer = tk.Toplevel(self.parent)
        self.image_displayer.title('Image Displayer')
        
        # Changes default closing window behavior
        self.image_displayer.protocol("WM_DELETE_WINDOW", self.on_close_display)
        tk.Frame(self.image_displayer).pack()
        
        self.canvas = tk.Label(self.image_displayer)
        self.canvas.pack()

        self.image_displayer.withdraw() # Hides display at startup
    
    def open_file(self):
        filename = filedialog.askopenfilename(
            initialdir=IMG_DIRECTORY, 
            title="Select File",
            filetypes=(('all files', '*.*'), ('JPG images', '*.jpeg'), ('Tif images', '*.tif'))
        )

        try:
            self.image_data = self.get_image_array(filename, 'L')
            self.modified_image_data = self.image_data
            self.image_displayer.deiconify() # Unhide window when opening a new file
            self.update_canvas(self.image_data)
        except Exception as e:
            print('File Error:', e)
            print('Filename:', filename)


    def apply_changes(self, *args):
        if(len(self.image_data) == 0):
            return
        
        self.modified_image_data = self.image_data

        if(self.is_negative.get()):
            self.modified_image_data = 1 - self.modified_image_data
        
        if(self.apply_log.get()):
            self.modified_image_data = np.log2(1 + self.modified_image_data)
        
        self.modified_image_data = self.modified_image_data ** (self.gamma_value.get() / 100)
        self.modified_image_data = self.modified_image_data * self.brightness.get() / 100

        self.update_canvas(self.modified_image_data)
        
    def update_canvas(self, image_data: np.ndarray):
        img = ImageTk.PhotoImage(self.array_to_image(image_data))
        self.canvas.configure(image=img)
        self.canvas.image = img
    
    def reset_modifications(self):
        self.modified_image_data = self.image_data
        self.update_canvas(self.modified_image_data)

    def on_close_display(self):
        self.image_displayer.withdraw()

    def show_histogram(self):
        plt.hist(self.array_to_image(self.modified_image_data).getdata(), bins=40)
        plt.show()
    
    def equalize_histogram(self):
        pixel_matrix = self.to_bytes_matrix(self.modified_image_data)
        image = Image.fromarray(pixel_matrix)
        
        hist = np.array(image.histogram())
        hist_prob = hist / (image.width * image.height)
        cum_hist = np.cumsum(hist_prob)
        
        color_lookup = np.uint8(np.clip(cum_hist * 255, 0, 255))
        
        self.modified_image_data = np.array([[color_lookup[v]/255 for v in row] for row in pixel_matrix])
        self.update_canvas(self.modified_image_data)

    def get_image_array(self, filename: str, mode: str) -> np.ndarray:
        with Image.open(filename) as im:
            img_array = np.asarray(im.convert(mode))
            return img_array / 255

    def to_bytes_matrix(self, image_data: np.ndarray) -> np.ndarray:
        return np.uint8(np.clip(image_data * 255, 0, 255))

    def array_to_image(self, array: np.ndarray) -> Image.Image:
        return Image.fromarray(self.to_bytes_matrix(array))

app = MainApp(root)
root.mainloop()