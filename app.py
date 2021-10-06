import os
from sys import path_importer_cache
import numpy as np
import tkinter as tk
from tkinter import Scale, Text, filedialog, image_names
from tkinter.constants import HORIZONTAL
from PIL import ImageTk, Image

IMG_DIRECTORY = '~/UFC/processamento_imagens/processing_project/img'
root = tk.Tk()

class MainApp(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title('Controls')
        parent.geometry('200x300')

        self.image_data = []

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
            from_=.1,
            to=5,
            resolution=.1,
            orient=HORIZONTAL,
            command=self.apply_changes
        )
        gamma_control.set(1)
        gamma_control.pack()

        apply = tk.Button(
            parent,
            text='Apply',
            padx=10, pady=5,
            fg='white', bg='#263D42',
            command=self.apply_changes
        )
        apply.pack(padx=10, pady=5)

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
            self.image_displayer.deiconify() # Unhide window when opening a new file
            self.update_canvas(self.image_data)
        except Exception as e:
            print('File Error:', e)
            print('Filename:', filename)


    def apply_changes(self, *args):
        if(len(self.image_data) == 0):
            return
        
        modified_image_data = self.image_data

        if(self.is_negative.get()):
            modified_image_data = 1 - modified_image_data
        
        if(self.apply_log.get()):
            modified_image_data = np.log2(1 + modified_image_data)
        
        modified_image_data = modified_image_data ** self.gamma_value.get()
        modified_image_data = modified_image_data * self.brightness.get() / 100

        self.update_canvas(modified_image_data)
        
    def update_canvas(self, image_data: np.ndarray):
        img = ImageTk.PhotoImage(self.array_to_image(image_data))
        self.canvas.configure(image=img)
        self.canvas.image = img
    
    def on_close_display(self):
        self.image_displayer.withdraw()

    def get_image_array(self, filename: str, mode: str) -> np.ndarray:
        with Image.open(filename) as im:
            img_array = np.asarray(im.convert(mode))
            return img_array / 255
    
    def array_to_image(self, array: np.ndarray) -> Image:
        scaled_array = np.uint8(np.clip(array * 255, 0, 255))
        return Image.fromarray(scaled_array)

app = MainApp(root)
root.mainloop()