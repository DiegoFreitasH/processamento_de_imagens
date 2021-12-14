from tkinter.constants import HORIZONTAL
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

IMG_DIRECTORY = '~/UFC/processamento_imagens/processing_project/img'
filetypes = (('all files', '*.*'), ('JPG images', '*.jpeg'), ('PNG images', '*.png'), ('Tif images', '*.tif'), ('BMP Images', '*.bmp'))

class Chroma:

    def __init__(self, base_img: np.ndarray, app=None) -> None:
        self.root = tk.Toplevel()
        self.app = app
        self.base_img = base_img
        self.green = np.array([0,1,0])
        self.threshold = tk.DoubleVar()
        self.w = len(base_img)
        self.h = len(base_img[0])
        self.has_chroma_img = False
        tk.Button(
            self.root,
            text='Open Chroma Image',
            command=self.open_file,
        ).pack()
        tk.Scale(
            self.root,
            variable=self.threshold,
            orient=HORIZONTAL,
            from_=1,
            to=100,
            command=self.update_threshold
        ).pack()
        self.threshold.set(45)
        tk.Button(
            self.root,
            text='Apply',
            command=self.apply
        ).pack()
        tk.Button(
            self.root,
            text='Cancel',
            command=self.cancel
        ).pack()
        if app == None:
            self.image_display = tk.Label(self.root)
            photo = ImageTk.PhotoImage(self.array_to_image(base_img))
            self.image_display.configure(image=photo)
            self.image_display.image = photo
            self.image_display.pack()
        

        self.chroma_mask = np.linalg.norm(self.base_img - self.green, axis=2) <= (self.threshold.get()/100)
        self.update_display(self.chroma_mask)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.root.mainloop()

    def open_file(self):
        filename = filedialog.askopenfilename(
            initialdir=IMG_DIRECTORY, 
            title="Select File",
            filetypes=filetypes
        )

        try:
            with Image.open(filename) as im:
                im=im.resize((self.h, self.w),Image.BILINEAR)
                self.chroma_data = np.asarray(im) / 255
                if im.mode == 'L':
                    self.chroma_data = np.repeat(self.chroma_data[:, :, np.newaxis], 3, axis=2)
                if im.mode == 'RGBA':
                    self.chroma_data = self.chroma_data[:,:,:-1]
                self.has_chroma_img = True
                self.update_threshold()
        except Exception as e:
            print('File Error:', e)
            print('Filename:', filename)
    
    def update_threshold(self, *event):
        self.chroma_mask = np.linalg.norm(self.green - self.base_img, axis=2) <= (self.threshold.get()/100)
        self.chroma_mask = self.chroma_mask.astype(np.int64)
        self.chroma_mask = np.repeat(self.chroma_mask[:, :, np.newaxis], 3, axis=2)
        if not self.has_chroma_img:
            self.update_display(self.chroma_mask)
        else:
            self.edited_img = ((1 - self.chroma_mask) * self.base_img) + (self.chroma_mask * self.chroma_data)
            self.update_display(self.edited_img)

    def update_display(self, img):
        if self.app == None:
            photo = ImageTk.PhotoImage(self.array_to_image(img))
            self.image_display.configure(image=photo)
            self.image_display.image = photo
            self.image_display.grid(row=1, column=0)
        else:
            self.app.update_canvas(img)

    def apply(self):
        self.app.modified_image_data = ((1 - self.chroma_mask) * self.base_img) + (self.chroma_mask * self.chroma_data)
        self.app.image_data = self.app.modified_image_data
        self.app.update_canvas(self.app.modified_image_data)
        self.root.destroy()

    def cancel(self):
        self.app.update_canvas(self.app.modified_image_data)
        self.root.destroy()

    def array_to_image(self, data):
        return Image.fromarray(np.uint8(np.clip(data * 255, 0, 255)))

if __name__ == '__main__':
    with Image.open('./img/char.jpg') as im:
        Chroma(np.asarray(im)/255)