from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
class Paint:

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self, img: Image.Image, fft: np.ndarray, inv_fourier, app = None):
        self.app = app
        self.fft = fft
        self.inv_fourier = inv_fourier
        self.root = Toplevel()
        self.root.title('FFT Painter')
        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=2)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=3)

        self.save_btn = Button(
            self.root,
            text='Save',
            command=self.save
        )
        self.save_btn.grid(row=0, column=4)

        self.img = img
        self.mask = Image.fromarray(np.ones((img.size[1], img.size[0]))*254)
        self.draw = ImageDraw.Draw(self.mask)
        self.image = ImageTk.PhotoImage(self.img)
        self.c = Canvas(self.root, bg='white', width=self.img.size[0], height=self.img.size[1])
        self.c.create_image(0,0,image=self.image,anchor=NW)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.active_button = self.brush_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_brush(self):
        self.activate_button(self.brush_button)
    
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        paint_num = 1 if self.eraser_on else 0
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([(self.old_x, self.old_y), (event.x, event.y)], fill=paint_num, width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def save(self):
        if self.app == None:
            path = filedialog.asksaveasfilename(
                initialdir='./img',
                title='Save as ...',
                filetypes=(('all files', '*.*'), ('JPG images', '*.jpeg'), ('PNG images', '*.png'), ('Tif images', '*.tif'), ('BMP Images', '*.bmp'))
            )
    
            self.img.save(path)
        else:
            self.app.modified_image_data = self.get_image_from_fft(self.fft * (np.uint8(self.mask)/255))
            self.app.update_canvas(self.app.modified_image_data)
            self.root.destroy()
    
    def get_image_from_fft(self, freq: np.ndarray):
        return np.real(self.inv_fourier(np.fft.ifftshift(freq)))

if __name__ == '__main__':
    Paint(Image.open('./img/Fig0314(a)(100-dollars).tif'))