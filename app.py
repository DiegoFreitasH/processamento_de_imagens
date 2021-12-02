import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import Menu, Scale, filedialog
from tkinter.constants import HORIZONTAL
from PIL import ImageTk, Image
from image_filter import ContraharmonicFilter, DiskFrequencyFilter, FrequencyFilter, GaussianFilter, GeometricFilter, HarmonicFilter, MedianFilter, LaplacianFilter, SobelX, SobelY
from paint import Paint
from curve import CurveEditor

IMG_DIRECTORY = '~/UFC/processamento_imagens/processing_project/img'
filetypes = (('all files', '*.*'), ('JPG images', '*.jpeg'), ('PNG images', '*.png'), ('Tif images', '*.tif'), ('BMP Images', '*.bmp'))
root = tk.Tk()

'''TODO
Filtros de Sobel – x e y separados DONE
Detecção não linear de bordas pelo gradiente (magnitude) DONE
Esteganografia
# FOURIER
Cálculo da Transformada Discreta de Fourier
Cálculo da transformada inversa
# COR 
Criar ferramenta para transformação entre sistemas de cores: RGB<->HSV
Algoritmos de escala de cinza: média aritmética simples e média ponderada.
Sépia
Chroma-Key
Histograma (R, G, B e V)
Equalização de Histograma em imagens coloridas
Suavização e Aguçamento em imagens coloridas
Ajuste de Matiz, Saturação e Brilho
Ajuste de Canal
Dividindo os tons em escuros, médios e claros
Implementar escala e rotação com interpolação pelo vizinho mais próximo e linear
'''
class MainApp(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.title('Controls')

        self.image_data = np.array([])
        self.modified_image_data = self.image_data

        menubar = tk.Menu()
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Open', command=self.open_file)
        filemenu.add_command(label='Save', command=self.save_image)
        menubar.add_cascade(label='File', menu=filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label='Values Curve', command=self.edit_values_curve)
        menubar.add_cascade(label='Edit', menu=editmenu)

        filtermenu = tk.Menu(menubar, tearoff=0)
        filtermenu.add_command(label='Gaussian Filter', command=self.create_filter_callback(GaussianFilter))
        filtermenu.add_command(label='Laplacian Filter', command=self.create_filter_callback(LaplacianFilter))
        filtermenu.add_command(label='Median Filter', command=self.create_filter_callback(MedianFilter))
        filtermenu.add_command(label='Geometric Filter', command=self.create_filter_callback(GeometricFilter))
        filtermenu.add_command(label='Harmonic Filter', command=self.create_filter_callback(HarmonicFilter))
        filtermenu.add_command(label='Contraharmonic Filter', command=self.create_filter_callback(ContraharmonicFilter))
        filtermenu.add_command(label='Sobel X', command=self.create_filter_callback(SobelX, normalize=True))
        filtermenu.add_command(label='Sobel Y', command=self.create_filter_callback(SobelY, normalize=True))
        filtermenu.add_command(label='Non linear border detection', command=self.sobel_border_detection)
        menubar.add_cascade(label='Filter', menu=filtermenu)

        sharpenmenu = tk.Menu(menubar, tearoff=0)
        sharpenmenu.add_command(label='High Boost Sharpen', command=self.sharpen_gaussian)
        sharpenmenu.add_command(label='Laplacian Sharpen', command=self.sharpen_laplacian)
        menubar.add_cascade(label='Sharpen', menu=sharpenmenu)

        frequencymenu = tk.Menu(menubar, tearoff=0)
        frequencymenu.add_command(label='Fast Fourier Transform', command=self.edit_image_frequency)
        frequencymenu.add_command(label='Passa Alta', command=self.pass_low_filter)
        frequencymenu.add_command(label='Rejeita Baixa', command=self.pass_high_filter)
        frequencymenu.add_command(label='Disk', command=self.disk_freq_filter)
        menubar.add_cascade(label='Frequency', menu=frequencymenu)

        self.parent.config(menu=menubar)
        
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

        tk.Label(parent, text='Log').pack()
        self.apply_log = tk.IntVar()
        log_control = tk.Scale(
            parent,
            variable=self.apply_log,
            from_=1,
            to=10,
            orient=HORIZONTAL,
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

        tk.Label(parent, text='Filter Size').pack(expand=False)
        self.filter_size = tk.IntVar() 
        filter_sizes = [3, 5, 7, 9]
        filter_size_controls = tk.OptionMenu(
            parent,
            self.filter_size,
            *filter_sizes
        )
        self.filter_size.set(3)
        filter_size_controls.pack()
        self.frequency_filter_radius = tk.IntVar()
        frequency_radius_controls = tk.Scale(
            parent,
            variable=self.frequency_filter_radius,
            from_=10,
            to=50,
            orient=HORIZONTAL
        )
        self.frequency_filter_radius.set(20)
        frequency_radius_controls.pack()
        self.gaussian_decay_check = tk.BooleanVar()
        gaussian_decay_controls = tk.Checkbutton(
            parent,
            text='Use Gaussian Decay',
            variable=self.gaussian_decay_check,
        )
        gaussian_decay_controls.pack()
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
            filetypes=filetypes
        )

        try:
            self.image_data = self.get_image_array(filename, 'L')
            self.modified_image_data = self.image_data
            self.image_displayer.deiconify() # Unhide window when opening a new file
            self.update_canvas(self.image_data)
        except Exception as e:
            print('File Error:', e)
            print('Filename:', filename)

    def save_image(self):
        if len(self.modified_image_data) == 0:
            return

        path = filedialog.asksaveasfilename(
            initialdir=IMG_DIRECTORY,
            title='Save as ...',
            filetypes=filetypes
        )

        self.array_to_image(self.modified_image_data).save(path)

    def apply_changes(self, *event):
        if(len(self.image_data) == 0):
            return
        
        self.modified_image_data = self.image_data

        if(self.is_negative.get()):
            self.modified_image_data = 1 - self.modified_image_data
        
        if(self.apply_log.get() > 2):
            self.modified_image_data = np.log(1 + self.normalize_image(self.modified_image_data)) / np.log(self.apply_log.get())
        
        self.modified_image_data = self.modified_image_data ** (self.gamma_value.get() / 100)
        self.modified_image_data = self.modified_image_data * self.brightness.get() / 100

        self.update_canvas(self.modified_image_data)
        
    def update_canvas(self, image_data: np.ndarray):
        img = self.array_to_image(image_data)
        img.thumbnail((600,600))
        photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=photo)
        self.canvas.image = photo

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
        self.image_data = np.array([[color_lookup[v]/255 for v in row] for row in pixel_matrix])
        self.modified_image_data = self.image_data
        
        self.update_canvas(self.modified_image_data)

    def edit_values_curve(self):
        x = CurveEditor(self)

    def sharpen_laplacian(self):
        mask = self.apply_filter(LaplacianFilter())

        self.modified_image_data = self.modified_image_data + 1.5*mask
        self.image_data = self.modified_image_data 
        self.update_canvas(self.image_data)

    def sharpen_gaussian(self):
        mask = self.apply_filter(GaussianFilter())
        self.modified_image_data = self.modified_image_data + 1.5*(self.modified_image_data - mask)
        self.image_data = self.modified_image_data 
        self.update_canvas(self.image_data)

    def sobel_border_detection(self):
        sobelx = SobelX(self.filter_size.get())
        sobely = SobelY(self.filter_size.get())

        dx = self.apply_filter(sobelx)
        dy = self.apply_filter(sobely)

        self.modified_image_data = np.abs(dx) + np.abs(dy)
        self.image_data = self.modified_image_data
        self.update_canvas(self.modified_image_data)

    def create_filter_callback(self, filter_obj, normalize=False):

        def filter_callback():
            self.image_data = self.apply_filter(filter_obj(self.filter_size.get()), normalize)
            self.modified_image_data = self.image_data
            self.update_canvas(self.image_data)
        
        return filter_callback

    def apply_filter(self, f, normalize=False):
        filtered_img = np.empty_like(self.modified_image_data)
        h = len(self.modified_image_data)
        w = len(self.modified_image_data[0])

        for i in range(h):
            for j in range(w):
                filtered_img[i][j] = f.apply(i,j,self.modified_image_data)
        if normalize:
            filtered_img = self.normalize_image(filtered_img)
        return filtered_img
    
    def get_fft_from_image(self, image: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft2(image))
    
    def get_image_from_fft(self, fft: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.ifftshift(fft))) 

    # Create apply custom image frequency
    def edit_image_frequency(self):
        frequency = self.get_fft_from_image(self.modified_image_data)
        # Display options
        img = self.array_to_image(np.real(frequency))
        Paint(img, frequency, self)

    def pass_low_filter(self):
        w = len(self.modified_image_data)
        h = len(self.modified_image_data[0])
        f = FrequencyFilter(self.frequency_filter_radius.get(), w, h, gauss=self.gaussian_decay_check.get())
        frequency = self.get_fft_from_image(self.modified_image_data)
        filtered_frequency = f.apply(frequency)
        
        self.modified_image_data = self.get_image_from_fft(filtered_frequency)
        self.update_canvas(self.modified_image_data)
    
    def pass_high_filter(self):
        w = len(self.modified_image_data)
        h = len(self.modified_image_data[0])
        f = FrequencyFilter(self.frequency_filter_radius.get(), w, h, True, gauss=self.gaussian_decay_check.get())
        frequency = self.get_fft_from_image(self.modified_image_data)
        filtered_frequency = f.apply(frequency)
        
        self.modified_image_data = self.get_image_from_fft(filtered_frequency)
        self.update_canvas(self.modified_image_data)

    def disk_freq_filter(self):
        w = len(self.modified_image_data)
        h = len(self.modified_image_data[0])
        r = self.frequency_filter_radius.get()
        f = DiskFrequencyFilter(r, r+5, w, h)
        frequency = self.get_fft_from_image(self.modified_image_data)
        filtered_frequency = f.apply(frequency)
        
        self.modified_image_data = self.get_image_from_fft(filtered_frequency)
        self.update_canvas(self.modified_image_data)

    def normalize_image(self, image_data: np.ndarray) -> np.ndarray:
        minv = np.abs(np.min(image_data))
        maxv = np.max(image_data)
        return (image_data + minv) / (maxv + minv)

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