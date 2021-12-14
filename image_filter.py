import numpy as np
import tkinter as tk
from numpy.core.fromnumeric import var
from numpy.lib.arraypad import pad
from scipy import signal

class Filter():
    def __init__(self, size) -> None:
        self.size = size
        self.offset = size // 2

    def apply(self, i: int, j: int, image_data: np.ndarray) -> float:
        pass

class NonLinearFilter(Filter):
    
    def __init__(self, size, f):
        super().__init__(size)
        self.f = f
    
    def apply(self, i: int, j: int, image_data: np.ndarray) -> float:
        w = len(image_data)
        h = len(image_data[0])
        minx, maxx = max(i - self.offset, 0), min(i + self.offset + 1, w)
        miny, maxy = max(j - self.offset, 0), min(j + self.offset + 1, h)
        if len(image_data.shape) == 3:
            return self.f(image_data[minx:maxx, miny:maxy], axis=(0,1))
        return self.f(image_data[minx:maxx, miny:maxy])

class ConvFilter(Filter):

    def __init__(self, size: int, kernel_weigths: np.ndarray, normalize: bool = True) -> None:
        Filter.__init__(self, size)
        self.kernel_weigths = kernel_weigths
        if(normalize):
            self.kernel_weigths = self.kernel_weigths / np.sum(kernel_weigths)

    def apply(self, i: int, j: int, image_data: np.ndarray) -> float:
        '''
        Acumulates the value of neighborhood weighted by filter_data
        '''
        v = 0
        w = len(image_data)
        h = len(image_data[0])
        for i_off in range(-self.offset, self.offset + 1):
            for j_off in range(-self.offset, self.offset + 1):
                x = i + i_off
                y = j + j_off
                if(x < 0 or y < 0 or x >= w or y >= h): continue
                
                v += image_data[x][y] * self.kernel_weigths[i_off+self.offset][j_off+self.offset]
        return v

class BoxBlur(ConvFilter):
    def __init__(self, size: int) -> None:
        super().__init__(size, np.ones((size, size)))

class LaplacianFilter(ConvFilter):

    def __init__(self, size=3) -> None:
        super().__init__(3, np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]), normalize=False)

class GaussianFilter(ConvFilter):
    def __init__(self, size=3) -> None:
        super().__init__(3, np.array([
            [1, 2, 1],
            [2, 8, 2],
            [1, 2, 1]
        ]), normalize=True)
    
class SobelX(ConvFilter):
    def __init__(self, size: int=3) -> None:
        super().__init__(3, np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]), normalize=False)
    
class SobelY(ConvFilter):
    def __init__(self, size: int=3) -> None:
        super().__init__(3, np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]), normalize=False)

class MedianFilter(NonLinearFilter):
    
    def __init__(self, size):
        super().__init__(size, np.median)

class GeometricFilter(NonLinearFilter):
    
    def geometric_mean(self, arr: np.ndarray, **kwargs):
        return arr.prod(**kwargs)**(1/len(arr))

    def __init__(self, size) -> None:
        super().__init__(size, self.geometric_mean)

class MeanFilter(NonLinearFilter):
    
    def __init__(self, size):
        super().__init__(size, np.mean)

class HarmonicFilter(NonLinearFilter):

    def harmonic_mean(self, arr, **kwargs):
        return len(arr) / np.sum(1.0/arr, **kwargs)

    def __init__(self, size) -> None:
        super().__init__(size, self.harmonic_mean)

class ContraharmonicFilter(NonLinearFilter):

    def contraharmonic_mean(self, arr, **kwargs):
        return np.sum(arr**2, **kwargs) / (np.sum(arr, **kwargs) + 1e-12)

    def __init__(self, size) -> None:
        super().__init__(size, self.contraharmonic_mean)

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

class FrequencyFilter:
    def __init__(self, r: int, w: int, h: int, invert: bool = False, gauss: bool = False) -> None:
        self.f = np.zeros((w,h))
        half_w = w//2
        half_h = h//2
        if gauss:
            mask = gkern(kernlen=2*r, std=r/2)
        else:
            Y, X = np.ogrid[-r:r:, -r:r]
            dist_from_center = np.sqrt(X**2 + Y**2)
            mask = dist_from_center <= r
        
        self.f[half_w-r:half_w+r, half_h-r:half_h+r] = mask

        if invert:
            self.f = 1 - self.f
    
    def get_mask(self):
        return self.f

    def apply(self, img_data: np.ndarray) -> np.ndarray:
        return self.f * img_data

class DiskFrequencyFilter:
    def __init__(self, r1: int, r2:int, w: int, h: int, invert: bool = False, gauss: bool = False) -> None:
        self.f = np.zeros((w,h))
        half_w = w//2
        half_h = h//2
        
        assert r2 > r1

        Y2, X2 = np.ogrid[-r2:r2:, -r2:r2]
        dist_from_center2 = np.sqrt(X2**2 + Y2**2)

        mask =  (r1 <= dist_from_center2) & (dist_from_center2 <= r2)

        if gauss:
            g = gkern(kernlen=2*r2, std=r2/4)
            mask = g * mask
        
        self.f[half_w-r2:half_w+r2, half_h-r2:half_h+r2] = mask
        
        if invert:
            self.f = 1 - self.f
    
    def get_mask(self):
        return self.f

    def apply(self, img_data: np.ndarray) -> np.ndarray:
        return self.f * img_data

class ConvFilterEditor:

    def __init__(self, size, app=None):
        self.root = tk.Toplevel()
        self.root.title('Conv. filter editor')
        self.app = app
        self.size = size
        offset = 2
        self.input_cells = []
        tk.Label(self.root, text='Filter Weights').grid(row=0, column=0, columnspan=self.size)
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell_input = tk.Entry(self.root, width=5)
                cell_input.insert(tk.END, '1.0')
                cell_input.grid(row=i+offset, column=j, padx=2, pady=2)
                row.append(cell_input)
            self.input_cells.append(row)
        
        self.norm = tk.BooleanVar()
        checkbox = tk.Checkbutton(
            self.root,
            text='Normalize',
            variable=self.norm
        )
        self.norm.set(False)
        checkbox.grid(row=self.size*i + offset, column=0, columnspan=self.size)
        apply_controls = tk.Button(
            self.root,
            text='Apply',
            command=self.apply
        )
        apply_controls.grid(row=self.size*i+1 + offset, column=0, columnspan=self.size)
        
    
    def apply(self):
        try:
            kernel_weights = [[float(self.input_cells[i][j].get()) for j in range(self.size)] for i in range(self.size)]
            f = ConvFilter(self.size, np.array(kernel_weights), normalize=self.norm.get())
            f_img = self.app.apply_filter(f)
            self.app.modified_image_data = f_img
            self.app.image_data = f_img
            self.app.update_canvas(f_img)
            self.root.destroy()
        except Exception as e:
            print(e)

class FrequencyFilterEditor:
    
    def __init__(self, app=None):
        self.root = tk.Toplevel()
        self.root.title('Fourier Editor')
        self.app = app
        self.w = len(self.app.modified_image_data)
        self.h = len(self.app.modified_image_data[0])
        self.max_r = min(self.w//2, self.h//2)
        self.freq = self.app.get_fft_from_image(self.app.modified_image_data)
        self.img = self.app.normalize_image(np.absolute(self.freq), 0, 1000)
        tk.Label(self.root, text='Inner Radius').pack()
        self.inner_r = tk.IntVar()
        inner_c = tk.Scale(
            self.root,
            variable=self.inner_r,
            from_=0,
            to=self.max_r,
            orient=tk.HORIZONTAL,
            command=self.update_mask
        )
        self.inner_r.set(0)
        inner_c.pack()

        tk.Label(self.root, text='Outer Radius').pack()
        self.outer_r = tk.IntVar()
        outer_c = tk.Scale(
            self.root,
            variable=self.outer_r,
            from_=0,
            to=self.max_r,
            orient=tk.HORIZONTAL,
            command=self.update_mask
        )
        self.outer_r.set(20)
        outer_c.pack()
        
        self.g_decay = tk.BooleanVar()
        tk.Checkbutton(
            self.root,
            text='Use Gaussian Decay',
            variable=self.g_decay,
            offvalue=False,
            onvalue=True,
            command=self.update_mask
        ).pack()

        self.invert = tk.BooleanVar()
        tk.Checkbutton(
            self.root,
            text='Invert',
            variable=self.invert,
            offvalue=False,
            onvalue=True,
            command=self.update_mask
        ).pack()

        tk.Button(
            self.root,
            text='Apply',
            command=self.apply
        ).pack()
        
        f = DiskFrequencyFilter(self.inner_r.get(), self.outer_r.get(), self.w, self.h, gauss=self.g_decay.get())
        self.app.update_canvas(f.apply(self.img))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def update_mask(self, *event):
        f = DiskFrequencyFilter(
            self.inner_r.get(), 
            self.outer_r.get(),
            self.w, self.h, gauss=self.g_decay.get(), invert=self.invert.get(),
        )
        self.app.update_canvas(f.apply(self.img))

    def apply(self):
        f = DiskFrequencyFilter(
            self.inner_r.get(), 
            self.outer_r.get(),
            self.w, self.h, gauss=self.g_decay.get(), invert=self.invert.get(),
        )
        frequency = self.app.get_fft_from_image(self.app.modified_image_data)
        filtered_frequency = f.apply(frequency)
        
        self.app.modified_image_data = self.app.get_image_from_fft(filtered_frequency)
        self.app.image_data = self.app.modified_image_data
        self.app.update_canvas(self.app.modified_image_data)
        self.on_close()

    def on_close(self):
        self.app.update_canvas(self.app.modified_image_data)
        self.root.destroy()

if __name__ == '__main__':
    ConvFilterEditor(3)