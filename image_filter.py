import numpy as np

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

class LaplacianFilter(ConvFilter):

    def __init__(self, size) -> None:
        super().__init__(3, np.array([
            [1, 1, 1],
            [1,-8, 1],
            [1, 1, 1]
        ]), normalize=False)

class GaussianFilter(ConvFilter):
    def __init__(self, size=3) -> None:
        super().__init__(3, np.array([
            [1, 2, 1],
            [2, 8, 2],
            [1, 2, 1]
        ]), normalize=True)


class MedianFilter(NonLinearFilter):
    
    def __init__(self, size):
        super().__init__(size, np.median)

class GeometricFilter(NonLinearFilter):

    def __init__(self, size) -> None:
        super().__init__(size, lambda arr: arr.prod()**(1/len(arr)))

class MeanFilter(NonLinearFilter):

    def __init__(self, size):
        super().__init__(size, np.mean)

class HarmonicFilter(NonLinearFilter):

    def __init__(self, size) -> None:
        super().__init__(size, lambda arr: len(arr) / np.sum(1.0/arr))

class ContraharmonicFilter(NonLinearFilter):

    def __init__(self, size) -> None:
        super().__init__(size, lambda arr: np.sum(arr**2) / np.sum(arr))

class FrequencyFilter:
    def __init__(self, r: int, w: int, h: int, invert: bool = False) -> None:
        self.f = np.zeros((w,h))
        half_w = w//2
        half_h = h//2
        Y, X = np.ogrid[-r:r:, -r:r]
        dist_from_center = np.sqrt(X**2 + Y**2)
        mask = dist_from_center <= r
        self.f[half_w-r:half_w+r, half_h-r:half_h+r] = mask

        if invert:
            self.f = 1 - self.f
    def apply(self, img_data: np.ndarray) -> np.ndarray:
        return self.f * img_data