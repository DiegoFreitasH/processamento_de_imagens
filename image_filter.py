import numpy as np

class Filter():
    def __init__(self, size) -> None:
        self.size = size
        self.offset = size // 2

    def apply(self, i: int, j: int, image_data: np.ndarray) -> float:
        pass

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

class MedianFilter(Filter):
    
    def apply(self, i: int, j: int, image_data: np.ndarray) -> float:
        '''
        Get median value of neighborhood
        '''
        w = len(image_data)
        h = len(image_data[0])
        minx, maxx = max(i - self.offset, 0), min(i + self.offset + 1, w)
        miny, maxy = max(j - self.offset, 0), min(j + self.offset + 1, h)

        return np.median(image_data[minx:maxx, miny:maxy])
   
    
class LaplacianFilter(ConvFilter):

    def __init__(self) -> None:
        super().__init__(3, np.array([
            [1, 1, 1],
            [1,-8, 1],
            [1, 1, 1]
        ]), normalize=False)

class GaussianFilter(ConvFilter):
    def __init__(self) -> None:
        super().__init__(3, np.array([
            [1, 2, 1],
            [2, 8, 2],
            [1, 2, 1]
        ]), normalize=True)