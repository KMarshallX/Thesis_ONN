import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.ndimage as scind

def rang(arr,top_left_corner):
    # top_left_corner is a tuple i.e. (2,2)
    x0 = top_left_corner[0]
    y0 = top_left_corner[1]
    delta = 3 # for 28*28
    # delta = 4 # for 56*56
    return arr[x0:x0+delta,y0:y0+delta]

# Ues this when the image is padded
def _detector_regions(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        # # for 56*56 pattern
        # rang(a,(83,83)), # 0
        # rang(a,(83,99)), # 1
        # rang(a,(83,115)), # 2
        # rang(a,(99,80)),  # 3
        # rang(a,(99,92)), # 4
        # rang(a,(99,104)), # 5
        # rang(a,(99,116)), # 6
        # rang(a,(115,83)), # 7
        # rang(a,(115,99)), # 8
        # rang(a,(115,115))  # 9

        # # for 28*28 pattern (1)
        # rang(a,(90,90)), # 0
        # rang(a,(90,98)), # 1
        # rang(a,(90,106)), # 2
        # rang(a,(98,86)),  # 3
        # rang(a,(98,94)), # 4
        # rang(a,(98,102)), # 5
        # rang(a,(98,110)), # 6
        # rang(a,(106,90)), # 7
        # rang(a,(106,98)), # 8
        # rang(a,(106,106))  # 9

        # # for 28*28 pattern (2)
        # rang(a,(91,91)), # 0
        # rang(a,(91,99)), # 1
        # rang(a,(91,107)), # 2
        # rang(a,(99,89)),  # 3
        # rang(a,(99,95)), # 4
        # rang(a,(99,102)), # 5
        # rang(a,(99,108)), # 6
        # rang(a,(107,91)), # 7
        # rang(a,(107,99)), # 8
        # rang(a,(107,107))  # 9

        # for 38*38 pattern 
        rang(a,(116,116)), # 0
        rang(a,(116,126)), # 1
        rang(a,(116,136)), # 2
        rang(a,(126,114)),  # 3
        rang(a,(126,122)), # 4
        rang(a,(126,130)), # 5
        rang(a,(126,138)), # 6
        rang(a,(136,116)), # 7
        rang(a,(136,126)), # 8
        rang(a,(136,136))  # 9


    ])) 

# # Use this when the image is not padded
# def _detector_regions(a):
#     return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
#         rang(a,(11,11)), # 0
#         rang(a,(11,26)), # 1
#         rang(a,(11,41)), # 2
#         rang(a,(26,8)),  # 3
#         rang(a,(26,20)), # 4
#         rang(a,(26,32)), # 5
#         rang(a,(26,44)), # 6
#         rang(a,(41,11)), # 7
#         rang(a,(41,26)), # 8
#         rang(a,(41,41))  # 9
#     ])) 
    
def detector_regions(a):
    b = _detector_regions(tf.abs(a))
    return tf.square(b)

class TrainDataLoader:
    def __init__(self, size, pad=True, phase_object=False) -> None:

        digits_MNIST = keras.datasets.mnist
        (self.train_img, self.train_lb), (_, _) = digits_MNIST.load_data() # 38*38
        
        self.size = size # pad the resized image to 256*256
        self.pad = pad
        self.phase_object = phase_object

    def __len__(self):
        return len(self.train_lb)
    
    def __getitem__(self, idx):
        X = self.train_img[idx]
        y = self.train_lb[idx]
        #resize
        X = scind.zoom(X, (1, 19/14, 19/14), order=0, mode='nearest') # double the size of the image
        # normailization
        X = X / 255.0
        # convert to phase object
        if self.phase_object == True:
            X = tf.math.exp(2 * np.pi * 1j * tf.cast(X, dtype=tf.complex64))
        else:
            # thresholding the intensity image
            thresh = 0.5
            X[X >= thresh] = 1
            X[X < thresh] = 0
        # padding
        if self.pad == True:
            x_padsize = (self.size-X.shape[1])//2
            y_padsize = (self.size-X.shape[2])//2
            pad_width = ((0,0),(x_padsize,x_padsize),(y_padsize,y_padsize))
            X = np.pad(X, pad_width, mode='constant', constant_values=0)
        
        X = tf.convert_to_tensor(X, dtype=tf.complex64)
        return X, y
    
class TestDataLoader:
    def __init__(self, size, pad=True, phase_object=False) -> None:

        digits_MNIST = keras.datasets.mnist
        (_, _), (self.test_img, self.test_lb) = digits_MNIST.load_data() # 38*38
        
        self.size = size # pad the resized image to 256*256
        self.pad = pad
        self.phase_object = phase_object

    def __len__(self):
        return len(self.test_lb)
    
    def __getitem__(self, idx):
        X = self.test_img[idx]
        y = self.test_lb[idx]
        #resize
        X = scind.zoom(X, (1, 19/14, 19/14), order=0, mode='nearest') # double the size of the image
        # normailization
        X = X / 255.0
        # convert to phase object
        if self.phase_object == True:
            X = tf.math.exp(2 * np.pi * 1j * tf.cast(X, dtype=tf.complex64))
        else:
            # thresholding the intensity image
            thresh = 0.5
            X[X >= thresh] = 1
            X[X < thresh] = 0
        # padding
        if self.pad == True:
            x_padsize = (self.size-X.shape[1])//2
            y_padsize = (self.size-X.shape[2])//2
            pad_width = ((0,0),(x_padsize,x_padsize),(y_padsize,y_padsize))
            X = np.pad(X, pad_width, mode='constant', constant_values=0)
        
        X = tf.convert_to_tensor(X, dtype=tf.complex64)
        return X, y

def new_rang(arr, shape, size=56, base = 500):
    x0 = shape[0] * size // base
    y0 = shape[2] * size // base
    delta = (shape[1]-shape[0])* size // base
    return arr[x0:x0+delta,y0:y0+delta]

def _new_detector_regions(a):
    return tf.map_fn(tf.math.reduce_mean, 
                    tf.map_fn(tf.math.square,tf.convert_to_tensor([
                                rang(a,(34,34)), # 0
                                rang(a,(34,108)), # 1
                                rang(a,(34,182)), # 2
                                rang(a,(108,18)),  # 3
                                rang(a,(108,78)), # 4
                                rang(a,(108,138)), # 5
                                rang(a,(108,198)), # 6
                                rang(a,(182,34)), # 7
                                rang(a,(182,108)), # 8
                                rang(a,(182,182))  # 9
                            ]))) 
    
def new_detector_regions(a):
    return tf.nn.softmax(tf.cast(_new_detector_regions(a), dtype=tf.float64))


