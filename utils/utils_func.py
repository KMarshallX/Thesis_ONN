import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.ndimage as scind

def rang(arr,top_left_corner, delta):
    # top_left_corner is a tuple i.e. (2,2)
    x0 = top_left_corner[0]
    y0 = top_left_corner[1]
    # delta = 3 # for 28*28
    # delta = 4 # for 56*56
    return arr[x0:x0+delta,y0:y0+delta]

# Ues this when the image is padded
def _detector_regions(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        # for 38*38 pattern 
        rang(a,(116,116),3), # 0
        rang(a,(116,126),3), # 1
        rang(a,(116,136),3), # 2
        rang(a,(126,114),3),  # 3
        rang(a,(126,122),3), # 4
        rang(a,(126,130),3), # 5
        rang(a,(126,138),3), # 6
        rang(a,(136,116),3), # 7
        rang(a,(136,126),3), # 8
        rang(a,(136,136),3)  # 9
    ])) 

# 2*2 detector area, need to change line 10
def _detector_regions2(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        rang(a,(117,117),2),
        rang(a,(117,127),2),
        rang(a,(117,137),2),
        rang(a,(127,115),2),
        rang(a,(127,123),2),
        rang(a,(127,131),2),
        rang(a,(127,138),2),
        rang(a,(137,117),2),
        rang(a,(137,127),2),
        rang(a,(137,137),2)
    ])) 

# 4*4 detector area, need to change line 10
def _detector_regions4(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        rang(a,(109,109),4),
        rang(a,(109,126),4),
        rang(a,(109,143),4),
        rang(a,(126,109),4),
        rang(a,(126,120),4),
        rang(a,(126,132),4),
        rang(a,(126,143),4),
        rang(a,(143,109),4),
        rang(a,(143,126),4),
        rang(a,(143,143),4)
    ]))

# 6*6 detector area, need to change line 10
def _detector_regions6(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        rang(a,(109,109),6),
        rang(a,(109,125),6),
        rang(a,(109,141),6),
        rang(a,(125,109),6),
        rang(a,(125,120),6),
        rang(a,(125,130),6),
        rang(a,(125,141),6),
        rang(a,(141,109),6),
        rang(a,(141,125),6),
        rang(a,(141,141),6)
    ]))

# 10*10 detector area, need to change line 10
def _detector_regions10(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        rang(a,(103,103),10),
        rang(a,(103,123),10),
        rang(a,(103,143),10),
        rang(a,(123,93),10),
        rang(a,(123,113),10),
        rang(a,(123,133),10),
        rang(a,(123,153),10),
        rang(a,(143,103),10),
        rang(a,(143,123),10),
        rang(a,(143,143),10)
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
    
def detector_regions(a, mode):
    if mode == 0:
        # original 3*3 detector area
        b = _detector_regions(tf.abs(a))
    elif mode == 1:
        # 2*2
        b = _detector_regions2(tf.abs(a))
    elif mode == 2:
        # 4*4
        b = _detector_regions4(tf.abs(a))
    elif mode == 3:
        # 6*6
        b = _detector_regions6(tf.abs(a))
    elif mode == 4:
        # 10*10
        b = _detector_regions10(tf.abs(a))
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
        #resize to 38*38 pixels
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
    
class TrainFashionDataLoader:
    def __init__(self, size, pad=True, phase_object=False) -> None:

        fashion_MNIST = keras.datasets.fashion_mnist
        (self.train_img, self.train_lb), (_, _) = fashion_MNIST.load_data() # 38*38
        
        self.size = size # pad the resized image to 256*256
        self.pad = pad
        self.phase_object = phase_object

    def __len__(self):
        return len(self.train_lb)
    
    def __getitem__(self, idx):
        X = self.train_img[idx]
        y = self.train_lb[idx]
        #resize to 38*38 pixels
        X = scind.zoom(X, (1, 19/14, 19/14), order=0, mode='nearest') # double the size of the image
        # normailization
        X = X / 255.0
        # convert to phase object
        if self.phase_object == True:
            X = tf.math.exp(2 * np.pi * 1j * tf.cast(X, dtype=tf.complex64))
        else:
            # # thresholding the intensity image
            # thresh = 0.5
            # X[X >= thresh] = 1
            # X[X < thresh] = 0
            X = X
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
        #resize to 38*38 pixels 
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
    
class TestFashionDataLoader:
    def __init__(self, size, pad=True, phase_object=False) -> None:

        fashion_MNIST = keras.datasets.fashion_mnist
        (_, _), (self.test_img, self.test_lb) = fashion_MNIST.load_data() # 38*38
        
        self.size = size # pad the resized image to 256*256
        self.pad = pad
        self.phase_object = phase_object

    def __len__(self):
        return len(self.test_lb)
    
    def __getitem__(self, idx):
        X = self.test_img[idx]
        y = self.test_lb[idx]
        #resize to 38*38 pixels 
        X = scind.zoom(X, (1, 19/14, 19/14), order=0, mode='nearest') # double the size of the image
        # normailization
        X = X / 255.0
        # convert to phase object
        if self.phase_object == True:
            X = tf.math.exp(2 * np.pi * 1j * tf.cast(X, dtype=tf.complex64))
        else:
            # # thresholding the intensity image
            # thresh = 0.5
            # X[X >= thresh] = 1
            # X[X < thresh] = 0
            X = X
        # padding
        if self.pad == True:
            x_padsize = (self.size-X.shape[1])//2
            y_padsize = (self.size-X.shape[2])//2
            pad_width = ((0,0),(x_padsize,x_padsize),(y_padsize,y_padsize))
            X = np.pad(X, pad_width, mode='constant', constant_values=0)
        
        X = tf.convert_to_tensor(X, dtype=tf.complex64)
        return X, y

def fashion_item_label_converter(numerical_label):
    fashion_items = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    fashion_dict = {i : fashion_items[i] for i in range(0, 10)}

    return fashion_dict[numerical_label]

# Discarded functions
# def new_rang(arr, shape, size=56, base = 500):
#     x0 = shape[0] * size // base
#     y0 = shape[2] * size // base
#     delta = (shape[1]-shape[0])* size // base
#     return arr[x0:x0+delta,y0:y0+delta]

# def _new_detector_regions(a):
#     return tf.map_fn(tf.math.reduce_mean, 
#                     tf.map_fn(tf.math.square,tf.convert_to_tensor([
#                                 rang(a,(34,34)), # 0
#                                 rang(a,(34,108)), # 1
#                                 rang(a,(34,182)), # 2
#                                 rang(a,(108,18)),  # 3
#                                 rang(a,(108,78)), # 4
#                                 rang(a,(108,138)), # 5
#                                 rang(a,(108,198)), # 6
#                                 rang(a,(182,34)), # 7
#                                 rang(a,(182,108)), # 8
#                                 rang(a,(182,182))  # 9
#                             ]))) 
    
# def new_detector_regions(a):
#     return tf.nn.softmax(tf.cast(_new_detector_regions(a), dtype=tf.float64))


