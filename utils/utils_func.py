import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import scipy.ndimage as scind

def ImgPreprocess(dataset, size):
    """
    :params dataset: numpy list
    :params size: new size of the feeding image, dtype = int
    """
    resized_dataset = []
    for num in range(len(dataset)):
        img = np.array(dataset[num]>100, dtype='uint8') # thresholding
        img = cv2.resize(img, (size,size)) # resizing
        resized_dataset.append(img)
    return resized_dataset

def data_loader(size):
    """
    :params size: new size of the feeding image, dtype = int
    """
    digits_MNIST = keras.datasets.mnist
    (train_img, train_lb), (test_img, test_lb) = digits_MNIST.load_data()
    # train_img: contains the 60000 training images, each image has 28*28 pixels
    # train_lb: contains the 60000 LABELS of training images
    # test_img: contains the 10000 training images, each image has 28*28 pixels
    # test_lb: contains the 10000 LABELS of training images
    train_set = ImgPreprocess(train_img, size)
    test_set = ImgPreprocess(test_img, size)

    return train_set, train_lb, test_set, test_lb

def rang(arr,top_left_corner):
    # top_left_corner is a tuple i.e. (2,2)
    x0 = top_left_corner[0]
    y0 = top_left_corner[1]
    delta = 40
    return arr[x0:x0+delta,y0:y0+delta]

def _detector_regions(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
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
    ])) 
    
def detector_regions(a):
    return tf.square(tf.abs(_detector_regions(a)))

class TrainDataLoader:
    def __init__(self, size) -> None:

        digits_MNIST = keras.datasets.mnist
        (self.train_img, self.train_lb), (_, _) = digits_MNIST.load_data()
        
        self.size = size
    
    def __len__(self):
        return len(self.train_lb)
    
    def __getitem__(self, idx):
        X = self.train_img[idx]
        y = self.train_lb[idx]
        # normailization
        X = X/255
        X = scind.zoom(X, (1, self.size/X.shape[1], self.size/X.shape[2]), order=0, mode='nearest')
        X = tf.convert_to_tensor(X, dtype=tf.complex64)
        return X, y

class TestDataLoader:
    def __init__(self, size) -> None:

        digits_MNIST = keras.datasets.mnist
        (_, _), (self.test_img, self.test_lb) = digits_MNIST.load_data()
        
        self.size = size
    
    def __len__(self):
        return len(self.test_lb)
    
    def __getitem__(self, idx):
        X = self.test_img[idx]
        y = self.test_lb[idx]
        # normailization
        X = X/255
        X = scind.zoom(X, (1, self.size/X.shape[1], self.size/X.shape[2]), order=0, mode='nearest')
        X = tf.convert_to_tensor(X, dtype=tf.complex64)
        return X, y

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



# Deprecated:

# def rang_one(arr,top_left_corner): # For early stage, 64*64 is used
#     # top_left_corner is a tuple i.e. (2,2)
#     x0 = top_left_corner[0]
#     y0 = top_left_corner[1]
#     delta = 8
#     arr[x0:x0+delta,y0:y0+delta] = 1
#     return arr

# def _labelMapping(label, size): # For early stage, 64*64 is used
#     """
#     Map the numerical label to a 2d positional label
#     """
#     canvas = np.zeros((size, size))
#     if label == 0:
#         rang_one(canvas, (10,10)) # 0
#     elif label == 1:
#         rang_one(canvas,(10,28)) # 1
#     elif label == 2:
#         rang_one(canvas,(10,46)) # 2
#     elif label == 3:
#         rang_one(canvas,(28,4)) # 3
#     elif label == 4:
#         rang_one(canvas,(28,20)) # 4
#     elif label == 5:
#         rang_one(canvas,(28,36)) # 5
#     elif label == 6:
#         rang_one(canvas,(28,52)) # 6
#     elif label == 7:
#         rang_one(canvas,(46,10)) # 7
#     elif label == 8:
#         rang_one(canvas,(46,28)) # 8
#     elif label == 9:
#         rang_one(canvas,(46,46)) # 9
#     else:
#         print("Invalid label value!")
    
#     return canvas

# def labelMapping(labelset, size):
#     """
#     :params size: new size of the feeding image, dtype = int
#     """
#     new_set = []
#     for i in range(len(labelset)):
#         new_set.append(_labelMapping(labelset[i], size))
    
#     return new_set

