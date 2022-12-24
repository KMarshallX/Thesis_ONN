import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

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

def rang_one(arr,top_left_corner): # For early stage, 64*64 is used
    # top_left_corner is a tuple i.e. (2,2)
    x0 = top_left_corner[0]
    y0 = top_left_corner[1]
    delta = 8
    arr[x0:x0+delta,y0:y0+delta] = 1
    return arr

def _labelMapping(label, size): # For early stage, 64*64 is used
    """
    Map the numerical label to a 2d positional label
    """
    canvas = np.zeros((size, size))
    if label == 0:
        rang_one(canvas, (10,10)) # 0
    elif label == 1:
        rang_one(canvas,(10,28)) # 1
    elif label == 2:
        rang_one(canvas,(10,46)) # 2
    elif label == 3:
        rang_one(canvas,(28,4)) # 3
    elif label == 4:
        rang_one(canvas,(28,20)) # 4
    elif label == 5:
        rang_one(canvas,(28,36)) # 5
    elif label == 6:
        rang_one(canvas,(28,52)) # 6
    elif label == 7:
        rang_one(canvas,(46,10)) # 7
    elif label == 8:
        rang_one(canvas,(46,28)) # 8
    elif label == 9:
        rang_one(canvas,(46,46)) # 9
    else:
        print("Invalid label value!")
    
    return canvas

def labelMapping(labelset, size):
    """
    :params size: new size of the feeding image, dtype = int
    """
    new_set = []
    for i in range(len(labelset)):
        new_set.append(_labelMapping(labelset[i], size))
    
    return new_set

def data_loader(size):
    """
    :params size: new size of the feeding image, dtype = int
    """
    digits_MNIST = keras.datasets.mnist
    (train_img, train_lb), (test_img, test_lb) = digits_MNIST.load_data()
    # train_img: contains the 60000 training images, each image has 64*64 pixels
    # train_lb: contains the 60000 LABELS of training images
    # test_img: contains the 10000 training images, each image has 64*64 pixels
    # test_lb: contains the 10000 LABELS of training images
    train_set = ImgPreprocess(train_img, size)
    test_set = ImgPreprocess(test_img, size)

    return train_set, train_lb, test_set, test_lb

def rang(arr,top_left_corner):
    # top_left_corner is a tuple i.e. (2,2)
    x0 = top_left_corner[0]
    y0 = top_left_corner[1]
    delta = 8
    return arr[x0:x0+delta,y0:y0+delta]

def _detector_regions(a):
    return tf.map_fn(tf.reduce_mean,tf.convert_to_tensor([
        rang(a,(10,10)), # 0
        rang(a,(10,28)), # 1
        rang(a,(10,46)), # 2
        rang(a,(28,4)),  # 3
        rang(a,(28,20)), # 4
        rang(a,(28,36)), # 5
        rang(a,(28,52)), # 6
        rang(a,(46,10)), # 7
        rang(a,(46,28)), # 8
        rang(a,(46,46))  # 9
    ])) 
    
def detector_regions(a):
    return tf.square(tf.abs(_detector_regions(a)))


