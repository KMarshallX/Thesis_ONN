"""
Test method for Optical Neural Network
"""

import tensorflow as tf
from tqdm import tqdm

from model.ONN import ONNModel
from utils.loss_metrics import loss, grad, optimizer_init
from utils.utils_func import data_loader


if __name__ == "__main__":

    pretrained_model = tf.keras.models.load_model("./saved_model/test1")