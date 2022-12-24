"""
Train loop for Optical Neural Network
"""

import tensorflow as tf
from tqdm import tqdm

from model.ONN import ONNModel
from utils.loss_metrics import loss, grad, optimizer_init
from utils.utils_func import data_loader, detector_regions

def train(epoch_num, batch_size, train_set, label_set, model, optimizer, save_name):
    # TODO: log the training loss and accuracy for each step
    train_steps = int(len(train_set) / batch_size)
    for epoch in tqdm(range(epoch_num)):
        loss_avg = tf.keras.metrics.Mean()
        acc = tf.keras.metrics.CategoricalAccuracy()

        step_len = len(train_set)
        lab_len = len(label_set)
        assert step_len == lab_len, "Train set and input label length not matched!"

        for step in tqdm(range(train_steps)):

            x_batch = train_set[step*batch_size:(step+1)*batch_size]
            y_batch = label_set[step*batch_size:(step+1)*batch_size]
            
            loss_val, grad_val = grad(ONN_block, y_batch, x_batch)
            optimizer.apply_gradients(zip(grad_val, ONN_block.trainable_variables))

            # Track progress
            loss_avg.update_state(loss_val) # add current loss

            # Update accuracy
            y_pred = []
            for z in range(len(x_batch)):
                pred = ONN_block(x_batch[z], training=True)
                y_pred.append(tf.argmax(detector_regions(pred)).numpy())
            
            acc.update_state(y_batch, y_pred)

            print(f"\nStep :[{step+1}/{step_len}]; Loss :{loss_avg.result():.4f}, Accuracy :{acc.result():.4%}")
        print(f"\nEpoch :[{epoch+1}/{epoch_num}]; Loss :{loss_avg.result():.4f}, Accuracy :{acc.result():.4%}")

    print("\nTraining completed!\n")

    #save the trained model
    path = './saved_model/'+save_name
    model.save(path)
    print("Model successfully saved!")

if __name__ == "__main__":

    # Load parameters
    downsample = 4
    planeSpacing = 25.14e-3 # plane spacing
    wavelength = 1565e-9 
    pixelSize = downsample*8e-6
    Nx = 256 / downsample # 64
    Ny = 256 / downsample # 64
    size = [64, 64]
    epoch_num = 10 # iteration number
    batch_size = 50 
    
    # Check devices
    device_name = tf.test.gpu_device_name()
    if len(device_name) > 0:
        print(f"Found GPU at: {device_name}\n")
    else:
        device_name = "/device:CPU:0"
        print(f"No GPU found, using {device_name}\n")

    #load date
    train_set, train_lab, test_set, test_lab = data_loader(64)
    #convert to tensors
    tf_train_set = tf.convert_to_tensor(train_set[0:5000], dtype=tf.complex64)
    # tf_test_set = tf.convert_to_tensor(test_set)
    tf_train_lab = train_lab[0:5000]

    #initialize optimizer
    learning_rate = 1e-3
    optimizer = optimizer_init(learning_rate)

    
    with tf.device(device_name):
        ONN_block = ONNModel(size, planeSpacing, wavelength, Nx, Ny, pixelSize, 7)

        # Train Loop
        train(epoch_num, batch_size, tf_train_set, tf_train_lab, ONN_block, optimizer, 'test2')