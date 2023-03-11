"""
Train loop for Optical Neural Network
"""

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.ONN import ONNModel
from utils.loss_metrics import grad, optimizer_init
from utils.utils_func import detector_regions, new_detector_regions, TrainDataLoader
import config

def train(epoch_num, batch_size, size, model, optimizer, save_name):

    # initialize training data loader
    TDL = TrainDataLoader(size)
    train_steps = int(len(TDL) / batch_size)

    train_loss_results = []
    train_accuracy_results = []

    for epoch in tqdm(range(epoch_num)):
        loss_avg = tf.keras.metrics.Mean()
        
        #TODO
        acc = tf.keras.metrics.SparseCategoricalAccuracy()

        for step in tqdm(range(train_steps)):

            x_batch, y_batch = TDL[step*batch_size:(step+1)*batch_size]
            
            loss_val, grad_val = grad(ONN_block, y_batch, x_batch)
            optimizer.apply_gradients(zip(grad_val, ONN_block.trainable_variables))

            # Track progress
            loss_avg.update_state(loss_val) # add current loss

            # Update accuracy
            y_pred = []
            for z in range(len(x_batch)):
                pred = ONN_block(x_batch[z], training=True)
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                y_pred.append(detector_regions(pred))
            
            acc.update_state(y_batch, y_pred)
            print(f"\nStep :[{step+1}/{train_steps}]; Current batch loss :{loss_avg.result():.4f}, Current batch accuracy :{acc.result():.4%}")
        
        # End epoch
        train_loss_results.append(loss_avg.result())
        train_accuracy_results.append(acc.result())
        print(f"\nEpoch :[{epoch+1}/{epoch_num}]")

    print("\nTraining completed!\n")

    #save the trained model
    path = './saved_model/'+save_name
    model.save(path)
    print("Model successfully saved!")

    # save the loss/accuracy curve
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.savefig("./saved_images/loss_acc.jpg" )

if __name__ == "__main__":

    args = config.args

    # Load parameters
    downsample = args.ds
    planeSpacing = args.ps # plane spacing
    wavelength = args.lamb 
    pixelSize = downsample*args.pix
    Nx = args.sz / downsample 
    Ny = args.sz / downsample 
    size = [args.sz, args.sz]
    epoch_num = args.ep # iteration number
    batch_size = args.bt 
    learning_rate = args.lr
    model_name = args.mo_name
    
    #Check devices
    device_name = tf.test.gpu_device_name()
    if len(device_name) > 0:
        print(f"Found GPU at: {device_name}\n")
    else:
        device_name = "/device:CPU:0"
        print(f"No GPU found, using {device_name}\n")
    # device_name = "/device:CPU:0"

    #initialize optimizer
    boundaries = [6000, 12000, 18000, 21000]
    values = [1e-2, 1e-3, 1e-4, 5e-5, 1e-6]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = optimizer_init(learning_rate_fn)

    with tf.device(device_name):
        ONN_block = ONNModel(size, planeSpacing, wavelength, Nx, Ny, pixelSize, 7)
        # Train Loop
        train(epoch_num, batch_size, size[0], ONN_block, optimizer, model_name)