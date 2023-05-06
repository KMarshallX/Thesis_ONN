"""
Train loop for Optical Neural Network
"""

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.ONN import ONNModel
from utils.loss_metrics import grad, optimizer_init
from utils.utils_func import detector_regions, TrainDataLoader, TrainFashionDataLoader
import config
import warnings

def train(data_type, epoch_num, batch_size, size, model, optimizer, save_name, step_num):

    # initialize training data loader
    if data_type == "digit":
        TDL = TrainDataLoader(size, phase_object=False)
    elif data_type == "fashion":
        TDL = TrainFashionDataLoader(size, phase_object=False)
    else:
        raise Exception("Choose data type between [digit] and [fashion]!")

    if step_num == 0:
        train_steps = int(len(TDL) / batch_size)
    else:
        train_steps = step_num

    train_loss_results = []
    train_accuracy_results = []

    for epoch in tqdm(range(epoch_num)):
        loss_avg = tf.keras.metrics.Mean()
        
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
            tqdm.write(f"\nEpoch :[{epoch+1}/{epoch_num}]; Step :[{step+1}/{train_steps}]; Current batch loss :{loss_avg.result():.4f}, Current batch accuracy :{acc.result():.4%}")
        
        # End epoch
        train_loss_results.append(loss_avg.result())
        train_accuracy_results.append(acc.result())

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
    out_img_path = "./saved_images/"+save_name+".jpg"
    plt.savefig(out_img_path)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    args = config.args

    # Load parameters
    downsample = args.ds
    planeSpacing = args.ps # plane spacing
    diffractionSpacing = args.df # Distance from source to first plane
    wavelength = args.lamb 
    pixelSize = downsample*args.pix
    size = args.sz
    Nx = size / downsample 
    Ny = size / downsample 
    
    data_type = args.da
    epoch_num = args.ep # iteration number
    layers_num = args.ly
    batch_size = args.bt 
    learning_rate = args.lr
    model_name = args.mo_name
    step_num = args.test 
    
    #Check devices
    device_name = tf.test.gpu_device_name()
    if len(device_name) > 0:
        print(f"Found GPU at: {device_name}\n")
    else:
        device_name = "/device:CPU:0"
        print(f"No GPU found, using {device_name}\n")
    # device_name = "/device:CPU:0"

    #initialize optimizer
    # boundaries = [2400, 3600]
    # values = [learning_rate, 0.8*learning_rate, 0.64*learning_rate]
    # learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = optimizer_init(learning_rate)

    # Any test message goes below here
    print(f"Source spacing: {diffractionSpacing} ,Plane spacing: {planeSpacing}, size: {size}, pixel dimension: {pixelSize}, epoch number: {epoch_num}, number of layers: {layers_num}, learning rate: {learning_rate}\n")


    with tf.device(device_name):
        ONN_block = ONNModel(size, diffractionSpacing, planeSpacing, wavelength, Nx, Ny, pixelSize, layers_num)
        # Train Loop
        train(data_type, epoch_num, batch_size, size, ONN_block, optimizer, model_name, step_num)