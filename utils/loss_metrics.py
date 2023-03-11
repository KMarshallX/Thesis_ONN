"""
Loss Metircs
"""
import tensorflow as tf
from tensorflow import keras
from utils.utils_func import detector_regions, new_detector_regions


def loss(model, target, inputs, training):

    # Initialize loss metric
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    logits = []
    for i in range(len(inputs)):
        pred = model(inputs[i], training=training)
        logits.append(detector_regions(pred))

    return loss_object(target, logits)
    # loss_object(y_true, y_pred)
    # y_true: Ground truth values. shape = [batch_size, d0, .. dN],

# Use the tf.GradientTape context to calculate the gradients used to optimize your model:
def grad(model, target, input):
    with tf.GradientTape() as tape:
        loss_val = loss(model, target, input, training=True)

    return loss_val, tape.gradient(loss_val, model.trainable_variables)

def optimizer_init(learning_rate):

    op = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return op
