import numpy as np
import tensorflow as tf
from tensorflow import keras

# Building an optical layer
class opticalLayer(tf.keras.layers.Layer):
  def __init__(self, units, distance, wavelength, Nx, Ny, pixelSize):
    super(opticalLayer, self).__init__()
    self.units = units   # input dimension

    # optical parameters goes here
    self.distance = distance # distance between two layers
    self.wavelength = wavelength 

    self.Nx = Nx
    self.Ny = Ny # Scale of one layer
    
    self.pixelSize = pixelSize

    self.X = (tf.range(1, Ny+1, dtype=tf.float32)-Ny//2 - 0.5)*pixelSize
    self.Y = (tf.range(1, Nx+1, dtype=tf.float32)-Nx//2 - 0.5)*pixelSize

  def build(self, input_shape):
    input_shape = self.units # input shape is the same as the output shape for an optical layer
    # initialize phases of one layer here
    self.phase = self.init_phase(input_shape)

  def call(self, inputs):
    # inputs: input source (dtype: tf.complex64)
    # This function will return the output sources after free-space propagation (the input sources of the next layer)
    # according to X.lin et al, a sigmoid function is used to confine the phase value from 0 to pi
    confined_phi = self.phase_confinement(self.phase) # confined_phi -> [0, pi]
    # amp = tf.constant(1, dtype=tf.float64) # phase-only modulation, set the amplitude to constant 1
    
    # make amp and phase complex values
    # amp = tf.dtypes.complex(real=amp, imag=tf.zeros(inputs.shape, dtype=tf.float64))
    imag_phi = tf.dtypes.complex(real=tf.zeros(tf.shape(inputs), dtype=tf.float32), imag=confined_phi)

    output_this = tf.math.multiply(inputs, imag_phi)
    output_next = self.propagation(output_this, self.distance)   
    
    return output_next

  def init_phase(self, input_shape):
    initializer = tf.random_normal_initializer(mean=0, stddev=1)
    return tf.Variable(initializer(shape=input_shape, dtype=tf.float32), trainable=True) 

  def phase_confinement(self, phase):
    # according to X.lin et al, a sigmoid function is used to confine the phase value from 0 to pi
    return tf.constant(np.pi, dtype=tf.float32) * tf.math.sigmoid(phase)

  def propagation(self, inputs, distance):
    # (@REF: "MPLC_StartHere.m" - J.Carpenter)
    # //TODO: how to setup k-space coordinate system? - DONE
    X = self.X
    Y = self.Y
    Nx = self.Nx
    Ny = self.Ny
    lamb = self.wavelength

    [X, Y] = tf.meshgrid(X, Y)

    fs_x = Nx/(tf.math.reduce_max(X)-tf.math.reduce_min(X))
    fs_y = Ny/(tf.math.reduce_max(Y)-tf.math.reduce_min(Y))

    v_x = fs_x*(tf.range(-Nx/2, Nx/2, dtype=tf.float32)//Nx)
    v_y = fs_y*(tf.range(-Ny/2, Ny/2, dtype=tf.float32)//Ny)

    [V_X, V_Y] = tf.meshgrid(v_x, v_y)
    
    # Free-space transfer function
    sqr = tf.cast(tf.sqrt(1/lamb**2-V_X**2-V_Y**2), dtype=tf.complex64)
    tfcoef = tf.constant(-2j*np.pi*distance, dtype=tf.complex64)*sqr
    H = tf.exp(tfcoef*distance)

    # Free-space propagation
    input_fft = tf.signal.fftshift(tf.signal.fft2d(inputs))
    output = tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(H, input_fft)))

    return output

if __name__ == "__main__":
  downsample = 4
  planeSpacing = 25.14e-3
  wavelength = 1565e-9
  pixelSize = downsample*8e-6
  Nx = 256 / downsample # 64
  Ny = 256 / downsample # 64
  size = [64, 64]

  test_tensor = np.ones((64,64))
  test_tensor = tf.convert_to_tensor(test_tensor, dtype=tf.complex64)

  test_layer = opticalLayer(size, planeSpacing, wavelength, Nx, Ny, pixelSize)

  test_output = test_layer(test_tensor)
  abs_test_output = tf.abs(test_output)
  flat_abs_output = tf.reshape(abs_test_output, [-1])

  print(test_output.shape)
  print(test_output)
  print(abs_test_output)
  print(flat_abs_output)