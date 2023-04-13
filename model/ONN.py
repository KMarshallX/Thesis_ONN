import numpy as np
import tensorflow as tf
from tensorflow import keras

# first layer is the free-space propagation
class FSpropagation(tf.keras.layers.Layer):
    def __init__(self, units, distance, wavelength, Nx, Ny, pixelSize):
        super(FSpropagation, self).__init__()
        self.units = units   # input dimension

        # optical parameters goes here
        self.distance = distance # distance between two layers
        self.wavelength = wavelength 

        self.Nx = Nx
        self.Ny = Ny # Scale of one layer

        self.pixelSize = pixelSize

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        def propagation(inputs):
            # (@REF: "MPLC_StartHere.m" - J.Carpenter)
            
            lamb = self.wavelength
            distance = self.distance
            X = (tf.range(1, self.Ny + 1, dtype=tf.float32) - self.Ny / 2 - 0.5) * self.pixelSize
            Y = (tf.range(1, self.Nx + 1, dtype=tf.float32) - self.Nx / 2 - 0.5) * self.pixelSize
            [X, Y] = tf.meshgrid(X, Y)

            fs_x = self.Nx / (tf.math.reduce_max(X) - tf.math.reduce_min(X))
            fs_y = self.Ny / (tf.math.reduce_max(Y) - tf.math.reduce_min(Y))

            v_x = fs_x * (tf.range(-self.Nx / 2, self.Nx / 2, dtype=tf.float32) / self.Nx)
            v_y = fs_y * (tf.range(-self.Ny / 2, self.Ny / 2, dtype=tf.float32) / self.Ny)
            [V_X, V_Y] = tf.meshgrid(v_x, v_y)
            
            # Free-space transfer function
            sqr = tf.cast(tf.sqrt(1 / lamb ** 2 - V_X ** 2-V_Y ** 2), dtype=tf.complex64)
            tfcoef = tf.constant(-2j * np.pi * distance, dtype=tf.complex64) * sqr
            H = tf.exp(tfcoef * distance)

            # Free-space propagation
            input_fft = tf.signal.fftshift(tf.signal.fft2d(inputs))
            output = tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(H, input_fft)))

            return output
        
        return propagation(inputs)

    

# Building an optical layer
class opticalLayer(tf.keras.layers.Layer):
    def __init__(self, units, distance, wavelength, Nx, Ny, pixelSize):
        super(opticalLayer, self).__init__()
        self.units = units   # input dimension e.g. 200

        # optical parameters goes here
        self.distance = distance # distance between two layers
        self.wavelength = wavelength 

        self.Nx = Nx
        self.Ny = Ny # Scale of one layer

        self.pixelSize = pixelSize

    def build(self, input_shape): 
        initializer = tf.random_normal_initializer(mean=np.pi, stddev=np.pi)
        # initializer = tf.random_normal_initializer()
        # input shape is the same as the output shape for an optical layer
        # initialize phases of one layer here
        self.phase = tf.Variable(initial_value=initializer(shape=(self.units, self.units), dtype=tf.float32), trainable=True, constraint=lambda t: tf.clip_by_value(t, 0, 2*np.pi))
    
    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        # inputs: input source (dtype: tf.complex64)
        # This function will return the output sources after free-space propagation (the input sources of the next layer)
        # make amp and phase complex values (the model only has phase modulation for now)
        imag_phi = tf.math.exp(1j*tf.cast(self.phase, dtype = tf.complex64))
        phase_modulated_image = tf.math.multiply(inputs, imag_phi)
        
        def propagation(inputs):
            # (@REF: "MPLC_StartHere.m" - J.Carpenter)
            
            lamb = self.wavelength
            distance = self.distance
            X = (tf.range(1, self.Ny + 1, dtype=tf.float32) - self.Ny / 2 - 0.5) * self.pixelSize
            Y = (tf.range(1, self.Nx + 1, dtype=tf.float32) - self.Nx / 2 - 0.5) * self.pixelSize
            [X, Y] = tf.meshgrid(X, Y)

            fs_x = self.Nx / (tf.math.reduce_max(X) - tf.math.reduce_min(X))
            fs_y = self.Ny / (tf.math.reduce_max(Y) - tf.math.reduce_min(Y))

            v_x = fs_x * (tf.range(-self.Nx / 2, self.Nx / 2, dtype=tf.float32) / self.Nx)
            v_y = fs_y * (tf.range(-self.Ny / 2, self.Ny / 2, dtype=tf.float32) / self.Ny)
            [V_X, V_Y] = tf.meshgrid(v_x, v_y)

            # Free-space transfer function
            sqr = tf.cast(tf.sqrt(1 / lamb ** 2 - V_X ** 2 - V_Y ** 2), dtype=tf.complex64)
            tfcoef = tf.constant(-2j * np.pi * distance, dtype=tf.complex64) * sqr
            H = tf.exp(tfcoef * distance)

            # Free-space propagation
            input_fft = tf.signal.fftshift(tf.signal.fft2d(inputs))
            output = tf.signal.ifft2d(tf.signal.ifftshift(tf.multiply(H, input_fft)))

            return output
        
        return propagation(phase_modulated_image)


## Building the network model
class ONNModel(keras.Model):
    def __init__(self, units, distance, wavelength, Nx, Ny, pixelSize, num_layers):
        super(ONNModel, self).__init__()

        self.prop = FSpropagation(units, distance, wavelength, Nx, Ny, pixelSize)
        self.all_layers = []

        self.all_layers.append(self.prop) # add the first defractive layer
        for i in range(num_layers):
            layer = opticalLayer(units, distance, wavelength, Nx, Ny, pixelSize)
            self.all_layers.append(layer)

    def call(self, input_tensor):

        x = self.all_layers[0](input_tensor)
        for layer in self.all_layers[1:]:
            x = layer(x)
        return x


if __name__ == "__main__":
    inputs = tf.ones([200, 200], tf.complex64)
    
    lay = opticalLayer(200, 3e-2, 0.75e-3, 200, 200, 400e-6)
    out1 = lay(inputs)
    mod = ONNModel(200, 3e-2, 0.75e-3, 200, 200, 400e-6, 7)
    out2 = mod(inputs)
    print(out1)
    print(out1.shape)
    print(type(out1))