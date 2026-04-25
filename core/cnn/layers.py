import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# Layer Interface
# ==========================================
class Layer:
    def forward(self, input_data, training=True): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ==========================================
# Core Layers
# ==========================================
class Dense(Layer):
    def __init__(self, units, activation=None):
        self.output_size = units
        self.activation = activation
        self.weights = None
        self.bias = None

    def forward(self, input_data, training=True):
        self.input = input_data
        if self.weights is None:
            input_size = input_data.shape[1]
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(2. / input_size)
            self.bias = np.zeros((1, self.output_size))
            
        self.z = np.dot(self.input, self.weights) + self.bias
        
        if self.activation == 'relu':
            return np.maximum(0, self.z)
        elif self.activation == 'softmax':
            return softmax(self.z)
        return self.z

    def backward(self, output_gradient, learning_rate):
        if self.activation == 'relu':
            output_gradient = output_gradient * (self.z > 0)
        
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient

class Flatten(Layer):
    def forward(self, input_data, training=True):
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)
        
    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)

class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            return input_data * self.mask
        return input_data

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask

# ==========================================
# Convolutional & Pooling Layers
# ==========================================
class Conv2D(Layer):
    def __init__(self, filters, kernel_size=(3, 3), activation=None, input_shape=None):
        self.out_channels = filters
        self.k = kernel_size[0] 
        self.activation = activation
        self.filters_weights = None

    def forward(self, input_data, training=True):
        self.input = input_data 
        batch_size, in_channels, h, w = input_data.shape
        
        if self.filters_weights is None:
            self.filters_weights = np.random.randn(self.out_channels, in_channels, self.k, self.k) * np.sqrt(2./(self.k*self.k*in_channels))
            
        # grab the sliding windows
        windows = sliding_window_view(self.input, window_shape=(self.k, self.k), axis=(2, 3))
        
        # use a vectorized tensordot instead of a loop
        # this sums over the input channels and kernel dimensions
        # MASSIVE PERFORMANCE BOOST, for loops are crappy
        output = np.tensordot(windows, self.filters_weights, axes=([1, 4, 5], [1, 2, 3]))
        
        # the result is (batch, out_h, out_w, filters)
        self.z = output.transpose(0, 3, 1, 2)
        
        if self.activation == 'relu':
            return np.maximum(0, self.z)
        return self.z

    def backward(self, output_gradient, learning_rate):
        if self.activation == 'relu':
            output_gradient = output_gradient * (self.z > 0)
            
        batch_size, in_channels, h, w = self.input.shape
        
        # figure out the filter gradient first
        windows = sliding_window_view(self.input, window_shape=(self.k, self.k), axis=(2, 3))
        
        # sum over the batch, out_h, and out_w dimensions
        filters_gradient = np.tensordot(output_gradient, windows, axes=([0, 2, 3], [0, 2, 3]))
        filters_gradient /= batch_size
        
        # build the input gradient with a full convolution
        # pad the output gradient by kernel_size - 1
        pad = self.k - 1
        pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        padded_grad = np.pad(output_gradient, pad_width, mode='constant')
        
        # slide windows over the padded gradient
        grad_windows = sliding_window_view(padded_grad, window_shape=(self.k, self.k), axis=(2, 3))
        
        # flip the filters spatially by 180 degrees
        flipped_filters = self.filters_weights[:, :, ::-1, ::-1]
        
        input_gradient = np.tensordot(grad_windows, flipped_filters, axes=([1, 4, 5], [0, 2, 3]))
        input_gradient = input_gradient.transpose(0, 3, 1, 2)
        
        # update the filters
        self.filters_weights -= learning_rate * filters_gradient
        return input_gradient

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.pool = pool_size[0]

    def forward(self, input_data, training=True):
        self.input = input_data
        N, C, H, W = input_data.shape
        P = self.pool
        
        # calculate the output dimensions
        out_h, out_w = H // P, W // P
        
        # if the image does not divide evenly, drop the edge pixels
        self.H_trunc, self.W_trunc = out_h * P, out_w * P
        self.truncated_input = input_data[:, :, :self.H_trunc, :self.W_trunc]
        
        # reshape the pooling patches into their own dimension
        self.reshaped_input = self.truncated_input.reshape(N, C, out_h, P, out_w, P)
        
        # take the max across each pool window
        return self.reshaped_input.max(axis=(3, 5))
        
    def backward(self, output_gradient, learning_rate):
        N, C, H, W = self.input.shape
        P = self.pool
        out_h, out_w = H // P, W // P
        
        # expand the incoming gradient to match the pooled chunks
        grad_reshaped = output_gradient.reshape(N, C, out_h, 1, out_w, 1)
        
        # find the pixels that won each max-pooling window
        max_vals = self.reshaped_input.max(axis=(3, 5), keepdims=True)
        mask = (self.reshaped_input == max_vals)
        
        # route the gradient only to those winning pixels
        trunc_grad = (mask * grad_reshaped).reshape(N, C, self.H_trunc, self.W_trunc)
        
        # pad the edge pixels back with zeros when needed
        pad_h, pad_w = H - self.H_trunc, W - self.W_trunc
        input_gradient = np.pad(trunc_grad, ((0,0), (0,0), (0,pad_h), (0,pad_w)), mode='constant')
        
        return input_gradient

# ==========================================
# Batch Normalization
# ==========================================
class BatchNormalization(Layer):
    def __init__(self):
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self, x, training=True):
        self.input_shape = x.shape
        if self.gamma is None:
            channels = x.shape[1] if len(x.shape) == 4 else x.shape[-1]
            shape = (1, channels, 1, 1) if len(x.shape) == 4 else (1, channels)
            self.gamma = np.ones(shape)
            self.beta = np.zeros(shape)
            self.running_mean = np.zeros(shape)
            self.running_var = np.ones(shape)

        axes = (0, 2, 3) if len(x.shape) == 4 else (0,)

        if training:
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.x_centered = x - mean
            self.std = np.sqrt(var + self.eps)
            self.x_norm = self.x_centered / self.std
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_norm + self.beta

    def backward(self, output_gradient, learning_rate):
        axes = (0, 2, 3) if len(self.input_shape) == 4 else (0,)
        N = np.prod([self.input_shape[i] for i in axes])
        
        gamma_grad = np.sum(output_gradient * self.x_norm, axis=axes, keepdims=True)
        beta_grad = np.sum(output_gradient, axis=axes, keepdims=True)
        
        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad
        
        dx_norm = output_gradient * self.gamma
        dx = (1. / N) / self.std * (N * dx_norm - np.sum(dx_norm, axis=axes, keepdims=True) - self.x_norm * np.sum(dx_norm * self.x_norm, axis=axes, keepdims=True))
        return dx