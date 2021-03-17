import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D

def get_session():
    """ Limit session memory usage                                          限制会话内存使用
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary                                     标量值Tensorflow摘要
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

def conv_layer(d, k):
    """ Returns a 2D Conv layer, with and ReLU activation                   返回带有ReLU激活的2D转换层
    """
    return Conv2D(d, k, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

def conv_block(inp, d=3, pool_size=(2, 2), k=3):
    """ Returns a 2D Conv block, with a convolutional layer, max-pooling    返回带有卷积层的2D Conv块，最大池化
    """
    conv = conv_layer(d, k)(inp)
    return MaxPooling2D(pool_size=pool_size)(conv)

class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)               奥恩斯坦-乌伦贝克噪声（@slowbull的原始代码）
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x
