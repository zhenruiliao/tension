import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations
import numpy as np

from base import FORCEModel
from models import EchoStateNetwork

class ConstrainedNoFeedbackESN(EchoStateNetwork):
    """ 
    Constrained ESN without feedback as described in `Hadjiabadi et al. <https://pubmed.ncbi.nlm.nih.gov/34197732/>`_. 

    :param structural_connectivity: ``self.units`` x ``self.units`` structural connectivity matrix
    :type structural_connectivity: Tensor[2D float]
    :param noise_param: Tuple containing the mean and standard deviation of white noise in the forward pass 
    :type noise_param: tuple[float]
    """
    def __init__(self, 
                 structural_connectivity,
                 noise_param,
                 **kwargs):
        self.structural_connectivity = structural_connectivity
        self.noise_param = noise_param
        super().__init__(output_size=kwargs['units'], output_kernel_trainable=False, recurrent_kernel_trainable=True, **kwargs)

    def initialize_recurrent_kernel(self, recurrent_kernel=None):
        if recurrent_kernel is None:        
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=1 / (self.units * self._p_recurr)**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                      minval=None, 
                                                                                      dtype=tf.dtypes.int64)[0])
        
            recurrent_kernel = self._p_recurr * keras.layers.Dropout(1 - self._p_recurr)(initializer(shape=(self.units, self.units)), 
                                                                                         training=True)

        recurrent_kernel = tf.linalg.set_diag(input=recurrent_kernel, diagonal=tf.zeros(self.units))  
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=keras.initializers.constant(recurrent_kernel),
                                                trainable=self._recurrent_kernel_trainable,
                                                name='recurrent_kernel')
 
    def call(self, inputs, states):
        prev_a, prev_h, _ = states      
        z = backend.dot(prev_h, self.recurrent_kernel)
        white_noise = tf.random.normal(shape=(1, self.units), 
                                       mean=self.noise_param[0], 
                                       stddev=self.noise_param[1])
        
        dadt = -prev_a + self._g * z + white_noise
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)

        return z, [a, h, z]

    def build(self, input_shape):

        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
          
        self.recurrent_nontrainable_boolean_mask = (self.recurrent_kernel == 0)
        self.built = True

    @classmethod
    def from_weights(cls, 
                     weights, 
                     recurrent_nontrainable_boolean_mask, 
                     **kwargs):
        """
        Creates a ConstrainedNoFeedbackESN object with pre-initialized weights. 

        :param weights: Two 2D Tensors containing the input and recurrent kernels  
        :type weights: tuple[Tensor[2D float]] of length 2

        :returns: A ConstrainedNoFeedbackESN object initialized with the input weights
        """

        input_kernel, recurrent_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        units = input_units      

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2]) == units)
        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'
        assert recurrent_kernel.shape == recurrent_nontrainable_boolean_mask.shape, "Boolean mask and recurrent kernel shape mis-match"
        # assert tf.math.count_nonzero(tf.boolean_mask(recurrent_kernel, recurrent_nontrainable_boolean_mask)).numpy() == 0, "Invalid boolean mask"  

        self = cls(units=units, output_size=units, p_recurr=None, **kwargs)
        self.recurrent_nontrainable_boolean_mask = tf.convert_to_tensor(recurrent_nontrainable_boolean_mask)
        
        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.built = True

        return self

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self._initial_a is not None:
          init_a = self._initial_a
        else:
          init_a = tf.zeros((batch_size, self.units))  

        init_h =  self.activation(init_a)
        init_out = backend.dot(init_h, self.recurrent_kernel) 

        return (init_a, init_h, init_out)

class BioFORCEModel(FORCEModel):
    """
    Trains constrained ESN without feedback based on `Hadjiabadi et al. 
    <https://pubmed.ncbi.nlm.nih.gov/34197732/>`_. 
    """
    def pseudogradient_wR(self, P_Gx, h, z, y, Ph, hPh):
        # Return pseudogradient for wR
        # Example array shapes
        # P_Gx : self.units x self.units x self.units 
        # h : 1 x self.units

        e = tf.transpose(z - y)
        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
        dwR = 1 / (1 + tf.transpose(hPh[:,:,0])) * e * Ph 
        s = self.original_force_layer.structural_connectivity

        return s * tf.transpose(dwR) 