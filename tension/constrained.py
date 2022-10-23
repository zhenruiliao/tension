import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations
import numpy as np

from .base import FORCEModel, FORCELayer

class ConstrainedNoFeedbackESN(FORCELayer):
    """ 
    Constrained ESN without feedback as described in `Hadjiabadi et al. <https://pubmed.ncbi.nlm.nih.gov/34197732/>`_. 

    **Note:** The target must have dimension equal to the number of neurons. 

    :param dtdivtau: dt divided by network dynamic time scale.
    :type dtdivtau: float
    :param structural_connectivity: ``self.units x self.units`` structural connectivity matrix
    :type structural_connectivity: Tensor[2D float]
    :param noise_param: Tuple of length 2 containing (in order) the mean and standard deviation
        for the white noise in the forward pass. 
    :type noise_param: tuple[float]
    :param initial_a: An optional ``1 x self.units`` tensor or numpy array specifying
        the initial neuron pre-activation firing rates. (*Default: None*)
    :type initial_a: Tensor[2D float] or None
    :param recurrent_kernel_trainable: Boolean on whether or not to train 
        recurrent kernel. Note that this is the only kernel that can be trained
        for this layer. (*Default: True*)
    :type recurrent_kernel_trainable: bool
    :param kwargs: See :class:`.FORCELayer` for additional required args. This layer 
        has no output weights and therefore the ``output_size`` and 
        ``output_kernel_trainable`` parameters are not supported.
    """
    
    def __init__(self, 
    	         dtdivtau,
                 structural_connectivity,
                 noise_param,
                 initial_a=None,
                 recurrent_kernel_trainable=True,
                 **kwargs):
        super().__init__(output_size=kwargs['units'], 
                         output_kernel_trainable=False, 
                         recurrent_kernel_trainable=recurrent_kernel_trainable, 
                         **kwargs)
        self.dtdivtau = dtdivtau
        self.structural_connectivity = structural_connectivity
        self.noise_param = noise_param
        self._noise_seed = None if self._seed is None else self._seed + 1
        self._initial_a = initial_a

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
        white_noise = tf.random.normal(shape=(inputs.shape[0], self.units), 
                                       mean=self.noise_param[0], 
                                       stddev=self.noise_param[1],
                                       seed=self._noise_seed)
        dadt = -prev_a + self._g * z + white_noise
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)

        return z, [a, h, z]

    def build(self, input_shape):
        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        if self._p_recurr == 1:
            self.recurrent_nontrainable_boolean_mask = None
        else:
            self.recurrent_nontrainable_boolean_mask = (self.recurrent_kernel == 0)
        self.built = True

    @classmethod
    def from_weights(cls, weights, recurrent_nontrainable_boolean_mask, **kwargs):
        """
        Creates a ConstrainedNoFeedbackESN object with pre-initialized weights. 
        
        **Note:** ``p_recurr`` parameter is not supported in this method. ``units`` parameter
        is inferred from the input weights. 

        :param weights: Two 2D Tensors containing the input and recurrent kernels  
        :type weights: tuple[Tensor[2D float]] of length 2
        :param recurrent_nontrainable_boolean_mask: A 2D boolean array with
            the same shape as the recurrent kernel, where True indicates that
            the corresponding weight in the recurrent kernel is not trainable. 
        :type recurrent_nontrainable_boolean_mask: Tensor[2D bool]
        :param kwargs: Additional parameters required to initialize the layer. 

        :returns: A :class:`.ConstrainedNoFeedbackESN` object initialized with the input weights
        """
        input_kernel, recurrent_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        units = input_units      

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2]) == units)
        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'
        assert recurrent_kernel.shape == recurrent_nontrainable_boolean_mask.shape, "Boolean mask and recurrent kernel shape mis-match"
        if tf.math.count_nonzero(tf.boolean_mask(recurrent_kernel, recurrent_nontrainable_boolean_mask)).numpy() != 0:
            print("Warning: Recurrent kernel has non-zero weights (indicating neuron connection) that are not trainable") 

        self = cls(units=units, p_recurr=None, **kwargs)
        self.recurrent_nontrainable_boolean_mask = tf.convert_to_tensor(recurrent_nontrainable_boolean_mask)
        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.built = True

        return self

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Initializes the states of the layer (called implicitly during layer build). See:
        https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

        :returns:
            - **a** (*Tensor[2D float]*) - ``1 x self.units`` tensor containing the 
              pre-activation neuron firing rates
            - **h** (*Tensor[2D float]*) - ``1 x self.units`` tensor containing the 
              neuron firing rates
            - **output** (*Tensor[2D float]*) - ``1 x self.units`` tensor containing 
              the output of each neuron  
        """
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
