import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations

from .base import FORCELayer, FORCEModel

class EchoStateNetwork(FORCELayer):
    """ 
    Implements the feedback echo state network. See ``base.FORCELayer`` for states.

    :param dtdivtau: dt divided by network dynamic time scale.
    :type dtdivtau: float
    :param hscale: A scaling factor for randomly initializing the initial network activities.
        (*Default: 0.25*)
    :type hscale: float
    :param initial_a: An optional ``1 x self.units`` tensor or numpy array specifying
        the initial neuron pre-activation firing rates. (*Default: None*)
    :type initial_a: Tensor[2D float] or None
    :param kwargs: See :class:`.FORCELayer` for additional required args.
    """

    def __init__(self, 
                 dtdivtau, 
                 hscale=0.25, 
                 initial_a=None, 
                 **kwargs):
        self.dtdivtau = dtdivtau 
        self.hscale = hscale
        self._initial_a = initial_a
        super().__init__(**kwargs)        

    def call(self, inputs, states):
        prev_a, prev_h, prev_output = states      
        input_term = backend.dot(inputs, self.input_kernel)
        recurrent_term = backend.dot(prev_h, self.recurrent_kernel)
        feedback_term = backend.dot(prev_output, self.feedback_kernel)

        dadt = -prev_a + input_term + recurrent_term + feedback_term 
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)
        output = backend.dot(h, self.output_kernel)

        return output, [a, h, output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Initializes the states of the layer (called implicitly during layer build). 
        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

        :returns:
            - **a** (*Tensor[2D float]*) - ``batch_size x self.units`` tensor containing 
              the pre-activation neuron firing rates.
            - **h** (*Tensor[2D float]*) - ``batch_size x self.units`` tensor containing 
              the neuron firing rates.
            - **output** (*Tensor[2D float]*) - ``batch_size x self.output_size`` tensor 
              containing the predicted output.  
        """
        if self._initial_a is not None:
            init_a = self._initial_a
        else:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=self.hscale, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                     minval=None, 
                                                                                     dtype=tf.dtypes.int64)[0])
            init_a = initializer((batch_size, self.units))  

        init_h = self.activation(init_a)
        init_out = backend.dot(init_h, self.output_kernel) 

        return (init_a, init_h, init_out)

class NoFeedbackESN(EchoStateNetwork):
    """ 
    Implements the no feedback echo state network. See ``base.FORCELayer`` 
    for states. 

    :param recurrent_kernel_trainable: Boolean on whether or not to train 
        recurrent kernel. (*Default: True*)
    :type recurrent_kernel_trainable: bool
    :param kwargs: See :class:`.FORCELayer`  and :class:`.EchoStateNetwork` 
        for additional required args.
    """

    def __init__(self, 
                 recurrent_kernel_trainable=True, 
                 **kwargs):
        super().__init__(recurrent_kernel_trainable=recurrent_kernel_trainable, **kwargs)

    def call(self, inputs, states):
        prev_a, prev_h, prev_output = states      
        input_term = backend.dot(inputs, self.input_kernel)
        recurrent_term = backend.dot(prev_h, self.recurrent_kernel)

        dadt = -prev_a + input_term + recurrent_term 
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)
        output = backend.dot(h, self.output_kernel)

        return output, [a, h, output]

    def build(self, input_shape):
        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        self.initialize_output_kernel()
        if self._p_recurr == 1:
            self.recurrent_nontrainable_boolean_mask = None
        else:
            self.recurrent_nontrainable_boolean_mask = (self.recurrent_kernel == 0)
        self.built = True

    @classmethod
    def from_weights(cls, weights, recurrent_nontrainable_boolean_mask, **kwargs):
        """
        Creates a NoFeedbackESN object with pre-initialized weights. 

        **Note:** ``p_recurr`` parameter is not supported in this method. ``units`` and 
        ``output_size`` parameters are inferred from the input weights. 

        :param weights: tuple of tensors containing the input, recurrent, and output kernels  
        :type weights: tuple[Tensor[2D float]] of length 3
        :param recurrent_nontrainable_boolean_mask: A 2D boolean array with
            the same shape as the recurrent kernel, where True indicates that
            the corresponding weight in the recurrent kernel is not trainable. 
        :type recurrent_nontrainable_boolean_mask: Tensor[2D bool]
        :param kwargs: Additional parameters required to initialize the layer. 

        :returns: A :class:`.NoFeedbackESN` object initialized with the input weights
        """
        input_kernel, recurrent_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        output_units, output_size = output_kernel.shape
        units = input_units      

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, output_units]) == units)
        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'
        assert recurrent_kernel.shape == recurrent_nontrainable_boolean_mask.shape, "Boolean mask and recurrent kernel shape mis-match"
        if tf.math.count_nonzero(tf.boolean_mask(recurrent_kernel, recurrent_nontrainable_boolean_mask)).numpy() != 0:
            print("Warning: Recurrent kernel has non-zero weights (indicating neuron connection) that are not trainable") 

        self = cls(units=units, output_size=output_size, p_recurr=None, **kwargs)
        self.recurrent_nontrainable_boolean_mask = tf.convert_to_tensor(recurrent_nontrainable_boolean_mask)
        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.initialize_output_kernel(output_kernel)
        self.built = True

        return self
    
class TargetGeneratingNetwork(FORCELayer):
    """ 
    For internal use in FullFORCEModel
    """

    def __init__(self, 
                 dtdivtau, 
                 hint_dim,
                 hscale=0.25, 
                 **kwargs): 
        self.dtdivtau = dtdivtau 
        self.hscale = hscale
        self._hint_dim = hint_dim
        super().__init__(**kwargs)  
        
    def build(self, input_shape):
        input_shape = input_shape[:-1] + (input_shape[-1] - self.output_size - self._hint_dim,)
        super().build(input_shape)

        if self._hint_dim > 0:
           self.hint_kernel = self.add_weight(shape=(self._hint_dim, self.units),
                                              initializer=keras.initializers.RandomNormal(mean=0., 
                                                                                          stddev=1 / self._hint_dim**0.5, 
                                                                                          seed=self.seed_gen.uniform([1], 
                                                                                                                     minval=None, 
                                                                                                                     dtype=tf.dtypes.int64)[0]),
                                              trainable=False,
                                              name='hint_kernel')

    @classmethod
    def from_weights(cls, weights, **kwargs):
        raise NotImplementedError

    def call(self, inputs, states):
        prev_a, prev_h, _ = states
        input_term = backend.dot(inputs[:, :-self.output_size-self._hint_dim], self.input_kernel)
        recurrent_term = backend.dot(prev_h, self.recurrent_kernel)

        if self._hint_dim > 0:
            feedback_hint_term = backend.dot(inputs[:, -self.output_size-self._hint_dim:-self.output_size], self.hint_kernel) + \
                                        backend.dot(inputs[:, -self.output_size:], self.feedback_kernel)
        else:
            feedback_hint_term = backend.dot(inputs[:, -self.output_size:], self.feedback_kernel)

        dadt = -prev_a + input_term + recurrent_term + feedback_hint_term  
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)
        output = backend.dot(h, self.output_kernel)

        return output, [a, h, feedback_hint_term]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initializer = keras.initializers.RandomNormal(mean=0., 
                                                      stddev=self.hscale, 
                                                      seed=self.seed_gen.uniform([1], 
                                                                                 minval=None, 
                                                                                 dtype=tf.dtypes.int64)[0])
        init_a = initializer((batch_size, self.units))  
        init_h = self.activation(init_a)

        return (init_a, init_h, init_h)

class FullFORCEModel(FORCEModel):
    """ 
    Subclassed from FORCEModel, implements full FORCE learning by `DePasquale et al. 
    <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191527>`_. 

    Target to the model should be of shape ``timesteps x 1`` or ``batch size x timesteps x 1``. 
    Presently this class does not support masking out recurrent weights during 
    training (all recurrent weights must be trainable).

    **Note** - Input shapes during training and inference differ:
    During training, input to the model should be of shape ``timesteps x input dimensions + hint dimensions``
    or ``batch size x timesteps x input dimensions + hint dimensions``. 
    During inference, input to the model should be of shape ``timesteps x input dimensions`` or 
    ``batch size x timesteps x input dimensions``.
    During model call when ``training=True``, input to the call should have shape 
    ``1 x timesteps x input dimensions + hint dimensions + output dimensions``.
    During model call when ``training=False``, input to the call should have shape 
    ``1 x timesteps x input dimensions``.

    :param hint_dim: Dimension of the hint input
    :type hint_dim: int
    :param target_output_kernel_trainable: If True, train the target output kernel (*Default: False*)
    :type target_output_kernel_trainable: bool
    :param kwargs:  Other key word arguments as needed. See ``base.FORCEModel``
    """

    def __init__(self, 
                 hint_dim, 
                 target_output_kernel_trainable=True, 
                 **kwargs):
        super().__init__(**kwargs)
        self._target_output_kernel_trainable = target_output_kernel_trainable
        self._output_dim = self.original_force_layer.output_size
        self._hint_dim = hint_dim

        assert self._hint_dim >= 0, 'Hint_dim cannot be negative'
        assert self._output_dim == 1, 'Output dimension must be 1'
        
    def build(self, input_shape):
        assert self._hint_dim < input_shape[-1] - self._output_dim, 'Hint dimension too large'

        self.initialize_target_network(input_shape)
        input_shape = input_shape[:-1] + (input_shape[-1] - self._output_dim - self._hint_dim,)
        super().build(input_shape)

    def initialize_target_network(self, input_shape):
        """
        Initializes the target generating network. 

        :param input_shape: Input shape
        :type input_shape: tuple
        """
        target_seed = None
        if self.original_force_layer._seed is not None:
            target_seed = self.original_force_layer._seed + 1
        self._target_network = TargetGeneratingNetwork(units=self.units, 
                                                       output_size=self._output_dim, 
                                                       activation=self.original_force_layer.activation,
                                                       output_kernel_trainable=self._target_output_kernel_trainable,
                                                       dtdivtau=self.original_force_layer.dtdivtau,
                                                       seed=target_seed,
                                                       hint_dim=self._hint_dim) 
        self.target_network = keras.layers.RNN(self._target_network, 
                                               stateful=True, 
                                               return_state=True, 
                                               return_sequences=True)
        self.target_network.build(input_shape)

    def initialize_P(self):
        """ 
        Initializes the P matrices corresponding to the output kernel  necessary for FORCE for the 
        task and target generating network. 
        """
        self.task_P_output = self.add_weight(name='task_P_output', 
                                             shape=(self.units, self.units), 
                                             initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                             trainable=True)
        if self._target_output_kernel_trainable:
            self.target_P_output = self.add_weight(name='target_P_output', 
                                                   shape=(self.units, self.units), 
                                                   initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                                   trainable=True)
        
    def initialize_train_idx(self):
        self._task_output_kernel_idx = None
        self._task_recurrent_kernel_idx = None
        self._target_output_kernel_idx = None

        # Find the index of each trainable weights in list of trainable variables
        for idx in range(len(self.trainable_variables)):
            trainable_name = self.trainable_variables[idx].name
            if 'target_generating_network' in trainable_name and 'output_kernel' in trainable_name:
              self._target_output_kernel_idx = idx
            elif 'output_kernel' in trainable_name:
              self._task_output_kernel_idx = idx
            elif 'target_P_output' in trainable_name:
              self._target_P_output_idx = idx
            elif 'task_P_output' in trainable_name:
              self._task_P_output_idx = idx
            elif 'recurrent_kernel' in trainable_name:
              self._task_recurrent_kernel_idx = idx

    def call(self, x, training=False, reset_states=False, **kwargs):
        if not reset_states:
            output = super().call(x=x, training=training, reset_states=reset_states, **kwargs)
        else:
            original_state = [tf.identity(state) for state in self.target_network.states]
            output = super().call(x=x, training=training, reset_states=reset_states, **kwargs)
            for i, state in enumerate(self.target_network.states):
                state.assign(original_state[i], read_value=False)      

        return output

    def force_layer_call(self, x, training, **kwargs):
        if training:
            output_target, _, h_target, fb_hint_sum = self.target_network(x, **kwargs)
            output_task, _, h_task, _ = self.force_layer(x[:,:,:-self._output_dim - self._hint_dim], **kwargs)
            return output_task, h_task, output_target, h_target, fb_hint_sum
        else:
            return self.force_layer(x, **kwargs) 

    @tf.function
    def train_step(self, data):
        x, y = data
        output_task, h_task, output_target, h_target, fb_hint_sum = self(tf.concat([x, y], axis=-1), 
                                                                         training=True, 
                                                                         reset_states=False) 

        if self.force_layer.return_sequences:
            output_task = output_task[:,0,:]
        if self.target_network.return_sequences:
            output_target = output_target[:, 0, :]

        trainable_vars = self.trainable_variables

        if tf.cond(self.update_kernel_condition(), lambda : True, lambda : False):
            if self._task_output_kernel_idx is not None:
                # Inherited from FORCEModel base class
                self.update_output_kernel(h_task, 
                                          output_task, 
                                          y[:,0,:], 
                                          trainable_vars[self._task_P_output_idx],
                                          trainable_vars[self._task_output_kernel_idx])

            if self._task_recurrent_kernel_idx is not None:
                self.update_recurrent_kernel(h_task, 
                                             h_target, 
                                             fb_hint_sum, 
                                             trainable_vars)
                
            if self._target_output_kernel_idx is not None:
                # Inherited from FORCEModel base class
                self.update_output_kernel(h_target, 
                                           output_target, 
                                           y[:,0,:], 
                                           trainable_vars[self._target_P_output_idx], 
                                           trainable_vars[self._target_output_kernel_idx])
                
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y[:,0,:], output_task)

        return {m.name: m.result() for m in self.metrics}

    def update_recurrent_kernel(self, h_task, h_target, fb_hint_sum, trainable_vars):
        """
        Performs pseudogradient updates for recurrent kernel and its corresponding P tensor.

        :param h_task: ``1 x self.units`` tensor of neuron firing rates for task network
        :type h_task: Tensor[2D float]
        :param h_target: ``1 x self.units`` tensor of neuron firing rates for target network
        :type h_target: Tensor[2D float]
        :param fb_hint_sum: ``1 x self.units`` tensor of sum of inputs from feedback and hint
            components
        :type fb_hint_sum: Tensor[2D float]  
        :param trainable_vars: List of the model's trainable variable
        :type trainable_vars: list[Tensor[2D floats]]
        """
        # Update output P of the task network if it hasn't been updated already
        if self._task_output_kernel_idx is None:
            dP_task, _, _ = self.pseudogradient_P(trainable_vars[self._task_P_output_idx], h_task)
            self.optimizer.apply_gradients(zip([dP_task], [trainable_vars[self._task_P_output_idx]]))

        dwR_task = self.pseudogradient_wR_task(trainable_vars[self._task_P_output_idx], 
                                               trainable_vars[self._task_recurrent_kernel_idx], 
                                               h_task,
                                               self._target_network.recurrent_kernel,
                                               h_target,
                                               fb_hint_sum) 
        self.optimizer.apply_gradients(zip([dwR_task], [trainable_vars[self._task_recurrent_kernel_idx]]))
    
    def pseudogradient_wR_task(self, P_task, wR_task, h_task, wR_target, h_target, fb_hint_sum): 
        """
        Return pseudogradient for wR for the task network. 

        :param P_task: ``self.units x self.units`` P matrix of the task network
        :type P_task: Tensor[2D floats]
        :param wR_task: ``self.units x self.units`` recurrent kernel of the task network
        :type wR_task: Tensor[2D floats]
        :param h_task: ``1 x self.units`` tensor of neuron firing rates for task network
        :type h_task: Tensor[2D floats]
        :param wR_target: ``self.units x self.units`` recurrent kernel of the target network 
        :type wR_target: Tensor[2D floats]
        :param h_target: ``1 x self.units`` tensor of neuron firing rates for target network
        :type h_target: Tensor[2D floats]
        :param fb_hint_sum: ``1 x self.units`` tensor of sum of inputs from feedback and hint
            components
        :type fb_hint_sum: Tensor[2D floats]
        """
        e = backend.dot(h_task, wR_task) - backend.dot(h_target, wR_target) - fb_hint_sum
        dwR_task = backend.dot(backend.dot(P_task, tf.transpose(h_task)), e)

        return dwR_task 

class OptimizedFORCEModel(FORCEModel):
    """ 
    Optimized version of the FORCEModel class if all recurrent weights in the
    network is trainable. 
     
    If the recurrent kernel is to be trained, then the target must be one dimensional. 
    """

    def initialize_P(self):
        if hasattr(self.original_force_layer, 'output_kernel'):
            if self.original_force_layer.output_kernel.trainable:
                self.P_output = self.add_weight(name='P_output', 
                                                shape=(self.units, self.units), 
                                                initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                                trainable=True)

        if hasattr(self.original_force_layer, 'recurrent_kernel'):
            if self.original_force_layer.recurrent_kernel.trainable:

                bool_mask = self.original_force_layer.recurrent_nontrainable_boolean_mask

                if bool_mask is None or tf.math.count_nonzero(bool_mask) == 0:
                    self.P_GG = self.add_weight(name='P_GG', 
                                                shape=(self.units, self.units), 
                                                initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                                trainable=True)    
                else:
                    identity_3d = np.zeros((self.units, self.units, self.units))
                    idx = np.arange(self.units)
                    identity_3d[:, idx, idx] = self.alpha_P 

                    I,J = np.nonzero(tf.transpose(bool_mask).numpy() == True)
                    identity_3d[I,:,J] = 0
                    identity_3d[I,J,:] = 0

                    self.P_GG = self.add_weight(name='P_GG', 
                                                shape=(self.units, self.units, self.units), 
                                                initializer=keras.initializers.constant(identity_3d), 
                                                trainable=True)
              
    def pseudogradient_wR(self, P_Gx, h, z, y, Ph, hPh):
        # If P is 2D, use optimized update rule
        if len(P_Gx.shape) == 2:
            e = z - y 
            assert e.shape == (1, 1), 'Output must only have 1 dimension'
            dwR_inter = backend.dot(P_Gx, tf.transpose(h)) * e
            return dwR_inter * tf.ones((P_Gx.shape))
        else:
            return super().pseudogradient_wR(P_Gx, h, z, y, Ph, hPh)

    def pseudogradient_P_Gx(self, P_Gx, h):
        if len(P_Gx.shape) == 2:
            return self.pseudogradient_P(P_Gx, h)
        return super().pseudogradient_P_Gx(P_Gx, h)
