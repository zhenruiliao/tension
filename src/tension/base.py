import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations
import abc

class FORCELayer(keras.layers.AbstractRNNCell, metaclass=abc.ABCMeta):
    """ 
    Base class for FORCE layers

    :param units: Number of recurrent units.
    :type units: int
    :param output_size: Dimension of the target. 
    :type output_size: int
    :param activation: Activation function applied to the state updates. Can be a string (i.e. 'tanh') or a function.  
    :type activation: str or function
    :param seed: Seed for random initialization (i.e. weights and initial states). (*Default: None*)
    :type seed: int or None
    :param g: Gain parameter that controls the chaos (by default, only scales the strengths of the 
        recurrent connections within the network at initialization). (*Default: 1.5*)
    :type g: float
    :param input_kernel_trainable: If True, sets the input kernel to be a trainable variable. (*Default: False*)
    :type input_kernel_trainable: bool
    :param recurrent_kernel_trainable: If True, sets the recurrent kernel to be a trainable variable. (*Default: False*)
    :type recurrent_kernel_trainable: bool
    :param output_kernel_trainable: If True, sets the output kernel to be a trainable variable. (*Default: True*)
    :type output_kernel_trainable: bool
    :param feedback_kernel_trainable: If True, sets the feedback kernel to be a trainable variable. (*Default: False*)
    :type feedback_kernel_trainable: bool
    :param p_recurr: Recurrent kernel initialization parameter where larger values means denser  
        connections (more non-zero weights). Value must be in (0,1]. (*Default: 1*)
    :type p_recurr: float 

    :states:
        - **a** - ``1 x self.units`` tensor containing the pre-activation neuron firing rates
        - **h** - ``1 x self.units`` tensor containing the neuron firing rates
        - **output** - ``1 x self.output_size`` tensor containing the predicted output  
    """

    def __init__(self, 
                 units, 
                 output_size, 
                 activation, 
                 seed=None, 
                 g=1.5, 
                 input_kernel_trainable=False, 
                 recurrent_kernel_trainable=False, 
                 output_kernel_trainable=True, 
                 feedback_kernel_trainable=False, 
                 p_recurr=1, 
                 **kwargs):
                
        self.units = units 
        self._output_size = output_size
        self.activation = activations.get(activation)

        if seed is None:
          self.seed_gen = tf.random.Generator.from_non_deterministic_state()
        else:
          self.seed_gen = tf.random.Generator.from_seed(seed)

        self._seed = seed
        self._g = g
        self._input_kernel_trainable = input_kernel_trainable
        self._recurrent_kernel_trainable = recurrent_kernel_trainable
        self._feedback_kernel_trainable = feedback_kernel_trainable
        self._output_kernel_trainable = output_kernel_trainable
        self._p_recurr = p_recurr
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return [self.units, self.units, self.output_size]

    @property 
    def output_size(self):
        return self._output_size

    def initialize_input_kernel(self, input_dim, input_kernel=None):
        """
        Initializes the input kernel. Called in the layer object's ``build`` and ``from_weights``
        methods.

        :param input_dim: Dimension of input
        :type input_dim: int
        :param input_kernel: Tensor containing the pre-initialized kernel. 
            If none, the kernel will be randomly initialized. (*Default: None*)
        :type input_kernel: Tensor[2D float]
        """
        if input_kernel is None:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=1 / input_dim**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                     minval=None, 
                                                                                     dtype=tf.dtypes.int64)[0])
            
            input_kernel = initializer(shape=(input_dim, self.units))
         
        self.input_kernel = self.add_weight(shape=(input_dim, self.units),
                                            initializer=keras.initializers.constant(input_kernel),
                                            trainable=self._input_kernel_trainable,
                                            name='input_kernel')
        
    def initialize_recurrent_kernel(self, recurrent_kernel=None):
        """
        Initializes the recurrent kernel. Called in the layer object's ``build`` and ``from_weights``
        methods.

        :param recurrent_kernel: Tensor containing the pre-initialized kernel. 
            If none, the kernel will be randomly initialized. (*Default: None*)
        :type recurrent_kernel: Tensor[2D float]  
        """
        if recurrent_kernel is None:        
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=self._g / self.units**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                      minval=None, 
                                                                                      dtype=tf.dtypes.int64)[0])
        
            recurrent_kernel = self._p_recurr * keras.layers.Dropout(1 - self._p_recurr)(initializer(shape=(self.units, self.units)), 
                                                                                         training=True)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=keras.initializers.constant(recurrent_kernel),
                                                trainable=self._recurrent_kernel_trainable,
                                                name='recurrent_kernel')
    
    def initialize_feedback_kernel(self, feedback_kernel=None):
        """
        Initializes the feedback kernel. Called in the layer object's ``build`` and ``from_weights``
        methods.

        :param feedback_kernel: Tensor array containing the pre-initialized kernel. 
            If none, the kernel will be randomly initialized. (*Default: None*)
        :type feedback_kernel: Tensor[2D float]
        """
        if feedback_kernel is None:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=1, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                     minval=None, 
                                                                                     dtype=tf.dtypes.int64)[0])
            feedback_kernel = initializer(shape=(self.output_size, self.units))

        self.feedback_kernel = self.add_weight(shape=(self.output_size, self.units),
                                               initializer=keras.initializers.constant(feedback_kernel),
                                               trainable=self._feedback_kernel_trainable,
                                               name='feedback_kernel')
                                            
    def initialize_output_kernel(self, output_kernel=None):
        """
        Initializes the output kernel. Called in the layer object's ``build`` and ``from_weights``
        methods.

        :param output_kernel: Tensor or numpy array containing the pre-initialized kernel. 
            If none, the kernel will be randomly initialized. (*Default: None*)
        :type output_kernel: Tensor[2D float]
        """
        if output_kernel is None:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=1 / self.units**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                     minval=None, 
                                                                                     dtype=tf.dtypes.int64)[0])
            output_kernel = initializer(shape=(self.units, self.output_size))

        self.output_kernel = self.add_weight(shape=(self.units, self.output_size),
                                             initializer=keras.initializers.constant(output_kernel),
                                             trainable=self._output_kernel_trainable,
                                             name='output_kernel')     
    
    def build(self, input_shape):
        """
        Inherited from ``tensorflow.keras.layers.Layer.build``, also calls method that performs the 
        initialization of layer kernals. Typically called implicitly.

        :param input_shape: Shape of the input tensor. 
        :type input_shape: tuple
        """
        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        self.initialize_output_kernel() 
        self.initialize_feedback_kernel()
        if self._p_recurr == 1:
            self.recurrent_nontrainable_boolean_mask = None
        else:
            self.recurrent_nontrainable_boolean_mask = (self.recurrent_kernel == 0)
        self.built = True

    @classmethod
    def from_weights(cls, weights, recurrent_nontrainable_boolean_mask, **kwargs):
        """
        Creates a layer object with pre-initialized weights. 

        **Note:** ``p_recurr`` parameter is not supported in this method. ``units`` and 
        ``output_size`` parameters are inferred from the input weights. 

        :param weights: Four 2D Tensors containing, respectively, the input, 
            recurrent, feedback, and output kernels  
        :type weights: tuple[Tensor[2D float]] of length 4
        :param recurrent_nontrainable_boolean_mask: A 2D boolean array with
            the same shape as the recurrent kernel, where True indicates that
            the corresponding weight in the recurrent kernel is not trainable. 
        :type recurrent_nontrainable_boolean_mask: Tensor[2D bool]
        :param kwargs: Additional parameters required to initialize the layer. 

        :returns: A sub-classed ``FORCELayer`` object initialized with the input weights
        """
        input_kernel, recurrent_kernel, feedback_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        feedback_output_size, feedback_units = feedback_kernel.shape 
        output_units, output_size = output_kernel.shape 
        units = input_units 

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, feedback_units, output_units]) == units)
        assert feedback_output_size == output_size, 'feedback and output kernel dimensions are inconsistent' 
        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'
        assert recurrent_kernel.shape == recurrent_nontrainable_boolean_mask.shape, "Boolean mask and recurrent kernel shape mis-match"
        if tf.math.count_nonzero(tf.boolean_mask(recurrent_kernel, recurrent_nontrainable_boolean_mask)).numpy() != 0:
            print("Warning: Recurrent kernel has non-zero weights (indicating neuron connection) that are not trainable")  

        self = cls(units=units, output_size=output_size, p_recurr=None, **kwargs)
        self.recurrent_nontrainable_boolean_mask = tf.convert_to_tensor(recurrent_nontrainable_boolean_mask)
        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.initialize_feedback_kernel(feedback_kernel)
        self.initialize_output_kernel(output_kernel)
        self.built = True

        return self

    @abc.abstractmethod
    def call(self, inputs, states):
        """
        Implements forward pass of the layer. 

        :param inputs: Input tensor of shape *(1, input dimensions)*.
        :type inputs: Tensor[2D float]
        :param states: List of tensors corresponding to the states of the layer.
        :type states: list[Tensor[2D float]]

        :returns:
            - **output** (*Tensor[2D float]*) - ``1 x self.output_size`` tensor containing the 
              output of the forward pass.
            - **updated states** (*list[Tensor[2D float]]*) - List of tensors containing the
              updated states of the layer.
        """
        return

    @abc.abstractmethod
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Initializes the states of the layer (called implicitly during layer build). 
        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

        :returns: A list of tensors containing the initial layer states 
        """
        return

class FORCEModel(keras.Model):
    """ 
    Base class for FORCE model per `Sussillo and Abbott 
    <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/>`_. 
    
    Input to the model during ``FORCEModel``'s `fit`, `predict`, or `evaluate` methods  
    should be of shape ``timesteps x input dimension`` or ``batch size x timesteps x input dimension``.
    If the latter is used, the batches are stacked. 

    Target to the model during `fit` or `evaluate` should be of shape ``timesteps x output dimension``
    or ``batch size x timesteps x output dimension``.

    If validation data is to be passed to `fit`, input and target of the validation 
    data should be of shape ``timesteps x input dimension`` and ``timesteps x output dimension`` 
    respectively or ``batch size x timesteps x input dimensions`` and 
    ``batch size x timesteps x output dimension`` respectively.

    Input to the model when calling the model should be of shape ``1 x timesteps x input dimensions``.

    If the recurrent kernel is to be trained, then the target must be one dimensional. 

    Use ``self.force_layer.states`` to access the states of ``self.force_layer``.

    :param force_layer: A FORCE Layer object or one of its subclasses.
    :type force_layer: class[FORCELayer] or equiv. 
    :param alpha_P: Constant parameter for initializing the P matrix. (*Default: 1.0*)
    :type alpha_P: float
    :param return_sequences: If True, target is returned as a sequence of the same 
        length as the number of time steps in the input. (*Default: True*) **Note:**
        must be ``True`` for compatibility with existing layers of this package. 
    :type return_sequences: bool
    """

    def __init__(self, 
                 force_layer, 
                 alpha_P=1.,
                 return_sequences=True):
        super().__init__()
        self.alpha_P = alpha_P
        self.force_layer = keras.layers.RNN(force_layer, 
                                            stateful=True, 
                                            return_state=True, 
                                            return_sequences=return_sequences)
        self.units = force_layer.units 
        self.original_force_layer = force_layer

    def build(self, input_shape):
        """ 
        Inherited from ``tensorflow.keras.Model.build``, also calls method that performs the 
        initialization of variables the model requires for training. Typically called implicitly.

        :param input_shape: Shape of the input tensor. 
        :type input_shape: tuple
        """
        super().build(input_shape)
        self.initialize_P()
        self.initialize_train_idx()

    def initialize_P(self):
        """ 
        Initializes the P matrices corresponding to the output and recurrent kernel necessary for FORCE. 
        """
        # P matrix for updating the output kernel
        if hasattr(self.original_force_layer, 'output_kernel'):
            if self.original_force_layer.output_kernel.trainable:
                self.P_output = self.add_weight(name='P_output', 
                                                shape=(self.units, self.units), 
                                                initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                                trainable=True)

        if hasattr(self.original_force_layer, 'recurrent_kernel'):
            if self.original_force_layer.recurrent_kernel.trainable:
                identity_3d = np.zeros((self.units, self.units, self.units))
                idx = np.arange(self.units)
                identity_3d[:, idx, idx] = self.alpha_P 

                if self.original_force_layer.recurrent_nontrainable_boolean_mask is not None:
                    # Transpose here is to be consistent with recurrent pseudogradient calculations
                    I,J = np.nonzero(tf.transpose(self.original_force_layer.recurrent_nontrainable_boolean_mask).numpy() == True)
                    identity_3d[I,:,J] = 0
                    identity_3d[I,J,:] = 0

                # P matrix for updating the recurrent kernel
                self.P_GG = self.add_weight(name='P_GG', 
                                            shape=(self.units, self.units, self.units), 
                                            initializer=keras.initializers.constant(identity_3d), 
                                            trainable=True)

    def initialize_train_idx(self):
        """ 
        Finds the indices inside ``self.trainable_variables`` corresponding to the relevant kernel and P tensors.
        """
        self._output_kernel_idx = None
        self._recurrent_kernel_idx = None

        # Find the index of each trainable weights in list of trainable variables
        for idx in range(len(self.trainable_variables)):
            trainable_name = self.trainable_variables[idx].name
              
            if 'output_kernel' in trainable_name:
                self._output_kernel_idx = idx
            elif 'P_output' in trainable_name:
                self._P_output_idx = idx
            elif 'P_GG' in trainable_name:
                self._P_GG_idx = idx
            elif 'recurrent_kernel' in trainable_name:
                self._recurrent_kernel_idx = idx

    def call(self, x, training=False, reset_states=False, **kwargs):
        """
        Inherited and serves the same function as from `tensorflow.keras.Model.call`. 

        **Note:** By default, ``reset_states=False`` and therefore, the ``self.force_layer``'s 
        states are **not** reset at the end of the execution. 

        :param x: Input tensor of shape *(1, timestep, input dimension)*.
        :type x: Tensor[3D float]
        :param training: Whether the model is being called during training or inference. (*Default: False*)
        :type training: bool
        :param reset_states: Whether to reset the state of ``self.force_layer`` after the model is called.
            (*Default: False*)
        :type reset_states: bool
        :param kwargs: Other key word arguments as needed. 
        """
        if not reset_states:
            output = self.force_layer_call(x, training, **kwargs)
        else:
            initialization = all(v is None for v in self.force_layer.states)
            
            # if a state exists, store it
            if not initialization:
                original_state = [tf.identity(state) for state in self.force_layer.states]
            output = self.force_layer_call(x, training, **kwargs)

            # reset the state
            if not initialization:
                for i, state in enumerate(self.force_layer.states):
                    state.assign(original_state[i], read_value=False)

        if training:
            return output
        else:
            return output[0]

    def force_layer_call(self, x, training, **kwargs):
        """
        Calls ``self.force_layer``; to be customized via sub-classing.  

        :param x: Input tensor.
        :type x: Tensor[3D float]
        :param training: If True, indicates that this function is being called during training. 
        :type training: bool
        :param kwargs: Other key word arguments as needed. 

        :returns: (*tuple[Tensor[2D float]]*) - Output of ``self.force_layer``'s call method.
        """
        return self.force_layer(x, **kwargs) 

    def update_kernel_condition(self):
        """
        Returns a scalar boolean tensor indicating whether weight updates should be performed or not. 
        This is called once per ``train_step``. 

        :returns: (*Tensor[bool]*) - If ``True``, perform weight updates. (*Default : tf.constant(True)*)
        """
        return tf.constant(True)

    @tf.function
    def train_step(self, data):
        """
        Inherited and serves the same function as from `tensorflow.keras.Model.train_step`. Performs
        weight updates for the output and recurrent kernels. 

        Intended to be customized via Keras style sub-classing. For more details see:
        https://keras.io/guides/customizing_what_happens_in_fit/
        """
        x, y = data
        z, _, h, _ = self(x, training=True, reset_states=False)

        if self.force_layer.return_sequences:
            z = z[:,0,:]

        trainable_vars = self.trainable_variables

        if tf.cond(self.update_kernel_condition(), lambda : True, lambda : False):
            # Update the output kernel
            if self._output_kernel_idx is not None:
                self.update_output_kernel(h, 
                                          z, 
                                          y[:,0,:], 
                                          trainable_vars[self._P_output_idx], 
                                          trainable_vars[self._output_kernel_idx])
          
            # Update the recurrent kernel
            if self._recurrent_kernel_idx is not None:
                self.update_recurrent_kernel(h, 
                                             z, 
                                             y[:,0,:],
                                             trainable_vars[self._P_GG_idx],
                                             trainable_vars[self._recurrent_kernel_idx])
          
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y[:,0,:], z)

        return {m.name: m.result() for m in self.metrics}

    def update_output_kernel(self, h, z, y, trainable_vars_P_output, trainable_vars_output_kernel):
        """ 
        Performs pseudogradient updates for output kernel and its corresponding P tensor.
            
        :param h: An ``1 x self.units`` tensor of firing ratings for each recurrent neuron.
        :type h: Tensor[2D float]
        :param z: An ``1 x output dimensions`` tensor of predictions. 
        :type z: Tensor[2D float]
        :param y: An ``1 x output dimensions`` tensor of ground truth target.
        :type y: Tensor[2D float]
        :param trainable_vars_P_output: The FORCE P tensor corresponding to the output kernel.
        :type trainable_vars_P_output: Tensor[2D float]
        :param trainable_vars_output_kernel: Output kernel of ``self.force_layer``.
        :type trainable_vars_output_kernel: Tensor[2D float]
        """
        # Compute pseudogradients and update the P matrix of the output kernel
        dP, Pht, hPht = self.pseudogradient_P(trainable_vars_P_output, h)
        self.optimizer.apply_gradients(zip([dP], [trainable_vars_P_output]))

        # Compute pseudogradient and update output kernel
        dwO = self.pseudogradient_wO(trainable_vars_P_output, h, z, y, Pht, hPht)
        self.optimizer.apply_gradients(zip([dwO], [trainable_vars_output_kernel]))

    def pseudogradient_P(self, P, h):
        """ 
        Returns pseudogradient for the P tensor. Can be customized as needed if different
        update rule is required. 

        Example array shapes: 

        | h : ``1 x self.units``
        | P : ``self.units x self.units``
        | k : ``self.units x 1`` 
        | hPht : ``1 x 1``
        | dP : ``self.units x self.units``

        :param P: The FORCE P tensor.
        :type P: Tensor[2D float]
        :param h: An ``1 x self.units`` tensor of firing ratings for each recurrent neuron.
        :type h: Tensor[2D float]

        :returns:
            - **dP** (*Tensor[2D float]*) - A ``self.units x self.units`` pseudogradient 
              of the FORCE P tensor.
            - **Pht** (*Tensor[2D float]*) - ``self.units x 1`` intermediate tensor from 
              pseudogradient computation.
            - **hPht** (*Tensor[2D float]*) - ``1 x 1`` intermediate tensor from pseudogradient 
              computation.
        """
        Pht = backend.dot(P, tf.transpose(h))
        hPht = backend.dot(h, Pht)
        c = 1. / (1. + hPht)
        dP = backend.dot(c * Pht, tf.transpose(Pht))

        return dP, Pht, hPht 

    def pseudogradient_wO(self, P, h, z, y, Pht, hPht):
        """ 
        Return pseudogradient for wO. Can be customized as needed if different
        update rule is required. 

        Example array shapes:

        | P : ``self.units x self.units``
        | h : ``1 x self.units``
        | z, y : ``1 x output dimension``

        :param P: The FORCE P tensor corresponding to output kernel.
        :type P: Tensor[2D float]
        :param h: An ``1 x self.units`` tensor of firing ratings for each recurrent neuron.
        :type h: Tensor[2D float]
        :param z: An ``1 x output dimensions`` tensor of predictions.
        :type z: Tensor[2D float]
        :param y: An ``1 x output dimensions`` tensor of ground truth target.
        :type y: Tensor[2D float]
        :param Pht: ``self.units x 1`` intermediate tensor from pseudogradient computation 
            in the `pseudogradient_P` method (unused by default).
        :type Pht: Tensor[2D float]
        :param hPht: ``1 x 1`` intermediate tensor from pseudogradient computation
            in the `pseudogradient_P` method (unused by default).
        :type hPht: Tensor[2D float]

        :returns: **dwO** (*Tensor[2D float]*) - Weight updates for the output kernel.
        """
        e = z - y
        Pht = backend.dot(P, tf.transpose(h))
        dwO = backend.dot(Pht, e)

        return dwO

    def update_recurrent_kernel(self, h, z, y, trainable_vars_P_Gx, trainable_vars_recurrent_kernel):
        """ 
        Performs pseudogradient updates for recurrent kernel and its corresponding P tensor
            
        :param h: An ``1 x self.units`` tensor of firing ratings for each recurrent neuron
        :type h: Tensor[2D float]
        :param z: An ``1 x output dimensions`` tensor of predictions  
        :type z: Tensor[2D float]
        :param y: An ``1 x output dimensions`` tensor of ground truth target
        :type y: Tensor[2D float]
        :param trainable_vars_P_Gx: A ``self.units x self.units x self.units`` P tensor 
            corresponding to the recurrent kernel 
        :type trainable_vars_P_Gx: Tensor[3D float]
        :param trainable_vars_recurrent_kernel: A ``self.units x self.units`` tensor 
            corresponding to the force layer's recurrent kernel
        :type trainable_vars_recurrent_kernel: Tensor[2D float]
        """
        # Compute pseudogradients and update the P matrix for the recurrent kernel
        dP_Gx, Ph, hPh = self.pseudogradient_P_Gx(trainable_vars_P_Gx, h)
        self.optimizer.apply_gradients(zip([dP_Gx], [trainable_vars_P_Gx]))

        # Compute pseudogradient and update recurrent kernel
        dwR = self.pseudogradient_wR(trainable_vars_P_Gx, h, z, y, Ph, hPh)
        self.optimizer.apply_gradients(zip([dwR], [trainable_vars_recurrent_kernel]))

    def pseudogradient_P_Gx(self, P_Gx, h):
        """ 
        Returns pseudogradient for P corresponding to recurrent kernel

        :param P_Gx: A ``self.units x self.units x self.units``  P tensor corresponding to
            the recurrent kernel. 
        :type P_Gx: Tensor[3D float]
        :param h: An ``1 x self.units`` tensor of firing ratings for each recurrent neuron
        :type h: Tensor[2D float] 

        :returns:
            - **dP_Gx** (*Tensor[3D float]*) - A ``self.units x self.units x self.units``  tensor
              that is the pseudogradient of the FORCE P tensor corresponding to the recurrent
              kernel.
            - **Ph** (*Tensor[2D float]*) - Intermediate tensor from pseudogradient computation.
            - **hPh** (*Tensor[3D float]*) - Intermediate tensor from pseudogradient computation.
        """
        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
        hPh = tf.expand_dims(backend.dot(Ph, tf.transpose(h)), axis = 2)
        dP_Gx = tf.expand_dims(Ph, axis = 2) * tf.expand_dims(Ph, axis = 1) / (1 + hPh)

        return dP_Gx, Ph, hPh
 
    def pseudogradient_wR(self, P_Gx, h, z, y, Ph, hPh):
        """ 
        Return pseudogradient for wR

        :param P_Gx: A ``self.units x self.units x self.units`` P tensor corresponding to
            the recurrent kernel. 
        :type P_Gx: Tensor[3D float]
        :param h: A ``1 x self.units`` tensor of firing ratings for each recurrent neuron. 
        :type h: Tensor[2D float]
        :param z: A ``1 x 1`` tensor of predictions.
        :type z: Tensor[2D float]
        :param y: A ``1 x 1`` tensor of ground truth target.
        :type y: Tensor[2D float]
        :param Ph: Intended to match output from the `pseudogradient_P_Gx` method (unused by default).
        :type Ph: Tensor[2D float]
        :param hPh: Intended to match output from `pseudogradient_P_Gx` method (unused by default).
        :type hPh: Tensor[3D float]

        :returns: **dwR** (*Tensor[2D float]*) - Weight updates for the recurrent kernel.
        """
        e = z - y 
        assert e.shape == (1, 1), 'Output must only have 1 dimension'
        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
        dwR = Ph * e 

        return tf.transpose(dwR) 

    def compile(self, metrics, **kwargs):
        """
        A wrapper around `tensorflow.keras.Model.compile`.

        **Note:** The *optimizer* and *loss* parameters are not supported. 

        :param metrics: Same as in `tensorflow.keras.Model.compile`.
        :type metrics: list[str] 
        """
        super().compile(optimizer=keras.optimizers.SGD(learning_rate=1), loss='mae', metrics=metrics, **kwargs)

    def _coerce_input_shape(self, inp, inp_type, method, **kwargs):
        """
        """
        if len(inp.shape) < 2 or len(inp.shape) > 3:
            raise ValueError(f'Shape of {inp_type} is invalid')
        
        if len(inp.shape) == 3:
            inp = tf.reshape(inp, shape=(-1, inp.shape[-1]))

        if len(inp.shape) == 2:
            inp = tf.expand_dims(inp, axis=1)
        
        if inp.shape[1] != 1:
            raise ValueError(f"Dim 1 of {inp_type} must be 1")

        return inp

    def coerce_input_data(self, x, y, method, **kwargs):
        """
        Coerces the input data to a desired shape based on method used 
        (`fit`, `predict`, `evaluate`). 

        :param x: Tensor of shape ``timestep x 1 x input dimensions``
        :type x: Tensor[3D float]
        :param y: Tensor of shape ``timestep x 1 x target dimensions``
        :type y: Tensor[3D float]
        :param method: One of *fit*, *validation*, *predict*, or *evaluate* 
        :type method: str
        :param kwargs: Other arguments as needed. 
        """
        if y is not None:
            assert x.shape[0] == y.shape[0], 'Timestep dimension must match'
        return x, y

    def fit(self, x, y=None, epochs=1, verbose='auto', **kwargs):
        """
        A wrapper around `tensorflow.keras.Model.fit`. 
        **Note:** the *batch_size*, *validation_batch_size*,  *shuffle*, 
        *steps_per_epoch*, and *validation_steps* parameters are not supported. 

        :param x: Tensor of input signal of shape ``timesteps x input dimensions`` or
            ``batch size x timesteps x input dimensions``. 
        :type x: Tensor[2D or 3D float]
        :param y: Tensor of target signal of shape ``timesteps x output dimensions`` or
            ``batch size x timesteps x input dimensions``. 
        :type y: Tensor[2D or 3D float]
        :param epoch: Number of epochs to train. 
        :type epoch: int
        :param verbose: Same as from `tensorflow.keras.Model.fit`.
        :type verbose: str
        :param kwargs: Other key word arguments as needed.        
        """
        if y is not None:
            assert len(x.shape) == len(y.shape), 'Input x and y shape are mismatched'
            assert x.shape[0] == y.shape[0], 'Leading input dimension are mismatched'
        x = self._coerce_input_shape(inp=x, inp_type='x', method='fit', **kwargs)
        y = self._coerce_input_shape(inp=y, inp_type='y', method='fit', **kwargs)
        x, y = self.coerce_input_data(x=x, y=y, method='fit', **kwargs)

        if 'validation_data' in kwargs.keys() and kwargs['validation_data'] is not None:
            val_x, val_y = kwargs['validation_data']
            if val_y is not None:
                assert len(val_x.shape) == len(val_y.shape), 'Validation input shapes are mismatched'
                assert val_x.shape[0] == val_y.shape[0], 'Leading input dimension are mismatched'

            val_x = self._coerce_input_shape(inp=val_x, inp_type='val_x', method='validation', **kwargs)
            val_y = self._coerce_input_shape(inp=val_y, inp_type='val_y', method='validation', **kwargs)
            val_x, val_y = self.coerce_input_data(x=val_x, y=val_y, method='validation', **kwargs)
            kwargs['validation_data'] = (val_x, val_y)

        return super().fit(x=x, 
                           y=y, 
                           epochs=epochs, 
                           batch_size=1, 
                           shuffle=False, 
                           verbose=verbose, 
                           validation_batch_size=1,
                           **kwargs)

    def predict(self, x, **kwargs):
        """
        A wrapper around `tensorflow.keras.Model.predict`. 
        **Note**: the *batch_size* and *steps* parameter are not supported.  

        :param x: Tensor of input signal of shape ``timesteps x input dimensions`` or
            ``batch size x timesteps x input dimensions``. 
        :type x: Tensor[2D or 3D float]

        :returns: (*Array[2D or 3D float]*) - Numpy array of predictions
        """
        x_shape = x.shape
        x = self._coerce_input_shape(inp=x, inp_type='x', method='predict', **kwargs)
        x, _ = self.coerce_input_data(x=x, y=None, method='predict', **kwargs)

        assert self.built, 'Error: Model not built'
        original_state = [tf.identity(state) for state in self.force_layer.states]
        output = super().predict(x=x, batch_size=1, **kwargs)[:,0,:] 
        for i, state in enumerate(self.force_layer.states):
            state.assign(original_state[i], read_value=False)

        if len(x_shape) == 3:
        	output = tf.reshape(output, shape=(x_shape[0], x_shape[1], -1))

        return output

    def evaluate(self, x, y, **kwargs):
        """
        A wrapper around `tensorflow.keras.Model.evaluate`. 
        **Note**: the *batch_size* and *steps* parameter are not supported.  

        :param x: Tensor of input signal of shape ``timesteps x input dimensions`` or
            ``batch size x timesteps x input dimensions``. 
        :type x: Tensor[2D or 3D float]
        :param y: Tensor of target signal of shape ``timesteps x output dimensions``  or
            ``batch size x timesteps x input dimensions``. 
        :type y: Tensor[2D or 3D float]

        :returns: (*List[float]*) - List of error / metrics evaluated on the input
        """
        if 'batch_size' not in kwargs.keys() or kwargs['batch_size'] is None:
            kwargs['batch_size'] = 1
        elif kwargs['batch_size'] != 1:
            raise ValueError('Batch size must be 1')

        if y is not None:
            assert len(x.shape) == len(y.shape), 'Input x and y shape are mismatched'
            assert x.shape[0] == y.shape[0], 'Leading input dimension are mismatched'
        x = self._coerce_input_shape(inp=x, inp_type='x', method='evaluate', **kwargs)
        y = self._coerce_input_shape(inp=y, inp_type='y', method='evaluate', **kwargs)
        x, y = self.coerce_input_data(x=x, y=y, method='evaluate', **kwargs)

        assert self.built, 'Error: Model not built'
        original_state = [tf.identity(state) for state in self.force_layer.states]
        output = super().evaluate(x=x, y=y, **kwargs)
        for i, state in enumerate(self.force_layer.states):
            state.assign(original_state[i], read_value=False)

        return output