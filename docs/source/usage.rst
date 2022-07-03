Usage
=====

TENSION uses the Keras interface for training models via the following steps:

1. Initialize an chaotic RNN layer object (a subclass of ``base.FORCELayer`` class).

2. Initialize a model object (typically ``base.FORCEModel`` or its subclass) capable of 
   training the layer object. 

3. Compile the model object with the relevant metrics like a regular Keras model.

4. Fit the model by calling `fit` method of a ``FORCEModel`` instance.

5. If desired, call `predict` or `evaluate` method of a ``FORCEModel`` instance.

**Note:** 

* Not all input parameters from the Keras model fit/predict/evaluate 
  are supported (see :ref:`base-forcemodel` for details).  

* Batching is not supported. Batch size parameters are set to 1 and each timestep is processed
  as an individual batch. 

Quick Start
-----------

The code example below implements FullFORCE by DePasquale et al. on a network
without feedback. 

.. code-block:: python

    from models import FullFORCEModel, NoFeedbackESN
    
    no_fb_esn_layer = NoFeedbackESN(dtdivtau=0.1, 
                                    units=400, 
                                    output_size=1, 
                                    activation='tanh')
    ffmodel = FullFORCEModel(force_layer=no_fb_esn_layer, 
                             target_output_kernel_trainable=False, 
                             hint_dim=1)  
    ffmodel.compile(metrics=["mae"])  
    history = ffmodel.fit(x=inputs_with_hint, 
                          y=target, 
                          epochs=5)
    predictions = ffmodel.predict(inputs)

Layer and Model Class Compatibilities
-------------------------------------

FORCELayer and FORCEModel class compatibilities are listed below:

* ``base.FORCEModel`` and ``model.FullFORCEModel``: 

  * ``model.EchoStateNetwork``: supports weight updates for recurrent and 
    output kernels.

  * ``model.NoFeedbackESN``: supports weight updates for recurrent and 
    output kernels.
  
  * **Note:** In full-FORCE, recurrent neurons are assumed fully connected 
    (no masking out of connections). 

* ``spiking.SpikingNNModel``

  * ``spiking.LIF``: supports weight updates for output kernel.

  * ``spiking.Izhikevich``: supports weight updates for output kernel.

  * ``spiking.Theta``: supports weight updates for output kernel.

* ``constrained.BioFORCEModel``

  * ``constrained.ConstrainedNoFeedbackESN``: supports weight updates for 
    recurrent kernel.

Refer to :ref:`forcelayers` and :ref:`forcemodel` for initialization parameters.  

Accessing Key Attributes
------------------------

``FORCELayer`` Classes
~~~~~~~~~~~~~~~~~~~~~~

* ``self.input_kernel``:  the input weights

* ``self.recurrent_kernel``: the recurrent weights

* ``self.output_kernel``: the output weights

* ``self.feedback_kernel``: the feedback weights 

* ``self.units``: number of recurrent neurons in the layer

* ``self.output_size``: output dimension of the layer

* ``self.state_size``: a list of integers containing the size of each state

**Note:** The layer must be built prior to the weights being accessible. 

``FORCEModel`` Classes
~~~~~~~~~~~~~~~~~~~~~~

* ``self.force_layer.states``: list of the current states in ``force_layer``

Creating New Layers
-------------------

Custom chaotic RNN layers can be created via Keras style subclassing of the 
``base.FORCELayer`` object and defining:

* A custom ``call(self, inputs, states)`` method that defines the forward pass 
  of the RNN. The call method will return the output and a list of tensors 
  corresponding to the updated states of the ``FORCELayer`` instance. 

* A custom ``get_initial_state(self, inputs=None, batch_size=None, dtype=None)``
  method which returns a list of tensors containing the initial
  states of the ``FORCELayer`` instance. 

**Note:** By default, the states are a list of 3 tensors of shape 
``1 x self.units``, ``1 x self.units``, and ``1 x self.output_size``. 
This may be changed by sub-classing and modifying the ``state_size`` property 
(see :ref:`custom_state_size`). 

The example below is from ``model.EchoStateNetwork`` (note that layer states are row 
vectors for matrix multiplication):

.. code-block:: python

   class EchoStateNetwork(FORCELayer):
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

If needed, existing ``base.FORCELayer`` methods can be also modified when sub-classing
(noting the required input and output as listed in the documentation). 

.. _custom_state_size:

Custom State Size
~~~~~~~~~~~~~~~~~	

The ``state_size(self)`` property may be modified if the 
default state definition needs to be changed via sub-classing.
By default (below), the states of a ``FORCELayer`` are 3 
tensors of shape ``1 x self.units``, 
``1 x self.units``, and ``1 x self.output_size``:

.. code-block:: python

    @property
    def state_size(self):
        return [self.units, self.units, self.output_size]

Custom Kernel Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default kernel initialization methods are:

* ``initialize_input_kernel(self, input_dim, input_kernel=None)`` 
  with name `input_kernel` and random normal initialization with mean 
  `0` and variance `1 / input dimension`.
* ``initialize_recurrent_kernel(self, recurrent_kernel=None)`` 
  with name `recurrent_kernel` and with mean `0` and variance 
  `self._g^2 /  self.units`.
* ``initialize_feedback_kernel(self, feedback_kernel=None)`` 
  with name `feedback_kernel` and  random normal initialization 
  with mean `0` and variance `1`.
* ``initialize_output_kernel(self, output_kernel=None)`` 
  with name `output_kernel` and random normal initialization 
  with mean `0` and variance `1 / self.units`.

Kernel initialization may be modified if a different initialization
scheme is required. See an example below for modifying kernel 
initialization via sub-classing:  

.. code-block:: python

    def initialize_recurrent_kernel(self, recurrent_kernel=None):
        #####
        #
        # Code to initialize kernel and save result in a variable named `recurrent_kernel` 
        # 
        #####
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=keras.initializers.constant(recurrent_kernel),
                                                trainable=self._recurrent_kernel_trainable,
                                                name='recurrent_kernel')

If a seed is desired during kernel initialization, then the 
``self.seed_gen`` attribute, the ``FORCELayer`` instance's 
`Tensorflow random generator object 
<https://www.tensorflow.org/api_docs/python/tf/random/Generator>`_, 
can be used to generate a deterministic seed to pass into the 
Keras initializer. 
    
Custom Build
~~~~~~~~~~~~

The ``build(self, input_shape)`` method by default calls the 
kernel initialization methods to build the layer and 
initialize other required variables
(i.e. ``self.recurrent_nontrainable_boolean_mask``). 

``base.FORCEModel`` class only implements update rules for the output
and recurrent kernels. If a custom ``FORCELayer`` requires the 
recurrent kernels to be trainable, it must have an attribute 
``self.recurrent_nontrainable_boolean_mask`` of shape 
``self.units x self.units`` where ``True`` indicates that the 
weights at the corresponding indices in ``self.recurrent_kernel`` 
is not trainable. This should be initialized in the build method. 
See below for the default ``build`` method from ``base.FORCELayer``:

.. code-block:: python

    def build(self, input_shape):
        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        self.initialize_output_kernel() 
        self.initialize_feedback_kernel()
        if self._p_recurr == 1:
            self.recurrent_nontrainable_boolean_mask = None
        else:
            self.recurrent_nontrainable_boolean_mask = (self.recurrent_kernel == 0)
        self.built = True

**Note:** After the layer is built, ``self.built`` must be set to ``True``.

Custom ``from_weights``
~~~~~~~~~~~~~~~~~~~~~~~

One can use the classmethod ``from_weights(self, ...)`` to create a layer 
object with weights pre-initialized externally. This method may need to be 
modified if different (number of) kernels are required to be loaded in. 
If the recurrent kernel is set to be trainable, then
``self.recurrent_nontrainable_boolean_mask`` must be initialized in this method. 
See below for default ``from_weights`` method from ``base.FORCELayer``:

.. code-block:: python

    @classmethod
    def from_weights(cls, weights, recurrent_nontrainable_boolean_mask, **kwargs):
        input_kernel, recurrent_kernel, feedback_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        feedback_output_size, feedback_units = feedback_kernel.shape 
        output_units, output_size = output_kernel.shape 
        units = input_units 

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, 
                            feedback_units, output_units]) == units)
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

**Note:** ``self.built`` must be set to ``True`` in this method.

Creating New Spiking Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating new spiking layers require sub-classing ``spiking.SpikingNN`` or 
``spiking.OptimizedSpikingNN`` and defining, at minimum, the following methods:

* ``initialize_voltage(self, batch_size)``: Returns a tensor of shape ``batch_size x self.units`` 
  of initial voltages for the neurons in the network.

* ``update_voltage(self, I, states)``: Returns a list of 3 tensors each of shape   
  ``1 x self.units``. The first result is the voltage trace of each
  neuron, the second an auxillary storage variable that may be unused, and the last
  a 1-0 tensor where 1 in the i-th position indicates that the voltage of the i-th 
  neuron exceeded the peak voltage ``self.v_peak``. 

Creating New Model Classes
--------------------------

Custom FORCE Model classes can be created using Keras style sub-classing of 
``base.FORCEModel``. 


Customizing ``train_step``
~~~~~~~~~~~~~~~~~~~~~~~~~~

See `the Keras guide <https://keras.io/guides/customizing_what_happens_in_fit/>`_ for details on customizing
``train_step`` in a Keras model. The ``train_step(self, data)`` method from ``base.FORCEModel`` is reproduced below. 
By default, the method performs the forward pass for one time step and performs weight updates 
for the output and recurrent kernels in ``self.force_layer`` (if those two kernels are 
set to be trainable). Below is the default ``train_step`` method from ``base.FORCEModel``:

.. code-block:: python

    def train_step(self, data):
        x, y = data
        z, _, h, _ = self(x, training=True, reset_states=False)

        if self.force_layer.return_sequences:
          z = z[:,0,:]

        trainable_vars = self.trainable_variables

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



Customizing ``force_layer_call``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default `force_layer_call` method calls ``self.force_layer`` of the 
``FORCEModel`` instance:

.. code-block:: python

      def force_layer_call(self, x, training, **kwargs):
          return self.force_layer(x, **kwargs) 

To be compatible with the default `train_step`, the `force_layer_call` method 
must return 4 tensors, the first of which is ``self.force_layer``'s
network output after the forward pass, and the third must be the post-activation firing rates of
the layer's neurons (and thus a ``1 x self.units`` tensor). By default, it is assumed 
that the call method of ``self.force_layer`` meets these requirements. If
this is not the case, then ``force_layer_call`` can be adjusted like the example below:

.. code-block:: python

   class SpikingNNModel(FORCEModel):
       def force_layer_call(self, x, training, **kwargs):
           output, t_step, v, u, h, _ , _, _, out =  self.force_layer(x, **kwargs) 
           return output, t_step, h, out

Alternatively, the indicated line below from the default ``train_step`` 
method can be adjusted to accomodate different output during calling 
the model in training. 

.. code-block:: python

    def train_step(self, data):
        ...
        z, _, h, _ = self(x, training=True, reset_states=False)
        ...

Customizing Pseudogradient Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For updating the output kernel, ``pseudogradient_P(self,...)`` and 
``pseudogradient_wO(self,...)`` methods return the pseudogradient 
updates for the P matrix and output kernel respectively. 
See method documentation in :ref:`forcemodel` for input and required
outputs of this method to be compatible with ``base.FORCEModel``.


For updating the recurrent kernel, `pseudogradient_P_Gx` and 
`pseudogradient_wR` methods return the pseudogradient updates for the 
P matrix corresponding to the recurrent kernel and the recurrent 
kernel respectively.  See method documentation in :ref:`forcemodel` 
for input and required outputs of this method to be compatible
with ``base.FORCEModel``. 


Callbacks
---------

Callbacks can be passed into a model's `fit`, `predict`, and `evaluate` methods 
like with a typical Keras model. See `the custom callbacks article 
<https://www.tensorflow.org/guide/keras/custom_callback>`_
for instructions on writing custom Keras callbacks. Inside the callback object, the layer 
states as a list of tensors can be accessed via ``self.model.force_layer.states``. 

GPU Support
-----------

TENSION models can use GPU like any normal Keras model. 
See `Tensorflow documentation <https://www.tensorflow.org/guide/gpu>`_ 
for GPU support for Tensorflow.

