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
  are supported (see TENSION documentation).  

* Batching is not supported. Batch size parameter are set to 1 and each timestep is processed
  as its an individual batch. 

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
    history = ffmodel.fit(x=input_with_hint, 
                          y=target, 
                          epochs=5)
    predictions = ffmodel.predict(input_tensor)

Layer and Model Class Compatibilities
-------------------------------------

FORCELayer and FORCEModel class compatibilities are listed below:

* ``base.FORCEModel``

  * ``model.EchoStateNetwork``: supports weight updates for output kernel.

  * ``model.NoFeedbackESN``: supports weight updates for recurrent and 
    output kernel.

* ``model.FullFORCEModel``

  * ``model.EchoStateNetwork``: supports weight updates for output kernel.

  * ``model.NoFeedbackESN``: supports weight updates for recurrent and 
    output kernels. Note that recurrent neurons are assumed fully connected 
    in full-FORCE (no masking out of connections).  

* ``spiking.SpikingNNModel``

  * ``spiking.LIF``: supports weight updates for output kernel.

  * ``spiking.Izhikevich``: supports weight updates for output kernel.

  * ``spiking.Theta``: supports weight updates for output kernel.

* ``constrained.BioFORCEModel``

  * ``constrained.ConstrainedNoFeedbackESN``: supports weight updates for 
    recurrent kernel.

Refer to documentation for initialization parameters.  


Creating New Layers
-------------------

Custom chaotic RNN layers can be created via Keras style subclassing of the 
``base.FORCELayer`` object and defining:

* A custom `call(self, inputs, states)` method that defines the forward pass 
  of the RNN. The call method will return the output and the updated states of 
  the ``FORCELayer`` instance, which by default are a list of 3 tensors of shape 
  ``1`` x ``self.units``, ``1`` x ``self.units``, and ``1`` x ``self.output_size``. 

* A custom `get_initial_state(self, inputs=None, batch_size=None, dtype=None)` 
  method which returns a list of 3 tensors of shape ``1`` x ``self.units``, 
  ``1`` x ``self.units``, and ``1`` x ``self.output_size`` containing the initial
  states of the ``FORCELayer`` instance.

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

If needed, existing ``base.FORCELayer`` methods can be modified via sub-classing
(noting the required input and output as listed in the documentation):

* ``base.FORCEModel`` class only implements update rules for the output
  and recurrent kernels. If a custom FORCELayer requires the recurrent
  kernels to be trainable, it must have an attribute ``self.recurrent_nontrainable_boolean_mask``
  of shape ``self.units`` x ``self.units`` where True indicates that the 
  weights at the corresponding indices in the recurrent kernel is not trainable. 

* Kernel initialization methods `initialize_input_kernel`,
  `initialize_recurrent_kernel`, `initialize_feedback_kernel`, and
  `initialize_output_kernel` may be modified if a different initialization
  scheme is required. The kernels must have names 'input_kernel', 'recurrent_kernel',
  'feedback_kernel', and 'output_kernel' respectively. 

  * If a seed is desired during kernel initialization, then the ``self.seed_gen`` attribute,
    the ``FORCELayer`` instance's `Tensorflow random generator object 
    <https://www.tensorflow.org/api_docs/python/tf/random/Generator>`_, can be used to generate
    a deterministic seed to pass into the Keras initializer. 

  * Alternatively, one can use the `from_weights` method
    to create a layer object with pre-initialized weights. 

* The `build` method which calls the kernel initialization methods may need
  to be modified if kernels are added or removed, as well as initializing any other
  required variables (i.e. ``self.recurrent_nontrainable_boolean_mask``). 

* The classmethod `from_weights` may need to be modified if different (number of) kernels are 
  required and / or if pre-initialized weights are desired to be loaded in.  

* `state_size` may be modified if the default state definition needs to be changed.
  By default, the states of a ``FORCELayer`` are 3 tensors of shape ``1`` x ``self.units``, 
  ``1`` x ``self.units``, and ``1`` x ``self.output_size``.


Creating New Spiking Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating new spiking layers require sub-classing ``spiking.SpikingNN`` or 
``spiking.OptimizedSpikingNN`` and defining the following methods:

* `initialize_voltage(self, batch_size)`: Returns a tensor of shape ``batch_size`` x 
  ``self.units`` of initial voltages for the neurons in the network.

* `update_voltage(self, I, states)`: Returns a list of 3 tensors each of shape   
  ``1`` x ``self.units``. The first result is the voltage trace of each
  neuron, the second an auxillary storage variable that may be unused, and the last
  a 1-0 tensor where 1 in the i-th position indicates that the voltage of the i-th 
  neuron exceeded the peak voltage ``self.v_peak``. 

Creating New Model Classes
--------------------------

Custom FORCE Model classes can be created by sub-classing ``base.FORCEModel``. 


Customizing `train_step`
~~~~~~~~~~~~~~~~~~~~~~~~

See `the Keras guide <https://keras.io/guides/customizing_what_happens_in_fit/>`_ for details on customizing
train_step in a Keras model. The `train_step` method from ``base.FORCEModel`` is reproduced below. 
By default, the method performs the forward pass for one time step and performs weight updates 
for the output and recurrent kernels in ``self.force_layer`` if those two kernels are 
set to be trainable. Below is the default `train_step` method in ``FORCEModel`` class:

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



Customizing `force_layer_call`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default `force_layer_call` method calls ``self.force_layer`` of the ``FORCEModel`` instance
as below:

.. code-block:: python

      def force_layer_call(self, x, training, **kwargs):
          return self.force_layer(x, **kwargs) 

To be compatible with the default `train_step`, the `force_layer_call` method 
of ``base.FORCEModel`` must return 4 tensors, the first of which is ``self.force_layer``'s
output after the forward pass, and the third must be the post-activation firing rates of
the RNN layer's neurons (thus must be a ``1`` x ``self.units`` tensor). By default, it is assumed 
that the call method of ``self.force_layer`` meets these requirements. If
this is not the case, then `force_layer_call` should be adjusted like the example below:

.. code-block:: python

   class SpikingNNModel(FORCEModel):
       def force_layer_call(self, x, training, **kwargs):
           output, t_step, v, u, h, _ , _, _, out =  self.force_layer(x, **kwargs) 
           return output, t_step, h, out

Alternatively, the indicated line below from the default `train_step` method can be adjusted 
to accomodate different output from calling the model during training. 

.. code-block:: python

    def train_step(self, data):
        ...
        z, _, h, _ = self(x, training=True, reset_states=False)
        ...

Customizing Pseudogradient Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For updating the output kernel, `pseudogradient_P` and `pseudogradient_wO` methods return
the pseudogradient updates for the P matrix and output kernel respectively. See documentation
for the required input and outputs. 


For updating the recurrent kernel, `pseudogradient_P_Gx` and `pseudogradient_wR` methods return
the pseudogradient updates for the P matrix corresponding to the recurrent kernel and the
recurrent kernel respectively.  See documentation for the required input and outputs. 


Callbacks
---------

Callbacks can be passed into a model's `fit`, `predict`, and `evaluate` methods 
like with a typical Keras model. See `the custom callbacks article 
<https://www.tensorflow.org/guide/keras/custom_callback>`_
for instructions on writing custom Keras callbacks. Inside the callback object, the layer 
states as a list of tensors can be accessed via ``self.model.force_layer.states``. 

GPU Support
-----------

See `Tensorflow documentation <https://www.tensorflow.org/guide/gpu>`_ for GPU support for Tensorflow.

