FORCE Layers
============


Base FORCE Layer
----------------
.. autoclass:: base.FORCELayer
   :members: initialize_input_kernel, initialize_recurrent_kernel, 
             initialize_feedback_kernel, initialize_output_kernel, 
             build, from_weights  

Echo State Networks
-------------------
.. autoclass:: models.EchoStateNetwork
   :members:
.. autoclass:: models.NoFeedbackESN
   :members:
.. autoclass:: constrained.ConstrainedNoFeedbackESN
   :members:

Spiking Networks
----------------
.. autoclass:: spiking.SpikingNN
   :members:
.. autoclass:: spiking.OptimizedSpikingNN
   :members:
.. autoclass:: spiking.LIF
   :members:
.. autoclass:: spiking.Izhikevich
   :members:
.. autoclass:: spiking.Theta
   :members: