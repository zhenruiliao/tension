
.. _forcelayers:

FORCE Layers
============

Base FORCE Layer
----------------
.. autoclass:: tension.base.FORCELayer
   :members: initialize_input_kernel, initialize_recurrent_kernel, 
             initialize_feedback_kernel, initialize_output_kernel, 
             build, from_weights, call, get_initial_state

Echo State Networks
-------------------
.. autoclass:: tension.models.EchoStateNetwork
   :members:
.. autoclass:: tension.models.NoFeedbackESN
   :members:
.. autoclass:: tension.constrained.ConstrainedNoFeedbackESN
   :members:

Spiking Networks
----------------
.. autoclass:: tension.spiking.SpikingNN
   :members:
.. autoclass:: tension.spiking.OptimizedSpikingNN
   :members:
.. autoclass:: tension.spiking.LIF
   :members:
.. autoclass:: tension.spiking.Izhikevich
   :members:
.. autoclass:: tension.spiking.Theta
   :members: