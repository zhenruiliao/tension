.. _forcemodel:

FORCE Model
===========

.. _base-forcemodel:


Base FORCE Model
----------------
.. autoclass:: tension.base.FORCEModel
   :members: build, initialize_P, initialize_train_idx, call, 
             force_layer_call, train_step, update_output_kernel,
             pseudogradient_P, pseudogradient_wO, update_recurrent_kernel,
             pseudogradient_P_Gx, pseudogradient_wR, compile, fit, predict,
             evaluate, coerce_input_data

Inherited FORCE Model
---------------------
.. autoclass:: tension.models.FullFORCEModel
   :members:  
.. autoclass:: tension.models.OptimizedFORCEModel
   :members:
.. autoclass:: tension.constrained.BioFORCEModel
   :members:
.. autoclass:: tension.spiking.SpikingNNModel
   :members: