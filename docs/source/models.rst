FORCE Model
===========

Base FORCE Model
----------------
.. autoclass:: base.FORCEModel
   :members: build, initialize_P, initialize_train_idx, call, 
             force_layer_call, train_step, update_output_kernel,
             pseudogradient_P, pseudogradient_wO, update_recurrent_kernel,
             pseudogradient_P_Gx, pseudogradient_wR, compile, fit, predict,
             evaluate, coerce_input_data

Inherited FORCE Model
---------------------
.. autoclass:: models.FullFORCEModel
   :members:  
.. autoclass:: models.OptimizedFORCEModel
   :members:
.. autoclass:: constrained.BioFORCEModel
   :members:
.. autoclass:: spiking.SpikingNNModel
   :members: