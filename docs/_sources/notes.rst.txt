
.. _notes:

Notes
=====

In FORCE, weight updates may be performed prior to the evaluation of 
all timesteps the forward pass through a RNN layer. In addition,
the weight update rules are such that the concept of batching is not feasible 
like that seen in traditional Keras RNNs trained via gradient descent. As a result,
batching is not supported, and each timestep in an example is treated as 
an individual batch to allow weight updates after each timestep in 
the forward pass (default behaviour). Inputs of shape 
``batch size x timestep x dimension`` are flattened prior to training / inference. 
 
During experimentation with echo state networks, it was noted that if the 
recurrent kernel is set to be trainable while the output kernel is not, then
depending on the kernel initialization, the mean of the recurrent kernel weights
gradually shifts away from 0, leading to saturation in the (tanh) activation function
that will stall learning.