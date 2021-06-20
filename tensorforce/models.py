import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations

from .base import FORCELayer

class EchoStateNetwork(FORCELayer):
    def __init__(self, dtdivtau, hscale = 0.25, **kwargs):
        self.dtdivtau = dtdivtau 
        self.hscale = hscale
        super().__init__(**kwargs)        

    def call(self, inputs, states):
        """Implements the forward step (i.e., the esn() function)
        """
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
        initializer = keras.initializers.RandomNormal(mean=0., 
                                                      stddev= self.hscale , 
                                                      seed = self.__seed_gen__.uniform([1], 
                                                                                       minval=None, 
                                                                                       dtype=tf.dtypes.int64)[0])
        init_a = initializer((batch_size, self.units))  
        init_h =  self.activation(init_a)
        init_out = backend.dot(init_h,self.output_kernel) 
 
        return (init_a, init_h, init_out)
