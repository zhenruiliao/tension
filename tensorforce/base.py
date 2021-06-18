import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations

class FORCELayer(keras.layers.AbstractRNNCell):
    def __init__(self, units, output_size, activation, **kwargs):
        self.units = units 
        self._output_size = output_size
        self.activation = activations.get(activation)

        super().__init__(**kwargs)

    @property
    def state_size(self):
        return [self.units, self.units, self.output_size]

    @property 
    def output_size(self):
        return self._output_size

    def build(self, input_shape):
 
        self.input_kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform', trainable=False,
                                    name='input_kernel')
        self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform', 
          trainable=False,
          name='recurrent_kernel')
        self.feedback_kernel = self.add_weight(
          shape=(self.output_size, self.units), 
          initializer='uniform', 
          trainable=False,
          name='feedback_kernel')
        self.output_kernel = self.add_weight(
          shape=(self.units, self.output_size),
          initializer='uniform', trainable=True,
          name='output_kernel')      

        self.built = True

    @classmethod
    def from_weights(cls, weights, **kwargs):
        """Initialize the network from a list of weights (i.e., user-generated)"""

        input_kernel, recurrent_kernel, feedback_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        feedback_output_size, feedback_units = feedback_kernel.shape 
        output_units, output_size = output_kernel.shape 


        units = input_units 
        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, 
                            feedback_units, output_units]) == units)

        assert feedback_output_size == output_size 

        self = cls(units=units, output_size=output_size, **kwargs)
        self.input_kernel = self.add_weight(shape=(input_shape, self.units),
                                      initializer=keras.initializers.constant(input_kernel),
                                      trainable = False,
                                      name='input_kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=keras.initializers.constant(recurrent_kernel),
            trainable = False,
            name='recurrent_kernel')
        self.feedback_kernel = self.add_weight(
            shape=(self.output_size, self.units),
            initializer=keras.initializers.constant(feedback_kernel),
            trainable = False,
            name='feedback_kernel')
        self.output_kernel = self.add_weight(
            shape=(self.units, self.output_size),
            initializer=keras.initializers.constant(output_kernel),
            trainable = True,
            name='output_kernel')      

        self.built = True

        return self

class FORCEModel(keras.Model):
    def __init__(self, force_layer, alpha_P=1.,return_sequences=True):
        super().__init__()
        self.alpha_P = alpha_P

        self.force_layer = keras.layers.RNN(force_layer, 
                                            stateful=True, 
                                            return_state=True, 
                                            return_sequences=return_sequences)

        self.units = force_layer.units 
 
 

    def build(self, input_shape):

        self.P = self.add_weight(name='P', shape=(self.units, self.units), 
                                 initializer=keras.initializers.Identity(
                                              gain=self.alpha_P), trainable=True)
        super().build(input_shape)
 

    def call(self, x, training=False,   **kwargs):
        #x = self.input_layer(x)
        if training:
            return self.force_layer(x, **kwargs) 
        else:
            return self.force_layer(x, **kwargs)[0] 

    def train_step(self, data):
        x, y = data
        for i in range(x.shape[1]):
          z, _, h, _ = self(x[:,i:i+1,:], training=True)

          if self.force_layer.return_sequences:
            z = z[:,0,:]

          # Compute pseudogradients
          trainable_vars = self.trainable_variables
          dP = self.__pseudogradient_P(h, z, y[:,i,:])

          # Update weights
          self.optimizer.apply_gradients(zip([dP], [trainable_vars[1]]))
          
          dwO = self.__pseudogradient_wO(h, z, y[:,i,:])
          
          # Update weights
          self.optimizer.apply_gradients(zip([dwO], [trainable_vars[0]]))

        # Update metrics (includes the metric that tracks the loss)
          self.compiled_metrics.update_state(y[:,i,:], z)
        # Return a dict mapping metric names to current value

        return {m.name: m.result() for m in self.metrics}


    def __pseudogradient_P(self, h, z, y):
        """Implements the training step i.e. the rls() function
        This not a real gradient (does not use gradient.tape())
        Computes the actual update"""  

        k = backend.dot(self.P, tf.transpose(h))
        hPht = backend.dot(h, k)
        c = 1./(1.+hPht)
        assert c.shape == (1,1)
        hP = backend.dot(h, self.P)
        dP = backend.dot(c*k, hP)
        
        return dP 

    def __pseudogradient_wO(self, h, z, y):

        e = z-y
        Ph = backend.dot(self.P, tf.transpose(h))
        dwO = backend.dot(Ph, e)

        return dwO

    def compile(self, metrics, **kwargs):
        super().compile(optimizer=keras.optimizers.SGD(learning_rate=1),  metrics=metrics)

    def fit(self, x, y, epochs, verbose, **kwargs):
        if len(x.shape) < 3:
            x = tf.expand_dims(x, axis = 0)
        
        if len(y.shape) < 3:
            y = tf.expand_dims(y, axis = 0)
        
        if x.shape[0] != 1:
            raise ValueError("Dim 0 of x must be 1")

        if y.shape[0] != 1:
            raise ValueError("Dim 0 of y must be 1")
        if x.shape[1] != y.shape[1]:
            raise ValueError('Timestep dimension of inputs must match')     
        super().fit(x = x, y = y, epochs = epochs, batch_size = 1, verbose = verbose, **kwargs)

    def predict(self, x, **kwargs):
        if len(x.shape) == 3 and x.shape[0] != 1:
            raise ValueError()
        
        if len(x.shape) < 3:
            x = tf.expand_dims(x, axis = 0)
        
        return self(x, training = False)[0] 
