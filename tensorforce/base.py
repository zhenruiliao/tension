import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations


class FORCELayer(keras.layers.AbstractRNNCell):
    def __init__(self, units, output_size, activation, seed = None, g = 1.5, 
                 input_kernel_trainable = False, recurrent_kernel_trainable = False, 
                 output_kernel_trainable = True, feedback_kernel_trainable = False, p_recurr = 1, **kwargs):
                
        self.units = units 
        self._output_size = output_size
        self.activation = activations.get(activation)

        if seed is None:
          self.seed_gen = tf.random.Generator.from_non_deterministic_state()
        else:
          self.seed_gen = tf.random.Generator.from_seed(seed)
        
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

    def initialize_input_kernel(self, input_shape, input_kernel = None):
        if input_kernel is None:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev= 1/input_shape**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                    minval=None, 
                                                                                    dtype=tf.dtypes.int64)[0])
            # initializer = keras.initializers.RandomUniform(minval=-1, 
            #                                                maxval=1,
            #                                                seed=self.seed_gen.uniform([1], 
            #                                                                          minval=None, 
            #                                                                          dtype=tf.dtypes.int64)[0])
            
            input_kernel = initializer(shape = (input_shape, self.units))
         
        self.input_kernel = self.add_weight(shape=(input_shape, self.units),
                                            initializer=keras.initializers.constant(input_kernel),
                                            trainable = self._input_kernel_trainable,
                                            name='input_kernel')
        
    def initialize_recurrent_kernel(self, recurrent_kernel = None):
        if recurrent_kernel is None:        
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev= self._g/self.units**0.5, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                      minval=None, 
                                                                                      dtype=tf.dtypes.int64)[0])
        
            recurrent_kernel = self._p_recurr*keras.layers.Dropout(1-self._p_recurr)(initializer(shape = (self.units, self.units)), 
                                                                                    training = True)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=keras.initializers.constant(recurrent_kernel),
                                                trainable = self._recurrent_kernel_trainable,
                                                name='recurrent_kernel')
    
    def initialize_feedback_kernel(self, feedback_kernel = None):
        if feedback_kernel is None:
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev= 1, 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                 minval=None, 
                                                                                 dtype=tf.dtypes.int64)[0])
            feedback_kernel = initializer(shape = (self.output_size, self.units))

        self.feedback_kernel = self.add_weight(shape=(self.output_size, self.units),
                                                initializer=keras.initializers.constant(feedback_kernel),
                                                trainable = self._feedback_kernel_trainable,
                                                name='feedback_kernel')
                                            

    def initialize_output_kernel(self, output_kernel = None):
        if output_kernel is None:
            initializer=keras.initializers.RandomNormal(mean=0., 
                                                        stddev= 1/self.units**0.5, 
                                                        seed=self.seed_gen.uniform([1], 
                                                                                   minval=None, 
                                                                                   dtype=tf.dtypes.int64)[0])
            output_kernel = initializer(shape = (self.units, self.output_size))

        self.output_kernel = self.add_weight(shape=(self.units, self.output_size),
                                              initializer=keras.initializers.constant(output_kernel),
                                              trainable = self._output_kernel_trainable,
                                              name='output_kernel')     
    
    def build(self, input_shape):

        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        self.initialize_feedback_kernel()
        self.initialize_output_kernel() 

        self.built = True

    @classmethod
    def from_weights(cls, weights, **kwargs):
        # Initialize the network from a list of weights (e.g., user-generated)
        input_kernel, recurrent_kernel, feedback_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        feedback_output_size, feedback_units = feedback_kernel.shape 
        output_units, output_size = output_kernel.shape 


        units = input_units 
        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, 
                            feedback_units, output_units]) == units)

        assert feedback_output_size == output_size 
        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'

        self = cls(units=units, output_size=output_size, p_recurr = None, **kwargs)

        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.initialize_feedback_kernel(feedback_kernel)
        self.initialize_output_kernel(output_kernel)

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

        self.original_force_layer = force_layer

    def build(self, input_shape):
        super().build(input_shape)
        self.initialize_P()
        self.initialize_train_idx()

    def initialize_P(self):

        self.P_output = self.add_weight(name='P_output', shape=(self.units, self.units), 
                                 initializer=keras.initializers.Identity(
                                              gain=self.alpha_P), trainable=True)

        if self.original_force_layer.recurrent_kernel.trainable:

            identity_3d = np.zeros((self.units, self.units, self.units))
            idx = np.arange(self.units)

#################### 

            identity_3d[:, idx, idx] = self.alpha_P 

            if self.original_force_layer.recurrent_nontrainable_boolean_mask is not None:
                I,J = np.nonzero(tf.transpose(self.original_force_layer.recurrent_nontrainable_boolean_mask).numpy()==True)
                identity_3d[I,:,J]=0
                identity_3d[I,J,:]=0

#################### 
# # new 
#          #  print('new')
#             identity_3d[idx, idx, :] = self.alpha_P 
#             J,I = np.nonzero(self.original_force_layer.recurrent_kernel.numpy()==0)
#             identity_3d[J,:,I]=0
#             identity_3d[:,J,I]=0

#################### 

            self.P_GG = self.add_weight(name='P_GG', shape=(self.units, self.units, self.units), 
                                    initializer=keras.initializers.constant(identity_3d), 
                                    trainable=True)

    def initialize_train_idx(self):
        self._output_kernel_idx = None
        self._recurrent_kernel_idx = None
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

    def call(self, x, training=False,   **kwargs):

        if training:
            return self.force_layer_call(x, training, **kwargs)
        else:
            initialization = all(v is None for v in self.force_layer.states)
            
            if not initialization:
              original_state = [i.numpy() for i in self.force_layer.states]
            output = self.force_layer_call(x, training, **kwargs)[0]

            if not initialization:
              self.force_layer.reset_states(states = original_state)
            return output

    def force_layer_call(self, x, training, **kwargs):
        return self.force_layer(x, **kwargs) 

    def train_step(self, data):

        x, y = data

        if self.run_eagerly:
          self.hidden_activation = []
                

        for i in range(x.shape[1]):
          z, _, h, _ = self(x[:,i:i+1,:], training=True)

          if self.force_layer.return_sequences:
            z = z[:,0,:]
         
          trainable_vars = self.trainable_variables

          if self._output_kernel_idx is not None:
            self.update_output_kernel(self.P_output, h, z, y[:,i,:], 
                                      trainable_vars[self._P_output_idx], 
                                      trainable_vars[self._output_kernel_idx])
          
          if self._recurrent_kernel_idx is not None:
            self.update_recurrent_kernel(self.P_GG, h, z, y[:,i,:],
                                         trainable_vars[self._P_GG_idx],
                                         trainable_vars[self._recurrent_kernel_idx])
          
        # Update metrics (includes the metric that tracks the loss)
          self.compiled_metrics.update_state(y[:,i,:], z)
        # Return a dict mapping metric names to current value

          if self.run_eagerly:
            self.hidden_activation.append(h.numpy()[0])

        return {m.name: m.result() for m in self.metrics}

    def update_output_kernel(self, P_output, h, z, y, trainable_vars_P_output, trainable_vars_output_kernel):

        # Compute pseudogradients
        dP = self.pseudogradient_P(P_output, h)
        # Update weights
        self.optimizer.apply_gradients(zip([dP], [trainable_vars_P_output]))

        dwO = self.pseudogradient_wO(P_output, h, z, y)
        self.optimizer.apply_gradients(zip([dwO], [trainable_vars_output_kernel]))

    def update_recurrent_kernel(self, P_Gx, h, z, y, trainable_vars_P_Gx, trainable_vars_recurrent_kernel):

        # Compute pseudogradients
        dP_Gx = self.pseudogradient_P_Gx(P_Gx, h)
        # Update weights
        self.optimizer.apply_gradients(zip([dP_Gx], [trainable_vars_P_Gx]))

        dwR = self.pseudogradient_wR(P_Gx, h, z, y)
        self.optimizer.apply_gradients(zip([dwR], [trainable_vars_recurrent_kernel]))


    def pseudogradient_P(self, P, h):
        # Implements the training step i.e. the rls() function
        # This not a real gradient (does not use gradient.tape())
        # Computes the actual update
        # Example array shapes
        # h : 1 x 500
        # P : 500 x 500 
        # k : 500 x 1 
        # hPht : 1 x 1
        # dP : 500 x 500 


        k = backend.dot(P, tf.transpose(h))
        hPht = backend.dot(h, k)
        c = 1./(1.+hPht)
      #  assert c.shape == (1,1)
        #hP = backend.dot(h, P)
        #dP = backend.dot(c*k, hP)
        dP = backend.dot(c*k, tf.transpose(k))
        return  dP 

    def pseudogradient_wO(self, P, h, z, y):
        # z : 1 x 20 
        # y : 1 x 20
        # e : 1 x 20
        # dwO : 500 x 20  

        e = z-y
        Ph = backend.dot(P, tf.transpose(h))
        dwO = backend.dot(Ph, e)

        return  dwO

#################### 

    def pseudogradient_wR(self, P_Gx, h, z, y):
        e = z - y 
        assert e.shape == (1,1), 'Output must only have 1 dimension'
        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]

        dwR = Ph*e ### only valid for 1-d output

        return tf.transpose(dwR) 

    def pseudogradient_P_Gx(self, P_Gx, h):
        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
        hPh = tf.expand_dims(backend.dot(Ph, tf.transpose(h)),axis = 2)
        #htP = backend.dot(h, P_Gx)[0]
        #dP_Gx = tf.expand_dims(Ph, axis = 2) * tf.expand_dims(htP, axis = 1)/(1+hPh)
        dP_Gx = tf.expand_dims(Ph, axis = 2) * tf.expand_dims(Ph, axis = 1)/(1+hPh)
        return dP_Gx

#################### 
#new 

    # def pseudogradient_wR(self, P_Gx, h, z, y):
    #     e = z - y 
    #     assert e.shape == (1,1)
    #     Pht = backend.dot(h, P_Gx)[0] 
    #     dwR = e*Pht ### only valid for 1-d output

    #     return dwR 


    # def pseudogradient_P_Gx(self, P_Gx, h):
    #    Pht = backend.dot(h, P_Gx)      # get 1 by j by i
    #    hPht = backend.dot(h, Pht)      # get 1 by 1 by i
    #    hP = tf.tensordot(h, P_Gx, axes = [[1],[0]]) # get 1 by k by i
    #    #dP_Gx = tf.reshape(Pht, (self.units, 1, self.units)) * hP / (1 + hPht)
    #    dP_Gx = tf.expand_dims(Pht[0], axis = 1) * hP / (1 + hPht)

    #    return dP_Gx

#################### 

    def compile(self, metrics, **kwargs):
        super().compile(optimizer=keras.optimizers.SGD(learning_rate=1), loss = 'mae', metrics=metrics,   **kwargs)


    def fit(self, x, y=None, epochs = 1, verbose = 'auto', **kwargs):

        if len(x.shape) < 2 or len(x.shape) > 3:
            raise ValueError('Shape of x is invalid')

        if len(y.shape) < 2 or len(y.shape) > 3:
            raise ValueError('Shape of y is invalid')
        
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis = 0)
        
        if len(y.shape) == 2:
            y = tf.expand_dims(y, axis = 0)
        
        if x.shape[0] != 1:
            raise ValueError("Dim 0 of x must be 1")

        if y.shape[0] != 1:
            raise ValueError("Dim 0 of y must be 1")
        
        if x.shape[1] != y.shape[1]: 
            raise ValueError('Timestep dimension of inputs must match')     

        return super().fit(x = x, y = y, epochs = epochs, batch_size = 1, verbose = verbose, **kwargs)

    def predict(self, x, **kwargs):
        if len(x.shape) == 3 and x.shape[0] != 1:
            raise ValueError('Dim 0 must be 1')
        
        if len(x.shape) < 2 or len(x.shape) > 3:
            raise ValueError('')

        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis = 0)

        return self(x, training = False)[0]
