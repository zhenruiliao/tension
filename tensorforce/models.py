import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations

from .base import FORCELayer

class EchoStateNetwork(FORCELayer):
    def __init__(self, dtdivtau, hscale = 0.25, initial_a = None, **kwargs):
        self.dtdivtau = dtdivtau 
        self.hscale = hscale
        self._initial_a = initial_a
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

        if self._initial_a is not None:
          init_a = self._initial_a
        else:
          initializer = keras.initializers.RandomNormal(mean=0., 
                                                        stddev= self.hscale , 
                                                        seed = self.seed_gen.uniform([1], 
                                                                                        minval=None, 
                                                                                        dtype=tf.dtypes.int64)[0])
          init_a = initializer((batch_size, self.units))  

        init_h =  self.activation(init_a)
        init_out = backend.dot(init_h,self.output_kernel) 

        return (init_a, init_h, init_out)

    
    
class NoFeedbackESN(EchoStateNetwork):

    def __init__(self, recurrent_kernel_trainable  = True, **kwargs):
        super().__init__(recurrent_kernel_trainable = recurrent_kernel_trainable, **kwargs)

    def call(self, inputs, states):
        """Implements the forward step (i.e., the esn() function)
        """
        prev_a, prev_h, prev_output = states      
        input_term = backend.dot(inputs, self.input_kernel)
        recurrent_term = backend.dot(prev_h, self.recurrent_kernel)

        dadt = -prev_a + input_term + recurrent_term 
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)
        output = backend.dot(h, self.output_kernel)

        return output, [a, h, output]


    def build(self, input_shape):

        self.initialize_input_kernel(input_shape[-1])
        self.initialize_recurrent_kernel()
        self.initialize_output_kernel()
          
        self.recurrent_nontrainable_boolean_mask = None
        self.built = True

    @classmethod
    def from_weights(cls, weights, recurrent_nontrainable_boolean_mask, **kwargs):
        # Initialize the network from a list of weights (e.g., user-generated)
        input_kernel, recurrent_kernel, output_kernel = weights 
        input_shape, input_units = input_kernel.shape 
        recurrent_units1, recurrent_units2 = recurrent_kernel.shape 
        output_units, output_size = output_kernel.shape 

        units = input_units      

        assert np.all(np.array([input_units, recurrent_units1, recurrent_units2, 
                            output_units]) == units)

        assert 'p_recurr' not in kwargs.keys(), 'p_recurr not supported in this method'
        assert recurrent_kernel.shape == recurrent_nontrainable_boolean_mask.shape, "Boolean mask and recurrent kernel shape mis-match"
        assert tf.math.count_nonzero(tf.boolean_mask(recurrent_kernel, recurrent_nontrainable_boolean_mask)).numpy() == 0, "Invalid boolean mask"  

        self = cls(units=units, output_size=output_size, p_recurr = None, **kwargs)

        self.recurrent_nontrainable_boolean_mask = tf.convert_to_tensor(recurrent_nontrainable_boolean_mask)
        
        self.initialize_input_kernel(input_shape, input_kernel)
        self.initialize_recurrent_kernel(recurrent_kernel)
        self.initialize_output_kernel(output_kernel)

        self.built = True

        return self
    
    
class TargetGeneratingNetwork(EchoStateNetwork):

    def __init__(self, hint_dim, **kwargs): 
        super().__init__(**kwargs)  
        self._hint_dim = hint_dim

# #####

#     def initialize_feedback_kernel(self, feedback_kernel = None):
#         if feedback_kernel is None:
#             initializer = keras.initializers.RandomUniform(minval=-1., 
#                                                           maxval= 1., 
#                                                           seed=self.seed_gen.uniform([1], 
#                                                                                  minval=None, 
#                                                                                  dtype=tf.dtypes.int64)[0])
#             feedback_kernel = initializer(shape = (self.output_size, self.units))

#         self.feedback_kernel = self.add_weight(shape=(self.output_size, self.units),
#                                                 initializer=keras.initializers.constant(feedback_kernel),
#                                                 trainable = self._feedback_kernel_trainable,
#                                                 name='feedback_kernel')
# #####

    def build(self, input_shape):
        input_shape = input_shape[:-1] + (input_shape[-1]-self.output_size-self._hint_dim,)
        super().build(input_shape)

        if self._hint_dim > 0:
           self.hint_kernel = self.add_weight(shape=(self._hint_dim, self.units),
                                              initializer=keras.initializers.RandomNormal(mean=0., 
                                                                                          stddev= 1/self._hint_dim**0.5, 
                                                                                          seed=self.seed_gen.uniform([1], 
                                                                                                                    minval=None, 
                                                                                                                    dtype=tf.dtypes.int64)[0]),
                                              # initializer=keras.initializers.RandomUniform(minval=-1, 
                                              #                                             maxval= 1, 
                                              #                                             seed=self.seed_gen.uniform([1], 
                                              #                                                                       minval=None, 
                                              #                                                                       dtype=tf.dtypes.int64)[0]),
                                              trainable=False,
                                              name='hint_kernel')

    @classmethod
    def from_weights(cls, weights, **kwargs):
        raise NotImplementedError

    
    def call(self, inputs, states):
        """Implements the forward step (i.e., the esn() function)
        """

        prev_a, prev_h, _ = states
     
        input_term = backend.dot(inputs[:, :-self.output_size-self._hint_dim], self.input_kernel)
        recurrent_term = backend.dot(prev_h, self.recurrent_kernel)

        if self._hint_dim > 0:
            feedback_hint_term = backend.dot(inputs[:, -self.output_size-self._hint_dim:-self.output_size], self.hint_kernel) + \
                                        backend.dot(inputs[:, -self.output_size:], self.feedback_kernel)
        else:
            feedback_hint_term = backend.dot(inputs[:, -self.output_size:], self.feedback_kernel)

        dadt = -prev_a + input_term + recurrent_term + feedback_hint_term  
        a = prev_a + self.dtdivtau * dadt
        h = self.activation(a)
        output = backend.dot(h, self.output_kernel)

        return output, [a, h, feedback_hint_term]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):

        if self._initial_a is not None:
          init_a = self._initial_a
        else:
          initializer = keras.initializers.RandomNormal(mean=0., 
                                                        stddev= self.hscale , 
                                                        seed = self.seed_gen.uniform([1], 
                                                                                        minval=None, 
                                                                                        dtype=tf.dtypes.int64)[0])
          init_a = initializer((batch_size, self.units))  

        init_h =  self.activation(init_a)

        return (init_a, init_h, init_h)
    
    
class fullFORCEModel(FORCEModel):
    def __init__(self, hint_dim, target_output_kernel_trainable = True, **kwargs):
        super().__init__(**kwargs)
        
        self._target_output_kernel_trainable = target_output_kernel_trainable
        self._output_dim = self.original_force_layer.output_size
        self._hint_dim = hint_dim
        assert self._hint_dim >= 0, 'Hint_dim cannot be negative'
        assert self._output_dim == 1, 'Output dimension must be 1'
        
    def build(self, input_shape):
        assert self._hint_dim < input_shape[-1] - self._output_dim, 'Hint dimension too large'

        self.initialize_target_network(input_shape)

        input_shape = input_shape[:-1] + (input_shape[-1]-self._output_dim-self._hint_dim,)
        super().build(input_shape)

    def initialize_target_network(self, input_shape):
        self._target_network = TargetGeneratingNetwork(dtdivtau = self.original_force_layer.dtdivtau, 
                                                          units = self.units, 
                                                          output_size = self._output_dim, 
                                                          activation = self.original_force_layer.activation,
                                                          output_kernel_trainable = self._target_output_kernel_trainable,
                                                          hint_dim = self._hint_dim)

        self.target_network = keras.layers.RNN(self._target_network, 
                                              stateful=True, 
                                              return_state=True, 
                                              return_sequences=True)
        
        self.target_network.build(input_shape)

    def initialize_P(self):
        self.task_P_output = self.add_weight(name='task_P_output', shape=(self.units, self.units), 
                                 initializer=keras.initializers.Identity(
                                 gain=self.alpha_P), trainable=True)

        if self._target_output_kernel_trainable:
            self.target_P_output = self.add_weight(name='target_P_output', shape=(self.units, self.units), 
                                    initializer=keras.initializers.Identity(
                                    gain=self.alpha_P), trainable=True)
        
    def initialize_train_idx(self):
        self._task_output_kernel_idx = None
        self._task_recurrent_kernel_idx = None
        self._target_output_kernel_idx = None
      
        for idx in range(len(self.trainable_variables)):
            trainable_name = self.trainable_variables[idx].name
            if 'target_generating_network' in trainable_name and 'output_kernel' in trainable_name:
              self._target_output_kernel_idx = idx
            elif 'output_kernel' in trainable_name:
              self._task_output_kernel_idx = idx
            elif 'target_P_output' in trainable_name:
              self._target_P_output_idx = idx
            elif 'task_P_output' in trainable_name:
              self._task_P_output_idx = idx
            elif 'recurrent_kernel' in trainable_name:
              self._task_recurrent_kernel_idx = idx

    def force_layer_call(self, x, training, **kwargs):
        if training:
            output_target, _, h_target, fb_hint_sum = self.target_network(x, **kwargs)
            output_task, _, h_task, _ = self.force_layer(x[:,:,:-self._output_dim-self._hint_dim], **kwargs)
            return output_task, h_task, output_target, h_target, fb_hint_sum
        else:
            return self.force_layer(x, **kwargs) 

    def train_step(self, data):

        x, y = data

        if self.run_eagerly:
            self.hidden_activation = []
                
        for i in range(x.shape[1]):
            output_task, h_task, output_target, h_target, fb_hint_sum = self(tf.concat([x[:,i:i+1,:], y[:,i:i+1,:]], axis = -1), 
                                                                training=True) # prev z, _, h, _

            if self.force_layer.return_sequences:
                output_task = output_task[:,0,:]

            if self.target_network.return_sequences:
                output_target = output_target[:, 0, :]

            trainable_vars = self.trainable_variables

            if self._task_output_kernel_idx is not None:
                self.update_output_kernel(self.task_P_output, h_task, output_task, y[:,i,:], 
                                          trainable_vars[self._task_P_output_idx],
                                          trainable_vars[self._task_output_kernel_idx])

                # if self._task_recurrent_kernel_idx is None:
                #   print('in task recurrent kernel')
                #   dP_task = self.pseudogradient_P(self.task_P_output, h_task)
                #   self.optimizer.apply_gradients(zip([dP_task], [trainable_vars[self._task_P_output_idx]]))

                # dwO_task = self.pseudogradient_wO(self.task_P_output, h_task, output_task, y[:,i,:])
                # self.optimizer.apply_gradients(zip([dwO_task], [trainable_vars[self._task_output_kernel_idx]]))

            if self._task_recurrent_kernel_idx is not None:
                self.update_recurrent_kernel(h_task, h_target, fb_hint_sum, trainable_vars)
                
                # dP_task = self.pseudogradient_P(self.task_P_output, h_task)
                # self.optimizer.apply_gradients(zip([dP_task], [trainable_vars[self._task_P_output_idx]]))

                # dwR_task = self.pseudogradient_wR_task(self.task_P_output, 
                #                                       trainable_vars[self._task_recurrent_kernel_idx], 
                #                                       h_task,
                #                                       self._target_network.recurrent_kernel,
                #                                       h_target,
                #                                       fb_hint_sum
                #                                       ) 
                # self.optimizer.apply_gradients(zip([dwR_task], [trainable_vars[self._task_recurrent_kernel_idx]]))

            if self._target_output_kernel_idx is not None:

                self.update_output_kernel(self.target_P_output, h_target, output_target, y[:,i,:], 
                                          trainable_vars[self._target_P_output_idx], 
                                          trainable_vars[self._target_output_kernel_idx])
                
                # dP_target = self.pseudogradient_P(self.target_P_output, h_target)
                # self.optimizer.apply_gradients(zip([dP_target], [trainable_vars[self._target_P_output_idx]]))

                # dwO_target = self.pseudogradient_wO(self.target_P_output, h_target, output_target, y[:,i,:])
                # self.optimizer.apply_gradients(zip([dwO_target], [trainable_vars[self._target_output_kernel_idx]]))

          # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y[:,i,:], output_task)
          # Return a dict mapping metric names to current value
            if self.run_eagerly:
                self.hidden_activation.append(h_task.numpy()[0])

        return {m.name: m.result() for m in self.metrics}

    def update_recurrent_kernel(self, h_task, h_target, fb_hint_sum, trainable_vars):

        if self._task_output_kernel_idx is None:
            dP_task = self.pseudogradient_P(self.task_P_output, h_task)
            self.optimizer.apply_gradients(zip([dP_task], [trainable_vars[self._task_P_output_idx]]))

        dwR_task = self.pseudogradient_wR_task(self.task_P_output, 
                                               trainable_vars[self._task_recurrent_kernel_idx], 
                                               h_task,
                                               self._target_network.recurrent_kernel,
                                               h_target,
                                               fb_hint_sum
                                               ) 
        self.optimizer.apply_gradients(zip([dwR_task], [trainable_vars[self._task_recurrent_kernel_idx]]))
    
    def pseudogradient_wR_task(self, P_task, wR_task, h_task, wR_target, h_target, fb_hint_sum): 
        e = backend.dot(h_task, wR_task) - backend.dot(h_target, wR_target) - fb_hint_sum
        dwR_task = backend.dot(backend.dot(P_task, tf.transpose(h_task)), e)

        return dwR_task #tf.transpose(dwR_task)
    
class optimizedFORCEModel(FORCEModel):

    def initialize_P(self):

        self.P_output = self.add_weight(name='P_output', shape=(self.units, self.units), 
                                        initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                        trainable=True)

        if self.original_force_layer.recurrent_kernel.trainable:

            bool_mask = self.original_force_layer.recurrent_nontrainable_boolean_mask

            if bool_mask is None or tf.math.count_nonzero(bool_mask) == 0:
          #    print('bool mask None or count is 0')
              self.P_GG = self.add_weight(name='P_GG', shape=(self.units, self.units), 
                                          initializer=keras.initializers.Identity(gain=self.alpha_P), 
                                          trainable=True)
              
              
            else:
              print('bool mask count is NOT zero')
              identity_3d = np.zeros((self.units, self.units, self.units))
              idx = np.arange(self.units)

  #################### 

              identity_3d[:, idx, idx] = self.alpha_P 

              I,J = np.nonzero(tf.transpose(bool_mask).numpy()==True)
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
              
    def pseudogradient_wR(self, P_Gx, h, z, y):
        e = z - y 
        assert e.shape == (1,1), 'Output must only have 1 dimension'

        if len(P_Gx.shape) == 2:
           # print('wR len')
            dwR_inter = backend.dot(P_Gx, tf.transpose(h))*e
            return dwR_inter*tf.ones((P_Gx.shape))
        else:
            Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
            dwR = Ph*e ### only valid for 1-d output
            return tf.transpose(dwR) 

    def pseudogradient_P_Gx(self, P_Gx, h):

        if len(P_Gx.shape) == 2:
            #print('PGx len')
            return self.pseudogradient_P(P_Gx,h)

        Ph = backend.dot(P_Gx, tf.transpose(h))[:,:,0]
        hPh = tf.expand_dims(backend.dot(Ph, tf.transpose(h)),axis = 2)
        dP_Gx = tf.expand_dims(Ph, axis = 2) * tf.expand_dims(Ph, axis = 1)/(1+hPh)
        return dP_Gx
    
