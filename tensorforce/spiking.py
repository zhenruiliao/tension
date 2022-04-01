import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend, activations
import numpy as np
from base import FORCELayer, FORCEModel

class SpikingNN(FORCELayer):
    """ Parent class part of the spiking neural networks implementation by Nicola and Clopath

        Args:
            dt : (float) Duration of one time step
            tau_decay : (float) Synaptic decay time 
            tau_rise : (float) Synaptic rise time
            tau_syn : (float) Synaptic time constant
            v_peak : (float) Voltage peak
            v_reset : (float) Reset potential
            I_bias : (float) Constant background current set near or at the rheobase value
            G : (float) Scales the static weight matrix
            Q : (float) Scales the feedback weight matrix
            initial_h : (2D array) An optional 1 x self.units tensor or numpy array specifying
                        the initial neuron firing rates 
            initial_voltage : (2D array) An optional 1 x self.units tensor or numpy array specifying
                        the initial voltage
            Additional parameters as outlined in FORCELayer

        States:
           t_step : (2D array) 1 x 1 tensor that tracks number of time steps  
           v : (2D array) 1 x self.units tensor containing voltage traces of each neuron
           u : (2D array) 1 x self.units tensor, auxillary storage variable (may be unused) 
           h : (2D array) 1 x self.units tensor containing neuron firing rates (r in paper)
           hr : (2D array) 1 x self.units tensor, storage variable for double exponential 
                              filter (h in paper)
           ipsc : (2D array) 1 x self.units tensor, Post synaptic current storage variable
           hr_ipsc : (2D array) 1 x self.units tensor, storage variable for ipsc
           out : (2D array) 1 x self.output_size tensor containing predicted output

    """

    def __init__(self, 
                 dt, 
                 tau_decay, 
                 tau_rise, 
                 tau_syn, 
                 v_peak, 
                 v_reset, 
                 I_bias,
                 G, 
                 Q,  
                 initial_h=None, 
                 initial_voltage=None,
                 **kwargs):
        self.dt = dt
        self.tau_decay = tau_decay
        self.tau_rise = tau_rise
        self.tau_syn = tau_syn
        self.v_peak = v_peak
        self.v_reset = v_reset
        self.I_bias = I_bias
        self.G = G 
        self.Q = Q 
        self._initial_h = initial_h
        self._initial_voltage = initial_voltage
        super().__init__(activation=None, recurrent_kernel_trainable=False, **kwargs)

    @property
    def state_size(self):
        return [1, self.units, self.units, self.units, self.units, self.units, self.units, self.output_size]

    def initialize_recurrent_kernel(self, recurrent_kernel=None):
        """
            Args:
                recurrent_kernel: (2D array) Tensor or numpy array containing the pre-initialized kernel. 
                                    If none, the kernel will be randomly initialized. 
        """
        if recurrent_kernel is None:        
            initializer = keras.initializers.RandomNormal(mean=0., 
                                                          stddev=self._g / (self.units**0.5 * self._p_recurr), 
                                                          seed=self.seed_gen.uniform([1], 
                                                                                      minval=None, 
                                                                                      dtype=tf.dtypes.int64)[0])
        
            recurrent_kernel = self._p_recurr * keras.layers.Dropout(1-self._p_recurr)(initializer(shape=(self.units, self.units)), 
                                                                                     training=True)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                initializer=keras.initializers.constant(self.G * recurrent_kernel),
                                                trainable=self._recurrent_kernel_trainable,
                                                name='recurrent_kernel')

    def initialize_feedback_kernel(self, feedback_kernel=None):
        """
            Args:
                feedback_kernel: (2D array) Tensor or numpy array containing the pre-initialized kernel. 
                                    If none, the kernel will be randomly initialized. 
        """
        if feedback_kernel is None:
            initializer = keras.initializers.RandomUniform(minval=-1., 
                                                           maxval=1., 
                                                           seed=self.seed_gen.uniform([1], 
                                                                                      minval=None, 
                                                                                      dtype=tf.dtypes.int64)[0])
            feedback_kernel = initializer(shape=(self.output_size, self.units))

        self.feedback_kernel = self.add_weight(shape=(self.output_size, self.units),
                                               initializer=keras.initializers.constant(self.Q * feedback_kernel),
                                               trainable=self._feedback_kernel_trainable,
                                               name='feedback_kernel')

    def initialize_output_kernel(self, output_kernel=None):
        """
            Args:
                output_kernel: (2D array) Tensor or numpy array containing the pre-initialized kernel.  
        """
        if output_kernel is None:
            output_kernel = tf.zeros((self.units, self.output_size))

        self.output_kernel = self.add_weight(shape=(self.units, self.output_size),
                                             initializer=keras.initializers.constant(output_kernel),
                                             trainable=self._output_kernel_trainable,
                                             name='output_kernel')    

    def initialize_voltage(self, batch_size):
        """ Initializes voltage trace for each neuron

            Args:
                batch_size : (int) The batch size

            Returns:
                A batch_size x self.units tensor of initial voltages
        """
        return tf.zeros((batch_size, self.units))

    def update_voltage(self, I, states):
        """ Updates the voltage of each neuron

            Args:
                I : (2D array) 1 x self.units tensor of neuron currents
                states : A tuple of tensors containing the layer states

            Returns: 
                Three 1 x self.units tensors containing updated v, updated u, and
                v_mask, a boolean mask where True indicates that the neuron 
                voltage exceeded self.v_peak 
        """
        return states[1], states[2], tf.zeros(states[1].shape)

    def update_firing_rate(self, v_mask, states):
        """ Updates the firing rate of each neuron

            Args:
                v_mask : (2D array) boolean mask where True indicates that the neuron 
                            voltage exceeded self.v_peak
                states : A tuple of tensors containing the layer states

            Returns:
                Four 1 x self.units tensors 
        """
        _, _, _, h, hr, ipsc, hr_ipsc, _ = states
        if self.tau_rise == 0:
          # single exponential synpatic filter
          h = h * tf.math.exp(-self.dt / self.tau_syn) + v_mask / self.tau_syn
        else:
          # double exponential synpatic filter
          h = h * tf.math.exp(-self.dt / self.tau_decay) + hr * self.dt
          hr = hr * tf.math.exp(-self.dt / self.tau_rise) + v_mask / (self.tau_rise * self.tau_decay)

        return h, hr, ipsc, hr_ipsc

    def compute_current(self, states):
        """ Computes current of each neuron

            Args:
                states : A tuple of tensors containing the layer states

            Returns:
                1 x self.units tensor of neuron currents 
        """
        _, _, _, h, _, _, _, out = states

        # Q included as part of feedback kernel; G as part of static recurrent kernel
        return self.I_bias + backend.dot(h, self.recurrent_kernel) + \
                  backend.dot(out, self.feedback_kernel) 
     
    def call(self, inputs, states):
        prev_t_step, prev_v, prev_u, prev_h, prev_hr, prev_ipsc, prev_hr_ipsc, prev_out = states

        I = self.compute_current(states)
        v, u, v_mask = self.update_voltage(I, states)
        h, hr, ipsc, hr_ipsc = self.update_firing_rate(v_mask, states)

        output = backend.dot(h, self.output_kernel)
        v = v + (self.v_reset - v) * v_mask
        t_step = prev_t_step + 1.0

        return output, [t_step, v, u, h, hr, ipsc, hr_ipsc, output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self._initial_h is not None:
          init_h = self._initial_h
        else:
          init_h = tf.zeros((batch_size, self.units))  

        if self._initial_voltage is not None:
          init_v = self._initial_voltage
        else:
          init_v = self.initialize_voltage(batch_size)

        init_out = backend.dot(init_h, self.output_kernel) 
        init_hr = tf.zeros((batch_size, self.units))
        init_u = tf.zeros((batch_size, self.units))
        init_ipsc = tf.zeros((batch_size, self.units))
        init_hr_ipsc = tf.zeros((batch_size, self.units)) 
        init_t_step = tf.zeros((batch_size, 1)) 

        return (init_t_step, init_v, init_u, init_h, init_hr, init_ipsc, init_hr_ipsc, init_out)

class OptimizedSpikingNN(SpikingNN):
    """ Optimizations added for improved computational speed
    """

    def update_firing_rate(self, v_mask, states):
        n_spike = tf.math.reduce_sum(v_mask)
        if n_spike > 0:
          jd = tf.math.reduce_sum(self.recurrent_kernel[v_mask[0] == 1], 
                                  axis=0,
                                  keepdims=True)
        else:
          jd = 0.0

        _, _, _, h, hr, ipsc, hr_ipsc, _ = states
        if self.tau_rise == 0:
          ipsc = ipsc * tf.math.exp(-self.dt / self.tau_syn) + jd / self.tau_syn 
          
          # single exponential synpatic filter 
          h = h * tf.math.exp(-self.dt / self.tau_syn) + v_mask / self.tau_syn
        else:
          ipsc = ipsc * tf.math.exp(-self.dt / self.tau_decay) + hr_ipsc * self.dt
          hr_ipsc = hr_ipsc * tf.math.exp(-self.dt / self.tau_rise) + jd / (self.tau_rise * self.tau_decay)
          
          # double exponential synpatic filter
          h = h * tf.math.exp(-self.dt / self.tau_decay) + hr * self.dt
          hr = hr * tf.math.exp(-self.dt / self.tau_rise) + v_mask / (self.tau_rise * self.tau_decay)

        return h, hr, ipsc, hr_ipsc

    def compute_current(self, states):
        _, _, _, _, _, ipsc, _, out = states
        return self.I_bias + ipsc + backend.dot(out, self.feedback_kernel) 

class LIF(OptimizedSpikingNN):
    """ Leaky integrate and fire neuron

        Args:
            tau_mem : (float) Membrane time constant
            tau_ref : (float) Refractory time constant

        See SpikingNN for states and additional args
    """

    def __init__(self,
                 tau_mem,
                 tau_ref,  
                 **kwargs):
      self.tau_mem = tau_mem
      self.tau_ref = tau_ref
      super().__init__(**kwargs)
      
    def initialize_voltage(self, batch_size):
        initializer = keras.initializers.RandomUniform(minval=0., 
                                                       maxval=1, 
                                                       seed=self.seed_gen.uniform([1], 
                                                                                  minval=None, 
                                                                                  dtype=tf.dtypes.int64)[0])
        return self.v_reset + initializer((batch_size, self.units)) * (30 - self.v_reset)

    def update_voltage(self, I, states):
        t_step, v, u, _, _, _, _, _ = states

        non_refract = tf.cast(self.dt * t_step > (u + self.tau_ref), tf.float32) 
        v = v + self.dt * non_refract * (-v + I) / self.tau_mem

        v_mask = tf.cast(v >= self.v_peak, tf.float32)
        u = u + (self.dt * t_step - u) * v_mask

        return v, u, v_mask

class Izhikevich(OptimizedSpikingNN):
    """ Implements Izhikevich neuron

        Args:
            adapt_time_inv : (float) Reciprocal of the adaptation time constant
            resonance_param : (float) Controls resonance properties of the model
            capacitance : (float) Membrane capacitance
            adapt_jump_curr : (float) Adaptation jump current 
            gain_on_v : (float) Controls action potential half-width
            v_resting : (float) Resting membrane potential 
            v_thres : (float) Threshold membrane potential

        See SpikingNN for states and additional args
    """

    def __init__(self,
                 adapt_time_inv, # a
                 resonance_param, # b
                 capacitance, #C
                 adapt_jump_curr, # d 
                 gain_on_v, # k or ff
                 v_resting,
                 v_thres, 
                 **kwargs):
        self.adapt_time_inv = adapt_time_inv
        self.resonance_param = resonance_param
        self.capacitance = capacitance
        self.adapt_jump_curr = adapt_jump_curr
        self.gain_on_v = gain_on_v
        self.v_resting = v_resting
        self.v_thres = v_thres
        super().__init__(**kwargs)

    def initialize_voltage(self, batch_size):
        initializer = keras.initializers.RandomUniform(minval=0., 
                                                       maxval=1, 
                                                       seed=self.seed_gen.uniform([1], 
                                                                                  minval=None, 
                                                                                  dtype=tf.dtypes.int64)[0])

        return self.v_resting + initializer((batch_size, self.units)) * (self.v_peak - self.v_resting)  
    
    def update_voltage(self, I, states):
        _, v, u, _, _, _, _, _ = states
        prev_v = v
        v = v + self.dt * (self.gain_on_v * (v - self.v_resting) * (v - self.v_thres) - u + I) / self.capacitance 
        v_mask = tf.cast(v >= self.v_peak, tf.float32)
        u = u + self.dt * self.adapt_time_inv * (self.resonance_param * (prev_v - self.v_resting) - u) + self.adapt_jump_curr * v_mask

        return v, u, v_mask
    
class Theta(OptimizedSpikingNN):
    """ Implements the Theta neuron. See SpikingNN for states and args.
    """

    def initialize_voltage(self, batch_size):
        initializer = keras.initializers.RandomUniform(minval=0., 
                                                       maxval=1, 
                                                       seed=self.seed_gen.uniform([1], 
                                                                                  minval=None, 
                                                                                  dtype=tf.dtypes.int64)[0])
        return self.v_reset + initializer((batch_size, self.units)) * (self.v_peak - self.v_reset)  
    
    def update_voltage(self, I, states):
        _, v, u, _, _, _, _, _ = states
        v = v + self.dt * (1 - tf.math.cos(v) + np.pi**2 * (1 + tf.math.cos(v)) * I)
        v_mask = tf.cast(v >= self.v_peak, tf.float32)

        return v, u, v_mask
    
class SpikingNNModel(FORCEModel):
    """ Implements FORCE training of spiking neural networks model. 
    """

    def force_layer_call(self, x, training, **kwargs):
        output, t_step, v, u, h, _ , _, _, out =  self.force_layer(x, **kwargs) 
        return output, t_step, h, out
