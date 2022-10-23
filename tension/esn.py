import tensorflow as tf
from .base import FORCEModel

class ESNFORCEModel(FORCEModel):
    def __init__(self, n_max, alpha_sq=1, num_skip=0, **kwargs):
        super().__init__(**kwargs)
        self.n_max = n_max
        self.num_step = tf.Variable(0., trainable=False)
        self.num_skip = num_skip
        assert alpha_sq >= 0, 'Error: alpha_sq must be nonnegative'
        self.alpha_sq = alpha_sq

    def initialize_P(self):
        if hasattr(self.original_force_layer, 'output_kernel'):
            if self.original_force_layer.output_kernel.trainable:
                self.P = tf.Variable(initial_value=tf.zeros(shape=(self.units, self.original_force_layer.output_size)),
                                     trainable=False)
                self.I = self.n_max * self.alpha_sq * tf.eye(num_rows=self.units)  
                self.R = tf.Variable(initial_value=tf.zeros(shape=(self.units, self.units)), trainable=False)
                
    @tf.function
    def train_step(self, data):
        x, y = data
        z, _, h, _ = self(x, training=True, reset_states=False)

        if self.force_layer.return_sequences:
            z = z[:,0,:]
        trainable_vars = self.trainable_variables

        self.num_step.assign_add(1.0, read_value=False)
        if tf.cond(self.update_kernel_condition(), lambda : True, lambda : False):
            # Update the output kernel
            if self._output_kernel_idx is not None:
                if tf.cond(self.num_step > self.num_skip, lambda : True, lambda : False):
                    self.P.assign_add(tf.matmul(tf.transpose(h), y[:,0,:]), read_value=False)
                    self.R.assign_add(tf.matmul(tf.transpose(h), h), read_value=False)
                
                if tf.cond(self.num_step == self.n_max, lambda : True, lambda : False):
                    self.update_output_kernel(trainable_vars[self._output_kernel_idx])
                    self.num_step.assign(0.0, read_value=False)
                    self.P.assign(tf.zeros(shape=self.P.shape), read_value=False)
                    self.R.assign(tf.zeros(shape=self.R.shape), read_value=False)
          
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y[:,0,:], z)

        return {m.name: m.result() for m in self.metrics}

    def update_output_kernel(self, trainable_vars_output_kernel):
        dwO = trainable_vars_output_kernel - tf.matmul(tf.linalg.inv(self.R + self.I), self.P)
        self.optimizer.apply_gradients(zip([dwO], [trainable_vars_output_kernel]))