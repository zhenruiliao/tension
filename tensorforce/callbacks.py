import tensorflow.keras as keras
import time 

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
class OutputTracking(keras.callbacks.Callback):
    def __init__(self, timesteps, output_size, output_state_idx = -1):
        self.tracking_arr = np.zeros((timesteps, output_size))
        self.output_state_idx = output_state_idx
    
    def on_batch_end(self, batch, logs=None):
        self.tracking_arr[batch] = self.model.force_layer.states[self.output_state_idx][0]
