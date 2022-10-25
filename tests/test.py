import unittest
import sys
import json

with open(sys.argv[-1], 'rb') as f:
    INPUT_DICT = json.load(f)
sys.path.insert(0, '../')

from tension.base import FORCEModel
from tension.models import EchoStateNetwork, NoFeedbackESN
from tension.constrained import ConstrainedNoFeedbackESN, BioFORCEModel
from tension.spiking import LIF, Izhikevich, Theta, SpikingNNModel 
import tensorflow as tf
import numpy as np
tf.config.set_visible_devices([], 'GPU')

def fullforce_oscillation_test(dt, showplots=0):
    dt_per_s = round(1/dt)
    
    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])
    
    # A simpler example: just a sine waves
    '''
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*omega)
    '''
    
    # A slightly harder example: sum of sine waves
    
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*2*omega) * np.sin(t*omega/4)
    
    
    inp = np.zeros(targ.shape)
    inp[0:round(0.05*dt_per_s),0] = np.ones((round(0.05*dt_per_s)))
    hints = np.zeros(targ.shape)

    if showplots == 1:
        plt.figure()
        plt.plot(targ)
        plt.plot(hints)
        plt.plot(inp)
        plt.legend(['Target','Hints','Input'])
    
    return inp.astype(np.float32), targ.astype(np.float32), hints.astype(np.float32)

class TestESNwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.01
        tau = 10 * dt
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestESNwithFORCE']['input_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestESNwithFORCE']['recurrent_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestESNwithFORCE']['feedback_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestESNwithFORCE']['output_kernel'], dtype=tf.float32))
        self.fb_esn_layer = EchoStateNetwork.from_weights(weights=input_weights, 
                                                          recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                                          dtdivtau=dt / tau,
                                                          activation='tanh',
                                                          initial_a=tf.convert_to_tensor(INPUT_DICT['TestESNwithFORCE']['initial_a'], 
                                                     								     dtype=tf.float32))
        self.force_model = FORCEModel(force_layer=self.fb_esn_layer)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))

        self.assertEqual(np.allclose(INPUT_DICT['TestESNwithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestESNwithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestESNwithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestESNwithFORCE']['final_output_kernel'], self.fb_esn_layer.output_kernel.numpy().tolist()), True, 
                         'Output kernel incorrect')
        self.assertEqual(np.allclose(INPUT_DICT['TestESNwithFORCE']['final_states'], self.force_model.force_layer.states[0].numpy().tolist()), True,
                         'States incorrect')

class TestNoFBESNwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.01
        tau = 10 * dt
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestNoFBESNwithFORCE']['input_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestNoFBESNwithFORCE']['recurrent_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestNoFBESNwithFORCE']['output_kernel'], dtype=tf.float32))
        self.no_fb_esn_layer = NoFeedbackESN.from_weights(weights=input_weights, 
                                                          recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                                          dtdivtau=dt / tau,
                                                          activation='tanh',
                                                          initial_a=tf.convert_to_tensor(INPUT_DICT['TestNoFBESNwithFORCE']['initial_a'], dtype=tf.float32))
        self.force_model = FORCEModel(force_layer=self.no_fb_esn_layer)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['final_output_kernel'], self.no_fb_esn_layer.output_kernel.numpy().tolist()), True, 
                         'Output kernel incorrect')
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['final_recurrent_kernel'], self.no_fb_esn_layer.recurrent_kernel.numpy().tolist()), True, 
                         'Recurrent kernel incorrect')
        self.assertEqual(np.allclose(INPUT_DICT['TestNoFBESNwithFORCE']['final_states'], self.force_model.force_layer.states[0].numpy().tolist()), True,
                         'States incorrect')

class TestConstrainedNoFBESNwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.01
        tau = 1.5
        alpha = 1
        g = 1.25
        noise_param = (0, 0)
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['input_kernel'], dtype=tf.float32), 
                 		 tf.convert_to_tensor(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['recurrent_kernel'], dtype=tf.float32))
        structural_connectivity = tf.ones(shape=input_weights[1].shape)
        self.target = np.repeat(self.target, input_weights[1].shape[1], axis=1)
        self.no_fb_esn_layer = ConstrainedNoFeedbackESN.from_weights(weights=input_weights, 
                                                                     recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                                                     dtdivtau=dt / tau,
                                                                     activation='tanh',
                                                                     structural_connectivity=structural_connectivity,
                                                                     noise_param=noise_param,
                                                                     initial_a=tf.convert_to_tensor(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['initial_a'], 
                                                                     	                            dtype=tf.float32))
        self.force_model = BioFORCEModel(force_layer=self.no_fb_esn_layer, alpha_P=alpha)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        
        self.assertEqual(np.allclose(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['final_recurrent_kernel'], self.no_fb_esn_layer.recurrent_kernel.numpy().tolist()), True, 
                         'Recurrent kernel incorrect')
        self.assertEqual(np.allclose(INPUT_DICT['TestConstrainedNoFBESNwithFORCE']['final_states'], self.force_model.force_layer.states[0].numpy().tolist()), True,
                         'States incorrect')

class TestLIFwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.001
        tau_ref = 0.002   
        tau_mem = 0.01 
        v_reset = -65
        v_peak = -40
        tau_decay = 0.02
        tau_rise = 0.002
        tau_syn = tau_decay
        alpha = dt
        Q = 10
        G = 0.04
        I_bias = -40
        g = 1.0
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestLIFwithFORCE']['input_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestLIFwithFORCE']['recurrent_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestLIFwithFORCE']['feedback_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestLIFwithFORCE']['output_kernel'], dtype=tf.float32))
        self.LIF_layer = LIF.from_weights(weights=input_weights,
                                          recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                          dt=dt,
                                          tau_decay=tau_decay,
                                          tau_rise=tau_rise, 
                                          tau_syn=tau_syn, 
                                          tau_ref=tau_ref,
                                          tau_mem=tau_mem,
                                          v_peak=v_peak, 
                                          v_reset=v_reset, 
                                          I_bias=I_bias,
                                          G=G, 
                                          Q=Q, 
                                          g=g,
                                          initial_h=tf.zeros(shape=(1,input_weights[1].shape[1])),
                                          initial_voltage=tf.convert_to_tensor(INPUT_DICT['TestLIFwithFORCE']['initial_v'], dtype=tf.float32))
        self.force_model = SpikingNNModel(force_layer=self.LIF_layer, alpha_P=alpha)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=3,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))

        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['final_output_kernel'], self.LIF_layer.output_kernel.numpy().tolist()), True, 
                         'Output kernel incorrect')
        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['final_voltage'], self.force_model.force_layer.states[1].numpy().tolist()), True,
                         'Final voltage incorrect')    
        self.assertEqual(np.allclose(INPUT_DICT['TestLIFwithFORCE']['final_h'], self.force_model.force_layer.states[3].numpy().tolist()), True,
                         'Final firing rate incorrect')  

class TestThetawithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.0005
        v_reset = -np.pi
        v_peak = np.pi
        tau_decay = 0.02
        tau_rise = 0.002
        tau_syn = tau_decay
        alpha = dt
        Q = 10**4
        G = 15
        I_bias = 0
        g = 1.0
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestThetawithFORCE']['input_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestThetawithFORCE']['recurrent_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestThetawithFORCE']['feedback_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestThetawithFORCE']['output_kernel'], dtype=tf.float32))
        self.Theta_layer = Theta.from_weights(weights=input_weights,
                                              recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                              dt=dt,
                                              tau_decay=tau_decay,
                                              tau_rise=tau_rise, 
                                              tau_syn=tau_syn, 
                                              v_peak=v_peak, 
                                              v_reset=v_reset, 
                                              I_bias=I_bias,
                                              G=G, 
                                              Q=Q, 
                                              g=g,
                                              initial_h=tf.zeros(shape=(1, input_weights[1].shape[1])),
                                              initial_voltage=tf.convert_to_tensor(INPUT_DICT['TestThetawithFORCE']['initial_v'], 
                                                                                   dtype=tf.float32))
        self.force_model = SpikingNNModel(force_layer=self.Theta_layer, alpha_P=alpha)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=3,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['final_output_kernel'], self.Theta_layer.output_kernel.numpy().tolist()), True, 
                         "Output kernel incorrect")
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['final_voltage'], self.force_model.force_layer.states[1].numpy().tolist()), True,
                         "Final voltage incorrect")    
        self.assertEqual(np.allclose(INPUT_DICT['TestThetawithFORCE']['final_h'], self.force_model.force_layer.states[3].numpy().tolist()), True,
                         "Final firing rate incorrect")  

class TestIzhikevichwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.01
        adapt_time_inv = 0.01
        resonance_param = -2
        capacitance = 250
        adapt_jump_curr = 200 
        gain_on_v = 2.5 
        v_resting = -60
        v_thres = v_resting + 40 - (resonance_param / gain_on_v)
        v_reset = -65  
        v_peak = 30 
        tau_decay = 20
        tau_rise = 2
        tau_syn = 20
        alpha = 2
        Q = 5000;
        G = 5000
        I_bias = 1000
        g = 1.0
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        input_weights = (tf.convert_to_tensor(INPUT_DICT['TestIzhikevichwithFORCE']['input_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestIzhikevichwithFORCE']['recurrent_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestIzhikevichwithFORCE']['feedback_kernel'], dtype=tf.float32), 
                         tf.convert_to_tensor(INPUT_DICT['TestIzhikevichwithFORCE']['output_kernel'], dtype=tf.float32))

        self.Izhikevich_layer = Izhikevich.from_weights(weights=input_weights,
                                                        recurrent_nontrainable_boolean_mask=(tf.ones(shape=input_weights[1].shape) != 1),
                                                        dt=dt,
                                                        tau_decay=tau_decay,
                                                        tau_rise=tau_rise, 
                                                        tau_syn=tau_syn, 
                                           	            v_peak=v_peak, 
                                           	            v_reset=v_reset, 
                                                        I_bias=I_bias,
                                                        G=G, 
                                                        Q=Q, 
                                                        adapt_time_inv=adapt_time_inv,
                                                        resonance_param=resonance_param,
                                                        capacitance=capacitance,
                                                        adapt_jump_curr=adapt_jump_curr,
                                                        gain_on_v=gain_on_v,
                                                        v_resting=v_resting,
                                                        v_thres=v_thres,
                                                        g=g,
                                           	            initial_h=tf.zeros(shape=(1, input_weights[1].shape[1])),
                                                        initial_voltage=tf.convert_to_tensor(INPUT_DICT['TestIzhikevichwithFORCE']['initial_v'], 
                                                                                             dtype=tf.float32))
        self.force_model = SpikingNNModel(force_layer=self.Izhikevich_layer, alpha_P=alpha)

    def test_normal_case(self):
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['mae'], history.history['mae']), True, 
                         "Wrong Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['val_mae'], history.history['val_mae']), True, 
                         "Wrong Validation Loss")
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['predict'], self.force_model.predict(self.inputs, verbose=0).tolist()), True, 
                         "Wrong Predictions")
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['final_output_kernel'], self.Izhikevich_layer.output_kernel.numpy().tolist()), True, 
                         "Output kernel incorrect")
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['final_voltage'], self.force_model.force_layer.states[1].numpy().tolist()), True,
                         "Final voltage incorrect")    
        self.assertEqual(np.allclose(INPUT_DICT['TestIzhikevichwithFORCE']['final_h'], self.force_model.force_layer.states[3].numpy().tolist()), True,
                         "Final firing rate incorrect")  

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)