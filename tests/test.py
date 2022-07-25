import unittest
import sys
sys.path.insert(0, './tension/')

from base import FORCEModel
from models import EchoStateNetwork, NoFeedbackESN
import tensorflow as tf
import numpy as np

# class TestGetAreaRectangleWithSetUp(unittest.TestCase):
#   def setUp(self):
#     self.rectangle = Rectangle(0, 0)
 
#   def test_normal_case(self):
#     self.rectangle.set_width(2)
#     self.rectangle.set_height(3)
#     self.assertEqual(self.rectangle.get_area(), 6, "incorrect area")
 
#   def test_negative_case(self): 
#     """expect -1 as output to denote error when looking at negative area"""
#     self.rectangle.set_width(-1)
#     self.rectangle.set_height(2)
#     self.assertEqual(self.rectangle.get_area(), -1, "incorrect negative output")


def fullforce_oscillation_test(dt, showplots=0):
    dt_per_s = round(1/dt)
    
    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])
    
    # A simpler example: just a sine wave
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
        dt = 0.5
        tau = 10 * dt
        self.inputs, self.target, self.hints = fullforce_oscillation_test(dt=dt)
        n = 5
        m = self.target.shape[-1]
        self.fb_esn_layer = EchoStateNetwork(dtdivtau=dt / tau,
                                             units=n,
                                             output_size=m,
                                             activation='tanh',
                                             seed=0)
        self.force_model = FORCEModel(force_layer=self.fb_esn_layer)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       validation_data=(self.inputs, self.target))

        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[0.004208364523947239], [0.004116879776120186], [0.003965079318732023], [0.0037609923165291548], [0.0035125312861055136]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.019901564344763756, 0.010144452564418316, 0.0026462802197784185, 0.002146422164514661, 0.00406615948304534],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [0.010797129943966866, 0.0028149173595011234, 0.002189910039305687, 0.004130148328840733, 0.003912769258022308],  
                         "Wrong Validation Loss")
        self.assertEqual(self.fb_esn_layer.output_kernel.numpy().tolist(),
                         [[0.34147801995277405], [0.34196045994758606], [0.07137219607830048], [0.33427685499191284], [0.25517088174819946]], 
                         'Output kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[0].numpy().tolist(),
        	             [[0.0037309550680220127, 0.019548101350665092, 0.05191062018275261, -0.05617915838956833, 0.04444186016917229]],
                         'States incorrect')

class TestNoFBESNwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.5
        tau = 10 * dt
        self.inputs, self.target, self.hints = fullforce_oscillation_test(dt=dt)
        n = 5
        m = self.target.shape[-1]
        self.no_fb_esn_layer = NoFeedbackESN(dtdivtau=dt / tau,
                                             units=n,
                                             output_size=m,
                                             activation='tanh',
                                             seed=0)
        self.force_model = FORCEModel(force_layer=self.no_fb_esn_layer)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       validation_data=(self.inputs, self.target))
        
        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[-0.0026113244239240885], [-0.0023705451749265194], [-0.0021602699998766184], [-0.001974294427782297], [-0.0018077106215059757]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.06675976514816284, 0.03077871724963188, 0.015200100839138031, 0.007426183670759201, 0.0038008312694728374],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [0.03447664901614189, 0.016537034884095192, 0.00788111798465252, 0.0039651961997151375, 0.002184828743338585],  
                         "Wrong Validation Loss")
        self.assertEqual(self.no_fb_esn_layer.output_kernel.numpy().tolist(),
                         [[0.26615843176841736], [0.32662534713745117], [0.11450266093015671], [0.37827223539352417], [0.21127833425998688]], 
                         'Output kernel incorrect')
        self.assertEqual(self.no_fb_esn_layer.recurrent_kernel.numpy().tolist(),
        	             [[0.5921735167503357, -0.07547042518854141, 1.1865568161010742, -1.0704275369644165, 0.1744104027748108],
        	              [-0.1696804165840149, -0.9573836326599121, -0.3475620448589325, 0.4918712377548218, 0.18235792219638824], 
        	              [-0.652995765209198, 0.270507276058197, -0.3373152017593384, 0.09467004239559174, 0.37325412034988403],
        	              [0.0885663777589798, -0.2630828320980072, -0.056322451680898666, 0.2754984498023987, -0.6227282881736755], 
        	              [-0.6143308281898499, 0.28893551230430603, -0.3807820677757263, 0.040561165660619736, 0.3180539011955261]],
                         'Recurrent kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[0].numpy().tolist(),
        	             [[0.048447586596012115, -0.028402414172887802, 0.04323524981737137, 0.03157589212059975, -0.11115431785583496]],
                         'States incorrect')



if __name__ == "__main__":
    unittest.main()