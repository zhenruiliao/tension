import unittest
import sys
sys.path.insert(0, './tension/')

from base import FORCEModel
from models import EchoStateNetwork, NoFeedbackESN
from constrained import ConstrainedNoFeedbackESN, BioFORCEModel
from spiking import LIF, Izhikevich, Theta, SpikingNNModel 
import tensorflow as tf
import numpy as np

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
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
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
                                       verbose=0,
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
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
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
                                       verbose=0,
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



class TestConstrainedNoFBESNwithFORCE(unittest.TestCase):
    def setUp(self):
        dt = 0.5
        tau = 1.5
        u = 1
        n = 10
        m = n
        alpha = 1
        g = 1.25
        p_recurr = 1.0
        structural_connectivity = np.ones((n, n))
        noise_param = (0, 0.001)
        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        self.target = np.repeat(self.target, m, axis=1)
        self.no_fb_esn_layer = ConstrainedNoFeedbackESN(units=n,
                                                        activation='tanh',
                                                        dtdivtau=dt / tau,
                                                        p_recurr=p_recurr,
                                                        structural_connectivity=structural_connectivity,
                                                        noise_param=noise_param,
                                                        seed=123)
        self.force_model = BioFORCEModel(force_layer=self.no_fb_esn_layer, alpha_P=alpha)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=5,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        
        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[-0.0005527124158106744, 7.28508093743585e-05, 0.00024000277335289866, -0.00013454988948069513, 3.6648172681452706e-05,
                           0.00016049246187321842, 0.00012705568224191666, 0.00032861618092283607, 0.00024022418074309826, -0.000368243403499946],
                          [-0.0008066667942330241, 0.00020711713295895606, 9.308953303843737e-05, -0.00014400960935745388, 8.350379357580096e-05,
                           0.00041092027095146477, 0.0003334234934300184, 0.00029915786581113935, 0.0004505707765929401, -0.00020199903519824147],
                          [-0.0006052791140973568, 0.00046594146988354623, 0.000184719028766267, 0.0002707492094486952, 6.283417314989492e-05,
                           0.00046018764260225, 0.00036729665589518845, 0.00017799223132897168, 0.00010792256944114342, -0.00010933288285741583],
                          [-8.484553109155968e-05, 0.0005619135918095708, -0.00029152497882023454, 0.0006761588156223297, -1.1885021194757428e-05,
                           0.00012260206858627498, 0.0001668022887315601, -0.00014863588148728013, 0.00016588148719165474, -0.0003157246101181954],
                          [0.00015052843082230538, 0.0003896093985531479, -0.00011871605966007337, 0.0007462073117494583, 6.074510019971058e-05,
                           0.00034326859167777, -5.4357951739802957e-05, -0.0001309302169829607, 0.00018437986727803946, -0.0004438954056240618]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.00018002462456934154, 0.00032678653951734304, 0.0004570478922687471, 0.0005912345950491726, 0.0003107060620095581],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [0.0002809282741509378, 0.00039542611921206117, 0.0007779054576531053, 0.0005354530876502395, 0.00036618756712414324],  
                         "Wrong Validation Loss")
        self.assertEqual(self.no_fb_esn_layer.recurrent_kernel.numpy().tolist(),
                         [[-6.614994617848424e-06, 0.05701013281941414, 0.514080286026001, 0.23804883658885956, -0.36670738458633423,
                           -0.13886617124080658, -0.6514415740966797, -0.1736982762813568, -0.13147559762001038, 0.3715091943740845],
                          [-0.14497926831245422, -2.7913754365727073e-06, -0.5194059610366821, 0.3947071433067322, -0.44249534606933594,
                           0.00897783413529396, -0.045265477150678635, -0.5266339182853699, 0.14838415384292603, -0.07040367275476456],
                          [0.014748326502740383, 0.32417091727256775, -1.1428372999944258e-05, 0.3198469579219818, -0.1436920166015625,
                           -0.18662036955356598, 0.00437370827421546, -0.19135309755802155, 0.08595722168684006, -0.08110229671001434],
                          [-0.06241016089916229, -0.341831773519516, -0.0029590455815196037, -2.0187426343909465e-06, -0.24242791533470154,
                           0.1910620480775833, -0.022551946341991425, 0.24424347281455994, 0.026685485616326332, -0.10875021666288376],
                          [0.22786687314510345, 0.17259322106838226, -0.3005034029483795, -0.26205968856811523, -5.2806299208896235e-06,
                           -0.551007866859436, -0.142117440700531, -0.23140913248062134, 0.12730148434638977, -0.43605557084083557],
                          [-0.4693325161933899, 0.04941367730498314,-0.22183124721050262,-0.06780306994915009,-0.04090646281838417,
                           -4.039796749566449e-06, 0.33713263273239136, 0.22704589366912842, 0.021877411752939224, -0.13439878821372986],
                          [0.147931307554245, -0.16133248805999756, -0.1939931958913803, -0.1422845870256424, 0.4690357744693756,
                           -0.1354893296957016, -4.395051291794516e-06, -0.10657046735286713, 0.07844342291355133, -0.4431016147136688],
                          [-0.2569754719734192, 0.2932233214378357, 0.48899051547050476, 0.44572603702545166, -0.003629720536991954,
                           0.4968884587287903, -0.2946406602859497, -2.3317820705415215e-06, 0.07479719817638397, 0.11198653280735016],
                          [0.8090202808380127, 0.46941977739334106, -0.3659151494503021, 0.6804642677307129, 0.2386152148246765,
                           -0.2034139484167099, -0.1442575454711914, 0.07280490547418594, -1.63778258865932e-06, -0.3398582935333252],
                          [-0.16904018819332123, -0.17302954196929932, 0.43153995275497437, -0.27185583114624023, -0.28726235032081604,
                           -0.060157328844070435, 0.07261291891336441, -0.4348706603050232, -0.4745822250843048, -5.1697065828193445e-06]],
                         'Recurrent kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[0].numpy().tolist(),
                         [[-0.0002334479067940265, -0.0004670600173994899, 0.0007589510059915483, 0.000735300243832171, 0.0004591463366523385, 
                           0.0005326654063537717, 0.0002541852882131934, 0.0006685779662802815, -0.00040216499473899603, -0.00011362780787749216]],
                         'States incorrect')

class TestLIFwithFORCE(unittest.TestCase):
    def setUp(self):
        N = 25
        dt = 0.1
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
        p_recurr = 1.0
        g = 1.0

        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        m = self.target.shape[-1]
        self.LIF_layer = LIF(units=N, 
                             output_size=m,
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
                             p_recurr=p_recurr,
                             g=g,
                             seed=0)
        self.force_model = SpikingNNModel(force_layer=self.LIF_layer, 
                                          alpha_P=alpha)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=1,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        
        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[1.3739306926727295],
                          [-0.21828502416610718],
                          [1.373980164527893],
                          [-0.2682124972343445],
                          [1.3736437559127808],
                          [-0.46937867999076843],
                          [1.3722883462905884],
                          [-0.14236479997634888],
                          [1.248632788658142],
                          [0.455087810754776],
                          [1.252658486366272],
                          [-0.3942628800868988],
                          [1.2469356060028076],
                          [0.14996783435344696],
                          [1.2506024837493896],
                          [0.3511563837528229],
                          [1.2519581317901611],
                          [-0.5201265811920166],
                          [1.2460875511169434],
                          [-1.3081644773483276],
                          [1.2407777309417725]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.5582211017608643],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [1.0316537618637085],  
                         "Wrong Validation Loss")
        self.assertEqual(self.LIF_layer.output_kernel.numpy().tolist(),
                         [[-7.965285476529971e-05],
                          [1.9971186702605337e-05],
                          [-0.00010043672955362126],
                          [-0.00028970217681489885],
                          [-0.00010009270044974983],
                          [-0.0002673060225788504],
                          [-0.0001071240403689444],
                          [-0.00019773642998188734],
                          [-0.00028952318825758994],
                          [0.00045449574827216566],
                          [0.00038551414036192],
                          [-5.0343587645329535e-05],
                          [0.00011727314995368943],
                          [-0.00011708005331456661],
                          [0.000178324815351516],
                          [-0.000127108593005687],
                          [0.0002632028772495687],
                          [0.00014927471056580544],
                          [0.00045922183198854327],
                          [0.00029229684150777757],
                          [-0.00037012348184362054],
                          [0.00014146548346616328],
                          [-0.0006995118455961347],
                          [8.308118412969634e-05],
                          [5.780866922577843e-05]], 
                         'Output kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[1].numpy().tolist(),
                         [[-65.0,
                           -65.0,
                           -65.0,
                           -65.0,
                           -579.6243286132812,
                           -65.0,
                           -721.7266845703125,
                           -65.0,
                           -322.9876708984375,
                           -963.38818359375,
                           -65.0,
                           -358.60968017578125,
                           -65.0,
                           -1498.767578125,
                           -65.0,
                           -1762.235107421875,
                           -65.0,
                           -65.0,
                           -65.0,
                           -691.7926635742188,
                           -65.0,
                           -65.0,
                           -450.28173828125,
                           -65.0,
                           -65.0]],
                         'Voltages incorrect')

class TestThetawithFORCE(unittest.TestCase):
    def setUp(self):
        N = 20
        dt = 0.1  
        v_reset = -np.pi
        v_peak = np.pi
        tau_decay = 0.02
        tau_rise = 0.002
        tau_syn = tau_decay
        alpha = dt
        Q = 10**4
        G = 15
        I_bias = 0
        p_recurr = 1.0
        g = 1.0

        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        m = self.target.shape[-1]
        self.Theta_layer = Theta(units=N, 
                                 output_size=m,
                                 dt=dt,
                                 tau_decay=tau_decay,
                                 tau_rise=tau_rise, 
                                 tau_syn=tau_syn, 
                                 v_peak=v_peak, 
                                 v_reset=v_reset, 
                                 I_bias=I_bias,
                                 G=G, 
                                 Q=Q, 
                                 p_recurr=p_recurr,
                                 g=g,
                                 seed=0)
        self.force_model = SpikingNNModel(force_layer=self.Theta_layer, 
                                          alpha_P=alpha)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=1,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[-1.1573212146759033],
                          [-0.8044050335884094],
                          [0.5988779664039612],
                          [-1.0826992988586426],
                          [0.9155135750770569],
                          [-0.4077891707420349],
                          [0.1475445181131363],
                          [-0.3132239878177643],
                          [0.60218745470047],
                          [-0.8639060258865356],
                          [-0.2683490216732025],
                          [-0.4157659411430359],
                          [-0.9547082781791687],
                          [-0.006432774011045694],
                          [0.014615507796406746],
                          [9.847838373389095e-05],
                          [1.9777209758758545],
                          [0.013325778767466545],
                          [-0.1977643221616745],
                          [-0.0013325243489816785],
                          [0.7308911085128784]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.42683202028274536],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [0.7866876721382141],  
                         "Wrong Validation Loss")
        self.assertEqual(self.Theta_layer.output_kernel.numpy().tolist(),
                         [[8.750840788707137e-05],
                          [-0.00042196462163701653],
                          [0.00036325992550700903],
                          [0.0],
                          [3.588495383155532e-05],
                          [-0.00016558314382564276],
                          [0.0],
                          [4.580225140671246e-05],
                          [0.0004836033913306892],
                          [0.0],
                          [0.00010280316200805828],
                          [-5.3481348004424945e-05],
                          [-0.0001256872492376715],
                          [-0.00043469382217153907],
                          [0.00013455974112730473],
                          [4.246017851983197e-05],
                          [0.00015422985597979277],
                          [0.0],
                          [-6.304802082013339e-05],
                          [-0.00038662628503516316]], 
                         'Output kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[1].numpy().tolist(),
                         [[-2.941601514816284,
                           -3.1416015625,
                           -3.1416015625,
                           -195236.015625,
                           -3.1415863037109375,
                           -2.94159197807312,
                           -205871.609375,
                           -123814.2734375,
                           -2.941601514816284,
                           -108505.15625,
                           -169.52386474609375,
                           -3.1416015625,
                           -2.941601514816284,
                           -2.9415862560272217,
                           -14764.015625,
                           -103.05840301513672,
                           -2.9415862560272217,
                           -39687.6015625,
                           -139900.421875,
                           -3.1416015625]],
                         'Voltages incorrect')

class TestIzhikevichwithFORCE(unittest.TestCase):
    def setUp(self):
        N = 25
        dt = 0.04 
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
        p_recurr = 1.0
        g = 1.0

        self.inputs, self.target, _ = fullforce_oscillation_test(dt=dt)
        m = self.target.shape[-1]
        self.Izhikevich_layer = Izhikevich(units=N, 
                                           output_size=m,
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
                                           p_recurr=p_recurr,
                                           g=g,
                                           seed=0)
        self.force_model = SpikingNNModel(force_layer=self.Izhikevich_layer, 
                                          alpha_P=alpha)

    def test_normal_case(self):
        tf.random.set_seed(0)
        self.force_model.compile(metrics=["mae"])
        history = self.force_model.fit(x=self.inputs,
                                       y=self.target,
                                       epochs=1,
                                       verbose=0,
                                       validation_data=(self.inputs, self.target))
        self.assertEqual(self.force_model.predict(self.inputs).tolist(), 
                         [[8.477662777295336e-05],
                          [8.63697932800278e-05],
                          [8.792487642494962e-05],
                          [8.944264118326828e-05],
                          [9.092382970266044e-05],
                          [9.236922051059082e-05],
                          [9.377949027111754e-05],
                          [9.515535202808678e-05],
                          [9.649752610130236e-05],
                          [9.780666005099192e-05],
                          [9.908345236908644e-05],
                          [0.00010032854333985597],
                          [0.00010154257324757054],
                          [0.00010272615327266976],
                          [0.00010387993097538128],
                          [0.00010500448115635663],
                          [0.0001061004149960354],
                          [0.00010716830001911148],
                          [0.00010820871830219403],
                          [0.00010922220826614648],
                          [0.00011020930105587468],
                          [0.00011117057874798775],
                          [0.0001121065579354763],
                          [0.00011301770427962765],
                          [0.00011390456347726285],
                          [0.00011476761574158445],
                          [0.00011560737766558304],
                          [0.00011642424942692742],
                          [0.0001172187621705234],
                          [0.00011799134517787024],
                          [0.00011874244955834001],
                          [0.00011947251914534718],
                          [0.00012018195411656052],
                          [0.00012087121285730973],
                          [0.00012154068826930597],
                          [0.00012219080235809088],
                          [0.00012282191892154515],
                          [0.0001234344526892528],
                          [0.0001240287529071793],
                          [0.00012460524158086628],
                          [0.00012516423885244876],
                          [0.00012570612307172269],
                          [0.00012623124348465353],
                          [0.00012673994933720678],
                          [0.00012723256077151746],
                          [0.00012770942703355104],
                          [0.00012817086826544255],
                          [0.0001286171900574118],
                          [0.00012904869799967855],
                          [0.00012946574133820832],
                          [0.00012986855290364474]],  
                         "Wrong Predictions")
        self.assertEqual(history.history['mae'], 
                         [0.4079670011997223],
                         "Wrong Loss")
        self.assertEqual(history.history['val_mae'], 
                         [0.4068711996078491],  
                         "Wrong Validation Loss")
        self.assertEqual(self.Izhikevich_layer.output_kernel.numpy().tolist(),
                         [[-6.658068741671741e-05],
                          [0.0005934484070166945],
                          [0.0],
                          [0.0],
                          [0.0],
                          [-0.00023281561152543873],
                          [0.0],
                          [0.0007484601810574532],
                          [0.00044222830911166966],
                          [0.0],
                          [-0.0004371832183096558],
                          [0.0],
                          [0.0015024871099740267],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0009379772818647325],
                          [-6.658417987637222e-05],
                          [0.0],
                          [-0.000913383555598557],
                          [0.0004422357596922666],
                          [0.002005265560001135],
                          [-0.0010145952692255378],
                          [-0.00043716785148717463],
                          [0.0]], 
                         'Output kernel incorrect')
        self.assertEqual(self.force_model.force_layer.states[1].numpy().tolist(),
                         [[-61.130226135253906,
                           -60.94710159301758,
                           -18.51944351196289,
                           -41.31599426269531,
                           -45.55293273925781,
                           -57.944515228271484,
                           -43.416900634765625,
                           -60.36229705810547,
                           -58.28667068481445,
                           2.209271192550659,
                           -57.827213287353516,
                           -52.404510498046875,
                           -59.28446960449219,
                           -51.09229278564453,
                           -2.0616860389709473,
                           -47.18423843383789,
                           -61.81425857543945,
                           -61.01917266845703,
                           -53.912391662597656,
                           -60.21732711791992,
                           -58.022579193115234,
                           -59.14211654663086,
                           -62.0188102722168,
                           -57.61737823486328,
                           -13.064287185668945]],
                         'Voltages incorrect')

if __name__ == "__main__":
    unittest.main()