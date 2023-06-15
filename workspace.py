#CONDA ENV == py_374 (BASE)

#Import packages
#---------------------------------------
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import tension
import tensorflow as tf #NB MUST BE TF < 3.11
from tension.constrained import ConstrainedNoFeedbackESN, BioFORCEModel
from sklearn.decomposition import PCA
import pkg_resources

import FORCE_model_f as fm

#Display versions
#--------------------------------------
pkg_resources.get_distribution("tensorflow").version
sys.version

# Define paths
#----------------------------------------------------------------------
s_data = '/cndd3/dburrows/DATA/bin/' #'/mnlsc/data/MCBL4/dburrows/'
s_code = '/cndd3/dburrows/CODE/bin/multiscale_dev_dynamics/' #'~/Documents/multiscale_dev_dynamics'


#LOAD IN TELENCEPHALON DATA
data = np.load(s_data + '/hadjiabadi-1.npz', allow_pickle=True)
keys = ['mask_name', 'cell_ids', 'tracez', 'all_coords']

assert len(data['cell_ids']) == len(data['mask_name'])
cell_ind = []
#Mop up all telencephalon neurons 
for i in range(len(data['mask_name'])):
    if 'Telencephalon' in data['mask_name'][i]:
        cell_ind = np.append(cell_ind, data['cell_ids'][i])
        
trace = data['tracez'][cell_ind.astype(int)]
trace = trace[:,50:250]
coords = data['all_coords'][cell_ind.astype(int)]
target = np.transpose(trace).astype(np.float32) # convert to shape (timestep, number of neurons)
print(str(trace.shape[0]) + ' neurons, ' + str(trace.shape[1]) + ' frames') 


#Define PARAMETERS
u = 1 # number of inputs, by default the forward pass does not use inputs so this is a stand-in
n = target.shape[1] # number of neurons
tau = 1.5 # neuron time constant
dt = 0.25 # time step
alpha = 1 # gain on P matrix at initialization
m = n # output dim equals the number of recurrent neurons
g = 1.25 # gain parameter controlling network chaos
p_recurr = 0.5 # (1 - p_recurr) of recurrent weights are randomly set to 0 and not trained
max_epoch = 500

structural_connectivity = np.ones((n, n)) # region connectivity matrix; set to all 1's since only looking at one subsection of the brain
noise_param = (0, 0.001) # mean and std of white noise injected to the forward pass

x_t = np.zeros((target.shape[0], u)).astype(np.float32) # stand-in input


class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


earlystopping = EarlyStoppingByLossVal(monitor="val_mae", value=0.12)
esn_layer = ConstrainedNoFeedbackESN(units=n,
                                    activation='tanh',
                                    dtdivtau=dt/tau,
                                    p_recurr=p_recurr,
                                    structural_connectivity=structural_connectivity,
                                    noise_param=noise_param)

model = BioFORCEModel(force_layer=esn_layer, alpha_P=alpha)
model.compile(metrics=["mae"])

# pass the input as validation data for early stopping
history = model.fit(x=x_t, 
                    y=target, 
                    epochs=max_epoch,
                    callbacks=[earlystopping],
                    validation_data=(x_t, target))

prediction = model.predict(x_t)

model.save_weights('model_weights.h5')
np.save('model_states.npy',model.force_layer.states[0])
np.save('model_dynamics.npy', prediction)