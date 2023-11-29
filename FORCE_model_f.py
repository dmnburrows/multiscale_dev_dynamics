
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import tension
import tensorflow as tf #NB MUST BE TF < 3.11
from tension.constrained import ConstrainedNoFeedbackESN, BioFORCEModel
from sklearn.decomposition import PCA
import pkg_resources


#==============================
def FORCE_NFESN_learn(x_t, target_transposed, units, activation, dt, tau, p_recurr, structural_connectivity, noise_param, alpha, max_epoch):
#==============================
    """
    This function performs FORCE learning in a no feedback echo state network.
    
    Inputs:
        data (np array): cells x timepoints, state vectors
        
    Returns:
        all_c (np array): 1d vector of cluster labels for each time point
        sub_c (np array): 1d vector of all unique cluster labels, that label more than a single time point
    """

    esn_layer = ConstrainedNoFeedbackESN(units=units,
                                        activation=activation,
                                        dtdivtau=dt/tau,
                                        p_recurr=p_recurr,
                                        structural_connectivity=structural_connectivity,
                                        noise_param=noise_param)

    model = BioFORCEModel(force_layer=esn_layer, alpha_P=alpha)
    model.compile(metrics=["mae"])

    # pass the input as validation data for early stopping
    history = model.fit(x=x_t, 
                        y=target_transposed, 
                        epochs=max_epoch,
                        #callbacks=[earlystopping],
                        validation_data=(x_t, target_transposed))
    
    prediction = model.predict(x_t)
    
    model.save_weights('model_weights.h5')
    np.save('model_states.npy',model.force_layer.states[0])
    np.save('model_dynamics.npy', prediction)

    return(esn_layer.recurrent_kernel, esn_layer.input_kernel, model.force_layer.states, prediction)


#==============================
def NFESN_load_checkpoint(x_t, units, activation, dt, tau, p_recurr, structural_connectivity, noise_param, alpha):
#==============================
    """
    This function performs FORCE learning in a no feedback echo state network.
    
    Inputs:
        data (np array): cells x timepoints, state vectors
        
    Returns:
        all_c (np array): 1d vector of cluster labels for each time point
        sub_c (np array): 1d vector of all unique cluster labels, that label more than a single time point
    """

    #Can you just load this in? 

    esn_layer = ConstrainedNoFeedbackESN(units=units,
                                        activation=activation,
                                        dtdivtau=dt/tau,
                                        p_recurr=p_recurr,
                                        structural_connectivity=structural_connectivity,
                                        noise_param=noise_param, #####
                                        initial_a = tf.constant(np.load('model_states.npy')) ###### 
                                        )

    model = BioFORCEModel(force_layer=esn_layer, alpha_P=alpha)
    model.build((1,1, x_t.shape[-1])) #######
    model.load_weights('model_weights.h5')
    model.compile(metrics=["mae"])

    dyn = model.predict(x_t)
    return(dyn)


#==============================
def NFESN_perturb(x_t, units, activation, dt, tau, p_recurr, structural_connectivity, noise_param, alpha):
#==============================
    """
    This function performs FORCE learning in a no feedback echo state network.
    
    Inputs:
        data (np array): cells x timepoints, state vectors
        
    Returns:
        all_c (np array): 1d vector of cluster labels for each time point
        sub_c (np array): 1d vector of all unique cluster labels, that label more than a single time point
    """

    #Can you just load this in? 

    esn_layer = ConstrainedNoFeedbackESN(units=units,
                                        activation=activation,
                                        dtdivtau=dt/tau,
                                        p_recurr=p_recurr,
                                        structural_connectivity=structural_connectivity,
                                        noise_param=noise_param, #####
                                        initial_a = tf.constant(np.load('model_states.npy')) ###### 
                                        )

    model = BioFORCEModel(force_layer=esn_layer, alpha_P=alpha)
    model.build((1,1, x_t.shape[-1])) #######
    model.load_weights('model_weights.h5')
    model.compile(metrics=["mae"])

    dyn1 = model.predict(x_t)

    r_weights = esn_layer.recurrent_kernel
    inp_weights = esn_layer.input_kernel
    prac = np.asarray(r_weights)
    prac[0] = 1
    esn_layer.recurrent_kernel.assign(prac) 

    dyn2 = model.predict(x_t)
    return(dyn1, dyn2)
