import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding
import commpy.channelcoding.turbo as turbo
import commpy.channelcoding.interleavers as RandInterlv
from commpy.utilities import *

import math
import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt

import time
import pickle


# =============================================================================
# Generating pairs of (noisy codewords, message bit sequence)
# =============================================================================


def generate_examples(k_test=1000, step_of_history=200, SNR=0, code_rate = 2):

    trellis1 = cc.Trellis(np.array([2]), np.array([[7,5]]))
    trellis2 = cc.Trellis(np.array([2]), np.array([[7,5]]))
    #print('trellis: cc.Trellis(np.array([2]), np.array([[7,5]]))') # G(D) corresponding to the convolutional encoder

    tic = time.time()

    ### TEST EXAMPLES

    # Initialize Test Examples/
    noisy_codewords = np.zeros([1,int(k_test/step_of_history),step_of_history,2])
    true_messages = np.zeros([1,int(k_test/step_of_history),step_of_history,1])

    iterations_number = int(k_test/step_of_history)

    #for idx in range(SNR_points):
    nb_errors = np.zeros([iterations_number,1])


    tic = time.time()

    noise_sigmas = 10**(-SNR*1.0/20)

    mb_test_collect = np.zeros([iterations_number,step_of_history])

    interleaver = RandInterlv.RandInterlv(step_of_history,0)

    for iterations in range(iterations_number):

    #    print(iterations)
        message_bits = np.random.randint(0, 2, step_of_history)
        mb_test_collect[iterations,:] = message_bits
        [sys, par1, par2] = turbo.turbo_encode(message_bits, trellis1, trellis2, interleaver)

        noise = noise_sigmas*np.random.standard_normal(sys.shape) # Generate noise
        sys_r = (2*sys-1) + noise # Modulation plus noise
        noise = noise_sigmas*np.random.standard_normal(par1.shape) # Generate noise
        par1_r = (2*par1-1) + noise # Modulation plus noise
        noise = noise_sigmas*np.random.standard_normal(par2.shape) # Generate noise
        par2_r = (2*par2-1) + noise # Modulation plus noise

        sys_symbols = sys_r
        non_sys_symbols_1 = par1_r
        non_sys_symbols_2 = par2_r

        # ADD Training Examples
        noisy_codewords[0,iterations,:,:] = np.concatenate([sys_r.reshape(step_of_history,1),par1_r.reshape(step_of_history,1)],axis=1)

        # Message sequence
        true_messages[0,iterations,:,:] = message_bits.reshape(step_of_history,1)


    noisy_codewords = noisy_codewords.reshape(int(k_test/step_of_history),step_of_history,code_rate)
    true_messages = true_messages.reshape(int(k_test/step_of_history),step_of_history,1)
    target_true_messages  = mb_test_collect.reshape([mb_test_collect.shape[0],mb_test_collect.shape[1],1])

    toc = time.time()

    #print('time to generate test examples:', toc-tic)

    return (noisy_codewords, true_messages, target_true_messages)

