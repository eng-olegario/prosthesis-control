# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import matplotlib.pyplot as plt
import numpy as np     
from numpy import average, power, absolute, mean, std, diff, where       
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import butter, lfilter 
from scipy import signal
from scipy.stats import linregress
import biosignalsnotebooks as bsnb

# Sample rate and desired cutoff frequencies (in Hz).
fs = 1000.0
lowcut = 20.0
highcut = 400.0

# Variables and other parameters
classes = [0, 1]
num_classes = 2

adc_range = [0, 1023]
sensor_range = [-1.64, 1.64]

E1_label=[]

###################################################### 
#                 Bandpass Filter
######################################################

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#####################################################
#                  EXECUTION
#####################################################

# Loading the txt file
mat = np.loadtxt('S4_closed_hand.txt')            
#mat = np.loadtxt('S4_thumb_up.txt')            
#mat = np.loadtxt('S4_open_hand.txt')          
#mat = np.loadtxt('S4_fine_pinch_open.txt')     
#mat = np.loadtxt('S4_medium wrap.txt')        
#mat = np.loadtxt('S4_pronation.txt')           
#mat = np.loadtxt('S4_supination.txt')          
#mat = np.loadtxt('S4_spherical_grip.txt')      
#mat = np.loadtxt('S4_fine_pinch_closed.txt') 
#mat = np.loadtxt('S4_tool_grip.txt')           

emg=mat

# calculate ADC to voltage conversion coefficients
a = sensor_range[1]-sensor_range[0]
b = adc_range[1]
c = sensor_range[0]

# Converts 0 to 1023 to values from -1.64 to +1.64
emg_total = a * emg / b + c
E1_emg = emg_total

# E1_emg[:,4]     E1_emg[:,5]      E1_emg[:,6]     E1_emg[:,7]
# E1_emg[:,14]    E1_emg[:,15]     E1_emg[:,16]    E1_emg[:,17]


# Plot EMG signals for visual inspection of the chosen channel
START = 00
STOP  = 60
t = np.linspace(0, 60, len(emg[:,16]))
plt.subplot(311)
plt.xlim(START, STOP)
plt.ylim(0, 2)
plt.plot(t, E1_emg[:,16], 'green')
plt.plot(t, E1_emg[:,17], 'blue')
plt.legend()
plt.show() 

# EMG channel chosen for TKEO operator
EMG_tkeo = E1_emg[:,15]

# EMG channel chosen for visual method
EMG_envol = E1_emg[:,15]


#####################################################################################
#                                TKEO Operator
#####################################################################################

EMG_tkeo = EMG_tkeo - average(EMG_tkeo)

# Signal Filtering
low_cutoff = 10     # Hz
high_cutoff = 300   # Hz
sr = 1000

# Application of the signal to the filter
EMG_tkeo = bsnb.aux_functions._butter_bandpass_filter(EMG_tkeo, low_cutoff, high_cutoff, sr)

tkeo = []

for i in range(0, len(EMG_tkeo)):
    if i == 0 or i == len(EMG_tkeo) - 1:
        tkeo.append(EMG_tkeo[i])
    else:
        tkeo.append(power(EMG_tkeo[i], 2) - (EMG_tkeo[i+1] * EMG_tkeo[i-1]))

# Rectification         
rect_signal = absolute(tkeo)
rect_signal = bsnb.aux_functions._moving_average(rect_signal, sr / 10)        

smoothing_level_perc = 20           
smoothing_level = int((smoothing_level_perc / 100) * sr)
smooth_signal = []
for i in range(0, len(rect_signal)):
    if smoothing_level < i < len(rect_signal) - smoothing_level:
        smooth_signal.append(mean(rect_signal[i - smoothing_level:i + smoothing_level]))
    else:
        smooth_signal.append(0)

avg_pre_pro_signal = average(EMG_tkeo)
std_pre_pro_signal = std(EMG_tkeo)

# Regression function
def normReg(thresholdLevel):
    threshold_0_perc_level = (- avg_pre_pro_signal) / float(std_pre_pro_signal)
    threshold_100_perc_level = (max(smooth_signal) - avg_pre_pro_signal) / float(std_pre_pro_signal)
    m, b = linregress([0, 100], [threshold_0_perc_level, threshold_100_perc_level])[:2]
    return m * thresholdLevel + b

threshold_level = 10 
threshold_level_norm_10 = normReg(threshold_level)

threshold_10 = avg_pre_pro_signal + threshold_level_norm_10 * std_pre_pro_signal

bin_signal = []
for i in range(0, len(EMG_tkeo)):
    if smooth_signal[i] >= threshold_10:
        bin_signal.append(1)
    else:
        bin_signal.append(0)

diff_signal = diff(bin_signal)
act_begin = where(diff_signal == 1)[0]
act_end = where(diff_signal == -1)[0]

print("\nInicio da ativação (TKEO): ", act_begin)
print("Fim da ativação (TKEO): ", act_end)

#####################################################################################


#####################################################################################
#                           Begin and end of activation
#####################################################################################

nyq = 0.5 * fs
low = 300 / nyq
high = 20 / nyq

# High pass filter
b, a = butter(2, high, btype='high')
EMG_envol  = signal.filtfilt(b, a, EMG_envol)

# Low pass filter
b1, a1 = butter(2, low, btype='low')
EMG_envol  = signal.filtfilt(b1, a1, EMG_envol)

# Rectification
EMG_envol=np.abs(EMG_envol)

# Low pass filter 
b, a = signal.butter(2, 7/(fs/2), btype = 'low')
rmss = signal.filtfilt(b, a, EMG_envol)                           

# Generation of a square wave reflecting the activation and inactivation periods.
media = np.mean(rmss)
binary_signal = []
for i in range(0, len(EMG_envol)):
    if rmss[i] >= (media*0.7):
        binary_signal.append(1)
    else:
        binary_signal.append(0)

# Begin and end of activation periods
diff_signal = diff(binary_signal)
inicio_ativ = where(diff_signal == 1)[0]
fim_ativ = where(diff_signal == -1)[0]
print("\nInicio da ativação: ", inicio_ativ)
print("Fim da ativação: ", fim_ativ)


#####################################################################################
#                Begin and end of activation - visual inspection
#####################################################################################

for i in range(len(emg)):
     if  ((i<= 5000)  or (i > 10000 and i < 15000) or 
                         (i > 20000 and i < 25000) or
                         (i > 30000 and i < 35000) or
                         (i > 40000 and i < 45000) or
                         (i > 50000 and i < 55000) or 
                         (i > 60000)):
         
         E1_label.append(0)
     else:
         E1_label.append(1) 

#####################################################################################

# Plot of signals   

START = 00
STOP  = 60

plt.subplot(311)
plt.xlim(START, STOP)
plt.ylim(0, 2)
plt.plot(t, E1_emg[:,16], 'green', label="EMG 16")
plt.plot(t, E1_emg[:,17], 'blue', label="EMG 17")
plt.plot(t, E1_label, 'red', label="Inspeção Visual" )
plt.legend()

plt.subplot(312)
plt.xlim(START, STOP)
plt.ylim(0, 0.2)
plt.plot(t, rmss, 'purple', label="Envoltoria")
plt.legend()

plt.subplot(313)
plt.xlim(START, STOP)
plt.ylim(0, 2)
plt.plot(t, binary_signal, 'red', label="Envoltoria com threshold")
plt.plot(t, bin_signal, 'green', label="Operador TKEO")
plt.legend()
plt.show() 

