# -*- coding: utf-8 -*-
"""
Spyder Editor
Este é um arquivo de script temporário.
"""

import matplotlib.pyplot as plt
import numpy as np            

# Variables
adc_range = [0, 1023]
sensor_range = [-1.64, 1.64]
START = 000
STOP  = 600

#####################################################
#                   EXECUTION
#####################################################

# Load .txt file
mat = np.loadtxt('kidopro\kidopro_S2.txt')
E1_emg=mat[:,2:10]
E1_label = mat[:,0]
E1_rep = mat[:,1]

# calculate ADC to voltage conversion coefficients
a = sensor_range[1]-sensor_range[0]
b = adc_range[1]
c = sensor_range[0]

# Converts 0 to 1023 to values from -1.64 to +1.64
E1_emg = a * E1_emg / b + c

# Rectification
E1_emg=np.abs(E1_emg)


# Plotting 1
t = np.linspace(0, STOP, len(E1_emg[:,0]))
plt.figure(figsize=(10, 8))

plt.subplot(311)
plt.ylim(0, 1.8)
plt.xlim(START, STOP)
plt.plot(t, E1_emg[:,0], 'purple', label="Electrode 0")
plt.plot(t, E1_emg[:,1], 'orange', label="Electrode 1")
plt.plot(t, E1_emg[:,2], 'green', label="Electrode 2")
plt.plot(t, E1_emg[:,3], 'blue', label="Electrode 3")
plt.ylabel('EMG (mV)')
plt.legend()

plt.subplot(312)
plt.ylim(0, 1.8)
plt.xlim(START, STOP)
plt.plot(t, E1_emg[:,4], 'purple', label="Electrode 4")
plt.plot(t, E1_emg[:,5], 'orange', label="Electrode 5")
plt.plot(t, E1_emg[:,6], 'green', label="Electrode 6")
plt.plot(t, E1_emg[:,7], 'blue', label="Electrode 7")
plt.ylabel('EMG (mV)')
plt.legend()

plt.subplot(313)
plt.ylim(0, 11)
plt.xlim(START, STOP)
plt.plot(t, E1_label, 'blue')
plt.xlabel('Time (s)')
plt.ylabel('Classes')
plt.legend()
plt.show() 


# Plotting 2
t = np.linspace(0, STOP, len(E1_emg[:,0]))
plt.figure(figsize=(10, 8))

plt.subplot(311)
plt.ylim(0, 1.8)
plt.xlim(START, STOP)
plt.plot(t, E1_emg[:,0], 'purple', label="Electrode 0")
plt.plot(t, E1_emg[:,1], 'orange', label="Electrode 1")
plt.plot(t, E1_emg[:,2], 'green', label="Electrode 2")
plt.plot(t, E1_emg[:,3], 'blue', label="Electrode 3")
plt.ylabel('EMG (mV)')
plt.legend()

plt.subplot(312)
plt.ylim(0, 7)
plt.xlim(START, STOP)
plt.plot(t, E1_rep, 'blue')
plt.xlabel('Time (s)')
plt.ylabel('Repetitions')
plt.legend()

plt.subplot(313)
plt.ylim(0, 11)
plt.xlim(START, STOP)
plt.plot(t, E1_label, 'blue')
plt.xlabel('Time (s)')
plt.ylabel('Classes')
plt.legend()
plt.show() 
