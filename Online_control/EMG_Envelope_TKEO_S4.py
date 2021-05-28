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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import butter, lfilter 
from scipy import signal
from scipy.stats import linregress

import biosignalsnotebooks as bsnb
import seaborn as sns
import statsmodels.api as sm
 

# Sample rate and desired cutoff frequencies (in Hz).
fs = 1000.0
lowcut = 20.0
highcut = 400.0

# Variáveis e outros parâmetros
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

###################################################### 
#                       FEATURES
######################################################

# Root Mean Square (RMS)
def featureRMS(data):
    return np.sqrt(np.mean(data**2, axis=0))

# Log Root Mean Square (RMS)
def featureLRMS(data):
    return np.log(np.sqrt(np.mean(data**2, axis=0)))

# Mean Absolute Value (MAV)
def featureMAV(data):
    return np.mean(np.abs(data), axis=0) 

# Waveform length (WL) 
def featureWL(data):
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

# Kurtosys(KUR)
def featureKUR(data): 
    return kurtosis(data)
    
# Skewness (SKEW)
def featureSKEW(data):
    return skew(data)

# Variance
def featureVAR(data):
    return np.var(data)

# AR coefficients
def featureAR(data):
    AR = [0.0, 0.0, 0.0, 0.0]
    e = 0
    method = "mle"
    AR, e = sm.regression.linear_model.yule_walker(data, order=4, method=method)
    return AR


###################################################### 
#           CONFUSION MATRIX - NORMALIZED
######################################################
def show_confusion_matrix(validations, predictions):   
    cm = confusion_matrix(validations, predictions)    
# Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cmn*100, 
                cmap='coolwarm',
                linecolor='white',
                linewidths=0.8,
                annot=True, 
                fmt='.1f', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.ylabel('Valor real')
    plt.xlabel('Valor predicto')
    plt.savefig('cmatrix_RF.png')


###################################################### 
#                ACCURACY (EACH CLASS)
######################################################
def show_acuracias_classes(validations, predictions): 
    cm = confusion_matrix(validations, predictions)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Imprime a acurácia de cada classe
    f = open('config/acuracias_por_classe.txt', 'a')
    f.write("Acurácia das classes\n")
    plt.show(block=False)
    index_acurácia = (cmn.diagonal())*100
    print('\nAcurácia das classes')
    for i in range(len(index_acurácia)): 
        print('Classe %d: %.2f' % (i, index_acurácia[i]), '%')
        f.write(str('%.2f' %(index_acurácia[i])))
        f.write("\n")
    f.write("\n")
    f.close()

###################################################### 
#                    ACCURACY
#####################################################
def show_acuracia_final(y_test, y_pred_test):
    score = accuracy_score(y_test, y_pred_test)
    print('\nAcurácia: %.2f' % (score*100),'%')
    f = open('config/acuracia_total.txt', 'a')
    #f.write("Acurácia final: ")
    f.write(str('%.2f' %(score*100)))
    f.write("\n")
    
    
###################################################### 
#            Sumário do balanço de classes
######################################################
def balanco_classes(data):
	df = DataFrame(data)
	counts = df.groupby(0).size()
	counts = counts.values
	for i in range(len(counts)):
		percent = counts[i] / len(df) * 100
		print('Classe=%d, total=%d, percentual=%.3f' % (i, counts[i], percent))
        

###################################################### 
#        Plotagem Real versus Predito
###################################################### 
def plotagem(emg, validations, predictions):
    t = np.linspace(0, 60, len(emg[:,5]))
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.xlim(0, 60)
    plt.ylim(0, 0.5)
    plt.plot(t, emg[:,5], 'purple', label="Eletrodo 0")
    plt.plot(t, emg[:,6], 'orange', label="Eletrodo 1")
    plt.ylabel('Amostras EMG (V)')
    plt.legend()
    
    plt.subplot(312)
    plt.xlim(0, 60)
    plt.ylim(0, 0.3)
    plt.plot(t, emg[:,7], 'green', label="Eletrodo 2")
    plt.plot(t, emg[:,8], 'blue', label="Eletrodo 3")
    plt.ylabel('Amostras EMG (V)')
    plt.legend()
    
    plt.subplot(313)
    plt.xlim(0, 60)
    plt.ylim(0, 5)
    t = np.linspace(0, 60, len(validations))
    plt.plot(t, validations, 'red', label="Valor real")
    plt.plot(t, predictions, 'blue', label="Valor predito")
    plt.ylabel('Classes')
    plt.xlabel('Tempo (s)')
    plt.legend()
    plt.show()     

#####################################################
#            Janelamento e Features
#####################################################
def janelamento(df, label, window, overlap):  
    labels = [] 
   
    segments_5 = []
    segments_6 = []
    segments_7 = []
    segments_8 = []
    feature_total = [] 

    for i in range(0, len(df) - window, overlap):
        
        x5 = df[5].values[i: i + window]   
        x6 = df[6].values[i: i + window]
        x7 = df[7].values[i: i + window]
        x8 = df[8].values[i: i + window] 

        teste = stats.mode(label[i: i + window])[0][0]   
        teste = teste.item()           
        labels.append(teste)  
 
        segments_5.append(x5)
        segments_6.append(x6)
        segments_7.append(x7)
        segments_8.append(x8)

    segments_5 = np.asarray(segments_5)
    segments_6 = np.asarray(segments_6)
    segments_7 = np.asarray(segments_7)
    segments_8 = np.asarray(segments_8)

      
    labels = np.asarray(labels)    

    for i in range (0, len(segments_5)):  
 
        r5 = featureRMS(segments_5[i])
        r6 = featureRMS(segments_6[i])
        r7 = featureRMS(segments_7[i])
        r8 = featureRMS(segments_8[i]) 

        w5 = featureWL(segments_5[i])
        w6 = featureWL(segments_6[i])
        w7 = featureWL(segments_7[i])
        w8 = featureWL(segments_8[i])

        v5 = featureMAV(segments_5[i])
        v6 = featureMAV(segments_6[i])
        v7 = featureMAV(segments_7[i])
        v8 = featureMAV(segments_8[i])

        b5 = featureKUR(segments_5[i])
        b6 = featureKUR(segments_6[i])
        b7 = featureKUR(segments_7[i])
        b8 = featureKUR(segments_8[i])

        c5 = featureSKEW(segments_5[i])
        c6 = featureSKEW(segments_6[i])
        c7 = featureSKEW(segments_7[i])
        c8 = featureSKEW(segments_8[i])

        d5 = featureVAR(segments_5[i])
        d6 = featureVAR(segments_6[i])
        d7 = featureVAR(segments_7[i])
        d8 = featureVAR(segments_8[i])
 
        e5 = featureLRMS(segments_5[i])
        e6 = featureLRMS(segments_6[i])
        e7 = featureLRMS(segments_7[i])
        e8 = featureLRMS(segments_8[i]) 

# Features apenas com sinais EMG
        
        feature_total.append([r5, r6, r7, r8,             # RMS
                              v5, v6, v7, v8,             # MAV
                              d5, d6, d7, d8,             # VAR
                              w5, w6, w7, w8])            # WL
        
    feature_total = np.asarray(feature_total)
        
    return (feature_total, labels)
  

#####################################################
#                   EXECUÇÃO
#####################################################

# carregando o arquivo txt
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

# Converte 0 a 1023 para valores de -1.64 a +1.64
emg_total = a * emg / b + c
E1_emg = emg_total

# E1_emg[:,4]     E1_emg[:,5]      E1_emg[:,6]     E1_emg[:,7]
# E1_emg[:,14]    E1_emg[:,15]     E1_emg[:,16]    E1_emg[:,17]


# Plota sinais EMG para inspeção visual do canal escolhido
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

# Canal EMG escolhido para operador TKEO
EMG_tkeo = E1_emg[:,15]

# Canal EMG escolhido para método da envoltória
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

smoothing_level_perc = 20           # Percentage
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

threshold_level = 10 # % Relative to the average value of the smoothed signal
threshold_level_norm_10 = normReg(threshold_level)

threshold_level = 80 # % Relative to the average value of the smoothed signal
threshold_level_norm_80 = normReg(threshold_level)

threshold_10 = avg_pre_pro_signal + threshold_level_norm_10 * std_pre_pro_signal
threshold_80 = avg_pre_pro_signal + threshold_level_norm_80 * std_pre_pro_signal

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
#                Ativação e Desativação através da envoltória
#####################################################################################

nyq = 0.5 * fs
low = 300 / nyq
high = 20 / nyq

# Filtro passa-alta
b, a = butter(2, high, btype='high')
EMG_envol  = signal.filtfilt(b, a, EMG_envol)

# Filtro passa-baixa
b1, a1 = butter(2, low, btype='low')
EMG_envol  = signal.filtfilt(b1, a1, EMG_envol)

# Retificação
EMG_envol=np.abs(EMG_envol)

# Filtro passa-baixa para envoltória
b, a = signal.butter(2, 7/(fs/2), btype = 'low')
rmss = signal.filtfilt(b, a, EMG_envol)       # envoltória em um canal EMG

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
#                Ativação e Desativação por inspeção visual (tempo)
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

# Plotagem dos sinais    

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

