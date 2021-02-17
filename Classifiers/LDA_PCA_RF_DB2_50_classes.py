##############################################################################
# Dataset: DB2
# Classes: 50
# Classifier: RF
# Feature Projection: LDA and PCA
# Sensors: EMG
##############################################################################


import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew
import scipy.io as scio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import statistics
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns


# Variáveis e outros parâmetros
classes = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 
           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 
           28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
           42, 43, 44, 45, 46, 47, 48, 49]
num_classes = 50
flag = 0
flag2 = 0
RANDOM_SEED = 58

###################################################### 
#                       FEATURES
######################################################

# Mean Absolute Value (MAV)
def featureMAV(data):
    return np.mean(np.abs(data), axis=0) 

# Root Mean Square (RMS)
def featureRMS(data):
    return np.sqrt(np.mean(data**2, axis=0))

# Log Root Mean Square (RMS)
def featureLRMS(data):
    return np.log(np.sqrt(np.mean(data**2, axis=0)))

# Waveform length (WL) 
def featureWL(data):
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

# Kurtosis (KUR)
def featureKUR(data):
    #data = norm.rvs(size=1000, random_state=3)
    return kurtosis(data)

# Skewness (SKEW)
def featureSKEW(data):
    return skew(data)

# Variance (VAR)
def featureVAR(data):
    variance = np.var(data)
    return variance

# AR coefficients
def featureAR(data):
    AR = [0.0, 0.0, 0.0, 0.0]
    e = 0
    method = "mle"
    AR, e = sm.regression.linear_model.yule_walker(data, order=4, method=method)
    return AR

###################################################### 
#            MATRIZ DE CONFUSÃO NORMALIZADA
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
#             ACURÁCIA DE CADA CLASSE
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
#                     ACURÁCIA FINAL
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
    t = np.linspace(0, 500, len(emg[:,0]))
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.xlim(0, 500)
    plt.plot(t, emg[:,0], 'purple', label="Eletrodo 0")
    plt.plot(t, emg[:,1], 'orange', label="Eletrodo 1")
    plt.plot(t, emg[:,2], 'green', label="Eletrodo 2")
    plt.plot(t, emg[:,3], 'blue', label="Eletrodo 3")
    plt.ylabel('Amostras EMG (V)')
    plt.legend()

    plt.subplot(312)
    plt.xlim(0, 500)
    plt.plot(t, emg[:,4], 'blue', label="Eletrodo 4")
    plt.plot(t, emg[:,5], 'purple', label="Eletrodo 5")
    plt.plot(t, emg[:,6], 'orange', label="Eletrodo 6")
    plt.plot(t, emg[:,7], 'green', label="Eletrodo 7")

    plt.ylabel('Amostras EMG (V)')
    plt.legend()
    
    plt.subplot(313)
    plt.xlim(0, 500)
    plt.plot(t, emg[:,8], 'blue', label="Eletrodo 8")
    plt.plot(t, emg[:,9], 'green', label="Eletrodo 9")
    plt.plot(t, emg[:,10], 'purple', label="Eletrodo 10")
    plt.plot(t, emg[:,11], 'orange', label="Eletrodo 11")
    plt.ylabel('Amostras EMG (V)')
    plt.legend()
    plt.show() 
    
    plt.subplot(311)
    plt.xlim(0, 500)
    plt.ylim(0, 50)
    t = np.linspace(0, 500, len(validations))
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
    segments_0 = []
    segments_1 = []
    segments_2 = []
    segments_3 = []
    segments_4 = []
    segments_5 = []
    segments_6 = []
    segments_7 = []
    segments_8 = []
    segments_9 = []
    segments_10 = []
    segments_11 = []
    feature_total = [] 
    
    """
    segments_a0 = [] 
    segments_a1 = []
    segments_a2 = []
    segments_a3 = []
    segments_a4 = []
    segments_a5 = []
    segments_a6 = []
    segments_a7 = []
    segments_a8 = []
    segments_a9 = []
    segments_a10 = [] 
    segments_a11 = []
    segments_a12 = []
    segments_a13 = []
    segments_a14 = []
    segments_a15 = []
    segments_a16 = []
    segments_a17 = []
    segments_a18 = []
    segments_a19 = []
    segments_a20 = [] 
    segments_a21 = []
    segments_a22 = []
    segments_a23 = []
    segments_a24 = []
    segments_a25 = []
    segments_a26 = []
    segments_a27 = []
    segments_a28 = []
    segments_a29 = []
    segments_a30 = [] 
    segments_a31 = []
    segments_a32 = []
    segments_a33 = []
    segments_a34 = []
    segments_a35 = []
    """
    for i in range(0, len(df) - window, overlap):
        x0 = df[0].values[i: i + window]
        x1 = df[1].values[i: i + window]
        x2 = df[2].values[i: i + window]
        x3 = df[3].values[i: i + window]
        x4 = df[4].values[i: i + window]
        x5 = df[5].values[i: i + window]   
        x6 = df[6].values[i: i + window]
        x7 = df[7].values[i: i + window]
        x8 = df[8].values[i: i + window] 
        x9 = df[9].values[i: i + window]
        x10 = df[10].values[i: i + window] 
        x11 = df[11].values[i: i + window]
        
        """
        a0 = df2[0].values[i: i + window]
        a1 = df2[1].values[i: i + window]
        a2 = df2[2].values[i: i + window]
        a3 = df2[3].values[i: i + window]
        a4 = df2[4].values[i: i + window]
        a5 = df2[5].values[i: i + window]   
        a6 = df2[6].values[i: i + window]
        a7 = df2[7].values[i: i + window]
        a8 = df2[8].values[i: i + window] 
        a9 = df2[9].values[i: i + window]
    
        a10 = df2[10].values[i: i + window]
        a11 = df2[11].values[i: i + window]
        a12 = df2[12].values[i: i + window]
        a13 = df2[13].values[i: i + window]
        a14 = df2[14].values[i: i + window]
        a15 = df2[15].values[i: i + window]   
        a16 = df2[16].values[i: i + window]
        a17 = df2[17].values[i: i + window]
        a18 = df2[18].values[i: i + window] 
        a19 = df2[19].values[i: i + window]    
        a20 = df2[20].values[i: i + window]
        a21 = df2[21].values[i: i + window]
        a22 = df2[22].values[i: i + window]
        a23 = df2[23].values[i: i + window]
        a24 = df2[24].values[i: i + window]
        a25 = df2[25].values[i: i + window]   
        a26 = df2[26].values[i: i + window]
        a27 = df2[27].values[i: i + window]
        a28 = df2[28].values[i: i + window] 
        a29 = df2[29].values[i: i + window]    
        a30 = df2[30].values[i: i + window]
        a31 = df2[31].values[i: i + window]
        a32 = df2[32].values[i: i + window]
        a33 = df2[33].values[i: i + window]
        a34 = df2[34].values[i: i + window]
        a35 = df2[35].values[i: i + window]
        """
     
        teste = stats.mode(label[i: i + window])[0][0]   
        teste = teste.item()           
        labels.append(teste)  
        segments_0.append(x0)
        segments_1.append(x1)
        segments_2.append(x2)
        segments_3.append(x3)
        segments_4.append(x4)
        segments_5.append(x5)
        segments_6.append(x6)
        segments_7.append(x7)
        segments_8.append(x8)
        segments_9.append(x9)
        segments_10.append(x10)
        segments_11.append(x11)
        
        """
        segments_a0.append(a0)
        segments_a1.append(a1)
        segments_a2.append(a2)
        segments_a3.append(a3)
        segments_a4.append(a4)
        segments_a5.append(a5)
        segments_a6.append(a6)
        segments_a7.append(a7)
        segments_a8.append(a8)
        segments_a9.append(a9)
        segments_a10.append(a10)
        segments_a11.append(a11)
        segments_a12.append(a12)
        segments_a13.append(a13)
        segments_a14.append(a14)
        segments_a15.append(a15)
        segments_a16.append(a16)
        segments_a17.append(a17)
        segments_a18.append(a18)
        segments_a19.append(a19)
        segments_a20.append(a20)
        segments_a21.append(a21)
        segments_a22.append(a22)
        segments_a23.append(a23)
        segments_a24.append(a24)
        segments_a25.append(a25)
        segments_a26.append(a26)
        segments_a27.append(a27)
        segments_a28.append(a28)
        segments_a29.append(a29)
        segments_a30.append(a30)
        segments_a31.append(a31)
        segments_a32.append(a32)
        segments_a33.append(a33)
        segments_a34.append(a34)
        segments_a35.append(a35) 
        """
    
    segments_0 = np.asarray(segments_0)
    segments_1 = np.asarray(segments_1)
    segments_2 = np.asarray(segments_2)
    segments_3 = np.asarray(segments_3)
    segments_4 = np.asarray(segments_4)
    segments_5 = np.asarray(segments_5)
    segments_6 = np.asarray(segments_6)
    segments_7 = np.asarray(segments_7)
    segments_8 = np.asarray(segments_8)
    segments_9 = np.asarray(segments_9)
    segments_10 = np.asarray(segments_10)
    segments_11 = np.asarray(segments_11)
    
    """
    segments_a0 = np.asarray(segments_a0)
    segments_a1 = np.asarray(segments_a1)
    segments_a2 = np.asarray(segments_a2)
    segments_a3 = np.asarray(segments_a3)
    segments_a4 = np.asarray(segments_a4)
    segments_a5 = np.asarray(segments_a5)
    segments_a6 = np.asarray(segments_a6)
    segments_a7 = np.asarray(segments_a7)
    segments_a8 = np.asarray(segments_a8)
    segments_a9 = np.asarray(segments_a9)

    segments_a10 = np.asarray(segments_a10)
    segments_a11 = np.asarray(segments_a11)
    segments_a12 = np.asarray(segments_a12)
    segments_a13 = np.asarray(segments_a13)
    segments_a14 = np.asarray(segments_a14)
    segments_a15 = np.asarray(segments_a15)
    segments_a16 = np.asarray(segments_a16)
    segments_a17 = np.asarray(segments_a17)
    segments_a18 = np.asarray(segments_a18)
    segments_a19 = np.asarray(segments_a19)

    segments_a20 = np.asarray(segments_a20)
    segments_a21 = np.asarray(segments_a21)
    segments_a22 = np.asarray(segments_a22)
    segments_a23 = np.asarray(segments_a23)
    segments_a24 = np.asarray(segments_a24)
    segments_a25 = np.asarray(segments_a25)
    segments_a26 = np.asarray(segments_a26)
    segments_a27 = np.asarray(segments_a27)
    segments_a28 = np.asarray(segments_a28)
    segments_a29 = np.asarray(segments_a29)

    segments_a30 = np.asarray(segments_a30)
    segments_a31 = np.asarray(segments_a31)
    segments_a32 = np.asarray(segments_a32)
    segments_a33 = np.asarray(segments_a33)
    segments_a34 = np.asarray(segments_a34)
    segments_a35 = np.asarray(segments_a35)
"""    
    
    labels = np.asarray(labels)    

    for i in range (0, len(segments_0)):  
            
        r0 = featureRMS(segments_0[i])
        r1 = featureRMS(segments_1[i])
        r2 = featureRMS(segments_2[i])
        r3 = featureRMS(segments_3[i])
        r4 = featureRMS(segments_4[i])
        r5 = featureRMS(segments_5[i])
        r6 = featureRMS(segments_6[i])
        r7 = featureRMS(segments_7[i])
        r8 = featureRMS(segments_8[i]) 
        r9 = featureRMS(segments_9[i])  
        r10 = featureRMS(segments_10[i])
        r11 = featureRMS(segments_11[i])

        w0 = featureWL(segments_0[i])
        w1 = featureWL(segments_1[i])
        w2 = featureWL(segments_2[i])
        w3 = featureWL(segments_3[i])
        w4 = featureWL(segments_4[i])
        w5 = featureWL(segments_5[i])
        w6 = featureWL(segments_6[i])
        w7 = featureWL(segments_7[i])
        w8 = featureWL(segments_8[i])
        w9 = featureWL(segments_9[i])  
        w10 = featureWL(segments_10[i])
        w11 = featureWL(segments_11[i])
    
        v0 = featureMAV(segments_0[i])
        v1 = featureMAV(segments_1[i])
        v2 = featureMAV(segments_2[i])
        v3 = featureMAV(segments_3[i])
        v4 = featureMAV(segments_4[i])
        v5 = featureMAV(segments_5[i])
        v6 = featureMAV(segments_6[i])
        v7 = featureMAV(segments_7[i])
        v8 = featureMAV(segments_8[i])
        v9 = featureMAV(segments_9[i])  
        v10 = featureMAV(segments_10[i])
        v11 = featureMAV(segments_11[i])
        
        
        b0 = featureKUR(segments_0[i])
        b1 = featureKUR(segments_1[i])
        b2 = featureKUR(segments_2[i])
        b3 = featureKUR(segments_3[i])
        b4 = featureKUR(segments_4[i])
        b5 = featureKUR(segments_5[i])
        b6 = featureKUR(segments_6[i])
        b7 = featureKUR(segments_7[i])
        b8 = featureKUR(segments_8[i])
        b9 = featureKUR(segments_9[i])  
        b10 = featureKUR(segments_10[i])
        b11 = featureKUR(segments_11[i])
        
        c0 = featureSKEW(segments_0[i])
        c1 = featureSKEW(segments_1[i])
        c2 = featureSKEW(segments_2[i])
        c3 = featureSKEW(segments_3[i])
        c4 = featureSKEW(segments_4[i])
        c5 = featureSKEW(segments_5[i])
        c6 = featureSKEW(segments_6[i])
        c7 = featureSKEW(segments_7[i])
        c8 = featureSKEW(segments_8[i])
        c9 = featureSKEW(segments_9[i])  
        c10 = featureSKEW(segments_10[i])
        c11 = featureSKEW(segments_11[i])
        
        d0 = featureVAR(segments_0[i])
        d1 = featureVAR(segments_1[i])
        d2 = featureVAR(segments_2[i])
        d3 = featureVAR(segments_3[i])
        d4 = featureVAR(segments_4[i])
        d5 = featureVAR(segments_5[i])
        d6 = featureVAR(segments_6[i])
        d7 = featureVAR(segments_7[i])
        d8 = featureVAR(segments_8[i])
        d9 = featureVAR(segments_9[i])  
        d10 = featureVAR(segments_10[i])
        d11 = featureVAR(segments_11[i])
        
        e0 = featureLRMS(segments_0[i])
        e1 = featureLRMS(segments_1[i])
        e2 = featureLRMS(segments_2[i])
        e3 = featureLRMS(segments_3[i])
        e4 = featureLRMS(segments_4[i])
        e5 = featureLRMS(segments_5[i])
        e6 = featureLRMS(segments_6[i])
        e7 = featureLRMS(segments_7[i])
        e8 = featureLRMS(segments_8[i]) 
        e9 = featureLRMS(segments_9[i])  
        e10 = featureLRMS(segments_10[i])
        e11 = featureLRMS(segments_11[i])
        
        z0 = featureAR(segments_0[i])
        z1 = featureAR(segments_1[i])
        z2 = featureAR(segments_2[i])
        z3 = featureAR(segments_3[i])
        z4 = featureAR(segments_4[i])
        z5 = featureAR(segments_5[i])
        z6 = featureAR(segments_6[i])
        z7 = featureAR(segments_7[i])
        z8 = featureAR(segments_8[i])
        z9 = featureAR(segments_9[i])  
        z10 = featureAR(segments_10[i])
        z11 = featureAR(segments_11[i])
        
        
        """
        a0 = featureMAV(segments_a0[i])
        a1 = featureMAV(segments_a1[i])
        a2 = featureMAV(segments_a2[i])
        a3 = featureMAV(segments_a3[i])
        a4 = featureMAV(segments_a4[i])
        a5 = featureMAV(segments_a5[i])
        a6 = featureMAV(segments_a6[i])
        a7 = featureMAV(segments_a7[i])
        a8 = featureMAV(segments_a8[i])
        a9 = featureMAV(segments_a9[i])
    
        a10 = featureMAV(segments_a10[i])
        a11 = featureMAV(segments_a11[i])
        a12 = featureMAV(segments_a12[i])
        a13 = featureMAV(segments_a13[i])
        a14 = featureMAV(segments_a14[i])
        a15 = featureMAV(segments_a15[i])
        a16 = featureMAV(segments_a16[i])
        a17 = featureMAV(segments_a17[i])
        a18 = featureMAV(segments_a18[i])
        a19 = featureMAV(segments_a19[i])
    
        a20 = featureMAV(segments_a20[i])
        a21 = featureMAV(segments_a21[i])
        a22 = featureMAV(segments_a22[i])
        a23 = featureMAV(segments_a23[i])
        a24 = featureMAV(segments_a24[i])
        a25 = featureMAV(segments_a25[i])
        a26 = featureMAV(segments_a26[i])
        a27 = featureMAV(segments_a27[i])
        a28 = featureMAV(segments_a28[i])
        a29 = featureMAV(segments_a29[i])
    
        a30 = featureMAV(segments_a30[i])
        a31 = featureMAV(segments_a31[i])
        a32 = featureMAV(segments_a32[i])
        a33 = featureMAV(segments_a33[i])
        a34 = featureMAV(segments_a34[i])
        a35 = featureMAV(segments_a35[i]) 
        """
        
# Features com sinais EMG e ACC
        """
        feature_total.append([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11,                 # RMS
                              b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11,                 # KUR
                              v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,                 # MAV
                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,                 # SKEW
                              d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,                 # VAR
                              e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11,                 # logRMS
                              w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                              a0, a3, a6, a9,  a12, a15, a18, a21, a24, a27, a30, a33,
                              a1, a4, a7, a10, a13, a16, a19, a22, a25, a28, a31, a34,
                              a2, a5, a8, a11, a14, a17, a20, a23, a26, a29, a32, a35])
    
        """
     
# Features apenas com sinais EMG
    
        feature_total.append([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11,                 # RMS
                              b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11,                 # KUR
                              v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,                 # MAV
                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,                 # SKEW
                              d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11,                 # VAR
                              e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11,                 # logRMS
                              w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11])                # WL

    feature_total = np.asarray(feature_total)   
    return (feature_total, labels)
    

#####################################################
#                   EXECUÇÃO
#####################################################

# Exercício 1
mat1 = loadmat('ninapro_db2/S7_E1_A1.mat')
emg1 = mat1['emg']
label1 = mat1['restimulus']
repeticao1 = mat1['rerepetition']
#acc1 = mat1['acc']

# Exercício 2
mat2 = loadmat('ninapro_db2/S7_E2_A1.mat')
emg2 = mat2['emg']
label2 = mat2['restimulus']
repeticao2 = mat2['rerepetition']
#acc2 = mat2['acc']

# Exercício 3
mat3 = loadmat('ninapro_db2/S7_E3_A1.mat')
emg3 = mat3['emg']
label3 = mat3['restimulus']
repeticao3 = mat3['rerepetition']
#acc3 = mat3['acc']

emg = np.vstack((emg1, emg2, emg3))
label = np.vstack((label1, label2, label3))
repeticao = np.vstack((repeticao1, repeticao2, repeticao3))
#acc = np.vstack((acc1, acc2, acc3))

############################################
#   Pre-processing (Offset e retificação)
############################################
emg=emg-np.mean(emg)
emg=np.abs(emg)
emg=emg/np.max(emg)

# Vai da classe 0 a classe 49
E1_emg = emg
E1_label = label
#E1_acc = acc

index_rep_1 = []        # repetição 1
index_rep_2 = []        # repetição 2
index_rep_3 = []        # repetição 3
index_rep_4 = []        # repetição 4
index_rep_5 = []        # repetição 5
index_rep_6 = []        # repetição 6

# Marcações de inicio e termino das repetições 1, 3, 4 e 6
inicio_rep_1 = []
fim_rep_1 = []

inicio_rep_2 = []
fim_rep_2 = []

inicio_rep_3 = []
fim_rep_3 = []

inicio_rep_4 = []
fim_rep_4 = []

inicio_rep_5 = []
fim_rep_5 = []

inicio_rep_6 = []
fim_rep_6 = []


for j in range(1, 50): 
    for i in range (len(E1_label)):
        if E1_label[i] == j and repeticao[i]==1:        # fim da repetição 1
            index2=i
                              
        if E1_label[i] == j and repeticao[i]==2:        # inicio da repetição 3
            index3=i
        if E1_label[i] == j and repeticao[i]==3:        # fim da repetição 3
            index4=i
            
        if E1_label[i] == j and repeticao[i]==3:        # inicio da repetição 4
            index5=i
        if E1_label[i] == j and repeticao[i]==4:        # fim da repetição 4
            index6=i
            
        if E1_label[i] == j and repeticao[i]==5:        # inicio da repetição 6
            index7=i
        if E1_label[i] == j and repeticao[i]==6:        # fim da repetição 6
            index8=i    
            
        if E1_label[i] == j and repeticao[i]==1:        # inicio da repetição 2
            index9=i
        if E1_label[i] == j and repeticao[i]==2:        # fim da repetição 2
            index10=i 
            
        if E1_label[i] == j and repeticao[i]==4:        # inicio da repetição 5
            index11=i
        if E1_label[i] == j and repeticao[i]==5:        # fim da repetição 5
            index12=i
    
       
    fim_rep_1.append(index2)
    
    inicio_rep_3.append(index3)
    fim_rep_3.append(index4)
    
    inicio_rep_4.append(index5)
    fim_rep_4.append(index6)
    
    inicio_rep_6.append(index7)
    fim_rep_6.append(index8)
    
    inicio_rep_2.append(index9)
    fim_rep_2.append(index10)
    
    inicio_rep_5.append(index11)
    fim_rep_5.append(index12)


for i in range(len(E1_label)): 
    if  (i<= fim_rep_1[0] or
        (i> fim_rep_6[0]  and i<= fim_rep_1[1])  or 
        (i> fim_rep_6[1]  and i<= fim_rep_1[2])  or
        (i> fim_rep_6[2]  and i<= fim_rep_1[3])  or
        (i> fim_rep_6[3]  and i<= fim_rep_1[4])  or
        (i> fim_rep_6[4]  and i<= fim_rep_1[5])  or
        (i> fim_rep_6[5]  and i<= fim_rep_1[6])  or 
        (i> fim_rep_6[6]  and i<= fim_rep_1[7])  or
        (i> fim_rep_6[7]  and i<= fim_rep_1[8])  or
        (i> fim_rep_6[8]  and i<= fim_rep_1[9])  or
        (i> fim_rep_6[9]  and i<= fim_rep_1[10])  or 
        (i> fim_rep_6[10]  and i<= fim_rep_1[11])  or
        (i> fim_rep_6[11]  and i<= fim_rep_1[12])  or
        (i> fim_rep_6[12]  and i<= fim_rep_1[13])  or
        (i> fim_rep_6[13]  and i<= fim_rep_1[14])  or
        (i> fim_rep_6[14]  and i<= fim_rep_1[15])  or 
        (i> fim_rep_6[15]  and i<= fim_rep_1[16])  or
        (i> fim_rep_6[16]  and i<= fim_rep_1[17])  or
        (i> fim_rep_6[17]  and i<= fim_rep_1[18])  or
        (i> fim_rep_6[18]  and i<= fim_rep_1[19])  or 
        (i> fim_rep_6[19]  and i<= fim_rep_1[20])  or
        (i> fim_rep_6[20]  and i<= fim_rep_1[21])  or
        (i> fim_rep_6[21]  and i<= fim_rep_1[22])  or
        (i> fim_rep_6[22]  and i<= fim_rep_1[23])  or
        (i> fim_rep_6[23]  and i<= fim_rep_1[24])  or
        (i> fim_rep_6[24]  and i<= fim_rep_1[25])  or 
        (i> fim_rep_6[25]  and i<= fim_rep_1[26])  or
        (i> fim_rep_6[26]  and i<= fim_rep_1[27])  or
        (i> fim_rep_6[27]  and i<= fim_rep_1[28])  or
        (i> fim_rep_6[28]  and i<= fim_rep_1[29])  or 
        (i> fim_rep_6[29]  and i<= fim_rep_1[30])  or
        (i> fim_rep_6[30]  and i<= fim_rep_1[31])  or
        (i> fim_rep_6[31]  and i<= fim_rep_1[32])  or
        (i> fim_rep_6[32]  and i<= fim_rep_1[33])  or
        (i> fim_rep_6[33]  and i<= fim_rep_1[34])  or
        (i> fim_rep_6[34]  and i<= fim_rep_1[35])  or 
        (i> fim_rep_6[35]  and i<= fim_rep_1[36])  or
        (i> fim_rep_6[36]  and i<= fim_rep_1[37])  or
        (i> fim_rep_6[37]  and i<= fim_rep_1[38])  or
        (i> fim_rep_6[38]  and i<= fim_rep_1[39])  or
        
        (i> fim_rep_6[39]  and i<= fim_rep_1[40])  or
        (i> fim_rep_6[40]  and i<= fim_rep_1[41])  or
        (i> fim_rep_6[41]  and i<= fim_rep_1[42])  or
        (i> fim_rep_6[42]  and i<= fim_rep_1[43])  or
        (i> fim_rep_6[43]  and i<= fim_rep_1[44])  or
        (i> fim_rep_6[44]  and i<= fim_rep_1[45])  or 
        (i> fim_rep_6[45]  and i<= fim_rep_1[46])  or
        (i> fim_rep_6[46]  and i<= fim_rep_1[47])  or
        (i> fim_rep_6[47]  and i<= fim_rep_1[48])):
        index_rep_1.append(i)

    if  ((i> inicio_rep_2[0] and i<= fim_rep_2[0])  or 
        (i> inicio_rep_2[1]  and i<= fim_rep_2[1])  or 
        (i> inicio_rep_2[2]  and i<= fim_rep_2[2])  or
        (i> inicio_rep_2[3]  and i<= fim_rep_2[3])  or
        (i> inicio_rep_2[4]  and i<= fim_rep_2[4])  or
        (i> inicio_rep_2[5]  and i<= fim_rep_2[5])  or
        (i> inicio_rep_2[6]  and i<= fim_rep_2[6])  or 
        (i> inicio_rep_2[7]  and i<= fim_rep_2[7])  or
        (i> inicio_rep_2[8]  and i<= fim_rep_2[8])  or
        (i> inicio_rep_2[9]  and i<= fim_rep_2[9])  or
        (i> inicio_rep_2[10]  and i<= fim_rep_2[10])  or
        (i> inicio_rep_2[11]  and i<= fim_rep_2[11])  or
        (i> inicio_rep_2[12]  and i<= fim_rep_2[12])  or
        (i> inicio_rep_2[13]  and i<= fim_rep_2[13])  or
        (i> inicio_rep_2[14]  and i<= fim_rep_2[14])  or
        (i> inicio_rep_2[15]  and i<= fim_rep_2[15])  or
        (i> inicio_rep_2[16]  and i<= fim_rep_2[16])  or
        (i> inicio_rep_2[17]  and i<= fim_rep_2[17])  or
        (i> inicio_rep_2[18]  and i<= fim_rep_2[18])  or
        (i> inicio_rep_2[19]  and i<= fim_rep_2[19])  or
        (i> inicio_rep_2[20]  and i<= fim_rep_2[20])  or
        (i> inicio_rep_2[21]  and i<= fim_rep_2[21])  or
        
        (i> inicio_rep_2[22]  and i<= fim_rep_2[22])  or
        (i> inicio_rep_2[23]  and i<= fim_rep_2[23])  or
        (i> inicio_rep_2[24]  and i<= fim_rep_2[24])  or
        (i> inicio_rep_2[25]  and i<= fim_rep_2[25])  or
        (i> inicio_rep_2[26]  and i<= fim_rep_2[26])  or
        (i> inicio_rep_2[27]  and i<= fim_rep_2[27])  or
        (i> inicio_rep_2[28]  and i<= fim_rep_2[28])  or
        (i> inicio_rep_2[29]  and i<= fim_rep_2[29])  or
        (i> inicio_rep_2[30]  and i<= fim_rep_2[30])  or
        (i> inicio_rep_2[31]  and i<= fim_rep_2[31])  or
        (i> inicio_rep_2[32]  and i<= fim_rep_2[32])  or
        (i> inicio_rep_2[33]  and i<= fim_rep_2[33])  or
        (i> inicio_rep_2[34]  and i<= fim_rep_2[34])  or
        (i> inicio_rep_2[35]  and i<= fim_rep_2[35])  or
        (i> inicio_rep_2[36]  and i<= fim_rep_2[36])  or
        (i> inicio_rep_2[37]  and i<= fim_rep_2[37])  or
        (i> inicio_rep_2[38]  and i<= fim_rep_2[38])  or
        (i> inicio_rep_2[39]  and i<= fim_rep_2[39])  or
        
        (i> inicio_rep_2[40]  and i<= fim_rep_2[40])  or
        (i> inicio_rep_2[41]  and i<= fim_rep_2[41])  or
        (i> inicio_rep_2[42]  and i<= fim_rep_2[42])  or
        (i> inicio_rep_2[43]  and i<= fim_rep_2[43])  or
        (i> inicio_rep_2[44]  and i<= fim_rep_2[44])  or
        (i> inicio_rep_2[45]  and i<= fim_rep_2[45])  or
        (i> inicio_rep_2[46]  and i<= fim_rep_2[46])  or
        (i> inicio_rep_2[47]  and i<= fim_rep_2[47])  or
        (i> inicio_rep_2[48]  and i<= fim_rep_2[48])):
        index_rep_2.append(i)
    
    if  ((i> inicio_rep_3[0] and i<= fim_rep_3[0])  or 
        (i> inicio_rep_3[1]  and i<= fim_rep_3[1])  or 
        (i> inicio_rep_3[2]  and i<= fim_rep_3[2])  or
        (i> inicio_rep_3[3]  and i<= fim_rep_3[3])  or
        (i> inicio_rep_3[4]  and i<= fim_rep_3[4])  or
        (i> inicio_rep_3[5]  and i<= fim_rep_3[5])  or
        (i> inicio_rep_3[6]  and i<= fim_rep_3[6])  or 
        (i> inicio_rep_3[7]  and i<= fim_rep_3[7])  or
        (i> inicio_rep_3[8]  and i<= fim_rep_3[8])  or
        (i> inicio_rep_3[9]  and i<= fim_rep_3[9])  or
        (i> inicio_rep_3[10]  and i<= fim_rep_3[10])  or
        (i> inicio_rep_3[11]  and i<= fim_rep_3[11])  or
        (i> inicio_rep_3[12]  and i<= fim_rep_3[12])  or
        (i> inicio_rep_3[13]  and i<= fim_rep_3[13])  or
        (i> inicio_rep_3[14]  and i<= fim_rep_3[14])  or
        (i> inicio_rep_3[15]  and i<= fim_rep_3[15])  or
        (i> inicio_rep_3[16]  and i<= fim_rep_3[16])  or
        (i> inicio_rep_3[17]  and i<= fim_rep_3[17])  or
        (i> inicio_rep_3[18]  and i<= fim_rep_3[18])  or
        (i> inicio_rep_3[19]  and i<= fim_rep_3[19])  or
        (i> inicio_rep_3[20]  and i<= fim_rep_3[20])  or
        (i> inicio_rep_3[21]  and i<= fim_rep_3[21])  or

        (i> inicio_rep_3[22]  and i<= fim_rep_3[22])  or
        (i> inicio_rep_3[23]  and i<= fim_rep_3[23])  or
        (i> inicio_rep_3[24]  and i<= fim_rep_3[24])  or
        (i> inicio_rep_3[25]  and i<= fim_rep_3[25])  or
        (i> inicio_rep_3[26]  and i<= fim_rep_3[26])  or
        (i> inicio_rep_3[27]  and i<= fim_rep_3[27])  or
        (i> inicio_rep_3[28]  and i<= fim_rep_3[28])  or
        (i> inicio_rep_3[29]  and i<= fim_rep_3[29])  or
        (i> inicio_rep_3[30]  and i<= fim_rep_3[30])  or
        (i> inicio_rep_3[31]  and i<= fim_rep_3[31])  or
        (i> inicio_rep_3[32]  and i<= fim_rep_3[32])  or
        (i> inicio_rep_3[33]  and i<= fim_rep_3[33])  or
        (i> inicio_rep_3[34]  and i<= fim_rep_3[34])  or
        (i> inicio_rep_3[35]  and i<= fim_rep_3[35])  or
        (i> inicio_rep_3[36]  and i<= fim_rep_3[36])  or
        (i> inicio_rep_3[37]  and i<= fim_rep_3[37])  or
        (i> inicio_rep_3[38]  and i<= fim_rep_3[38])  or
        (i> inicio_rep_3[39]  and i<= fim_rep_3[39])  or
        
        (i> inicio_rep_3[40]  and i<= fim_rep_3[40])  or
        (i> inicio_rep_3[41]  and i<= fim_rep_3[41])  or
        (i> inicio_rep_3[42]  and i<= fim_rep_3[42])  or
        (i> inicio_rep_3[43]  and i<= fim_rep_3[43])  or
        (i> inicio_rep_3[44]  and i<= fim_rep_3[44])  or
        (i> inicio_rep_3[45]  and i<= fim_rep_3[45])  or
        (i> inicio_rep_3[46]  and i<= fim_rep_3[46])  or
        (i> inicio_rep_3[47]  and i<= fim_rep_3[47])  or
        (i> inicio_rep_3[48]  and i<= fim_rep_3[48])):
        
        index_rep_3.append(i)
        
    if  ((i> inicio_rep_4[0] and i<= fim_rep_4[0])  or 
        (i> inicio_rep_4[1]  and i<= fim_rep_4[1])  or 
        (i> inicio_rep_4[2]  and i<= fim_rep_4[2])  or
        (i> inicio_rep_4[3]  and i<= fim_rep_4[3])  or
        (i> inicio_rep_4[4]  and i<= fim_rep_4[4])  or
        (i> inicio_rep_4[5]  and i<= fim_rep_4[5])  or
        (i> inicio_rep_4[6]  and i<= fim_rep_4[6])  or 
        (i> inicio_rep_4[7]  and i<= fim_rep_4[7])  or
        (i> inicio_rep_4[8]  and i<= fim_rep_4[8])  or
        (i> inicio_rep_4[9]  and i<= fim_rep_4[9])  or
        (i> inicio_rep_4[10]  and i<= fim_rep_4[10])  or
        (i> inicio_rep_4[11]  and i<= fim_rep_4[11])  or
        (i> inicio_rep_4[12]  and i<= fim_rep_4[12])  or
        (i> inicio_rep_4[13]  and i<= fim_rep_4[13])  or
        (i> inicio_rep_4[14]  and i<= fim_rep_4[14])  or
        (i> inicio_rep_4[15]  and i<= fim_rep_4[15])  or
        (i> inicio_rep_4[16]  and i<= fim_rep_4[16])  or
        (i> inicio_rep_4[17]  and i<= fim_rep_4[17])  or
        (i> inicio_rep_4[18]  and i<= fim_rep_4[18])  or
        (i> inicio_rep_4[19]  and i<= fim_rep_4[19])  or
        (i> inicio_rep_4[20]  and i<= fim_rep_4[20])  or
        (i> inicio_rep_4[21]  and i<= fim_rep_4[21])  or

        (i> inicio_rep_4[22]  and i<= fim_rep_4[22])  or
        (i> inicio_rep_4[23]  and i<= fim_rep_4[23])  or
        (i> inicio_rep_4[24]  and i<= fim_rep_4[24])  or
        (i> inicio_rep_4[25]  and i<= fim_rep_4[25])  or
        (i> inicio_rep_4[26]  and i<= fim_rep_4[26])  or
        (i> inicio_rep_4[27]  and i<= fim_rep_4[27])  or
        (i> inicio_rep_4[28]  and i<= fim_rep_4[28])  or
        (i> inicio_rep_4[29]  and i<= fim_rep_4[29])  or
        (i> inicio_rep_4[30]  and i<= fim_rep_4[30])  or
        (i> inicio_rep_4[31]  and i<= fim_rep_4[31])  or
        (i> inicio_rep_4[32]  and i<= fim_rep_4[32])  or
        (i> inicio_rep_4[33]  and i<= fim_rep_4[33])  or
        (i> inicio_rep_4[34]  and i<= fim_rep_4[34])  or
        (i> inicio_rep_4[35]  and i<= fim_rep_4[35])  or
        (i> inicio_rep_4[36]  and i<= fim_rep_4[36])  or
        (i> inicio_rep_4[37]  and i<= fim_rep_4[37])  or
        (i> inicio_rep_4[38]  and i<= fim_rep_4[38])  or
        (i> inicio_rep_4[39]  and i<= fim_rep_4[39])  or
        
        (i> inicio_rep_4[40]  and i<= fim_rep_4[40])  or
        (i> inicio_rep_4[41]  and i<= fim_rep_4[41])  or
        (i> inicio_rep_4[42]  and i<= fim_rep_4[42])  or
        (i> inicio_rep_4[43]  and i<= fim_rep_4[43])  or
        (i> inicio_rep_4[44]  and i<= fim_rep_4[44])  or
        (i> inicio_rep_4[45]  and i<= fim_rep_4[45])  or
        (i> inicio_rep_4[46]  and i<= fim_rep_4[46])  or
        (i> inicio_rep_4[47]  and i<= fim_rep_4[47])  or
        (i> inicio_rep_4[48]  and i<= fim_rep_4[48])):
        
        index_rep_4.append(i)

    if  ((i> inicio_rep_5[0] and i<= fim_rep_5[0])  or 
        (i> inicio_rep_5[1]  and i<= fim_rep_5[1])  or 
        (i> inicio_rep_5[2]  and i<= fim_rep_5[2])  or
        (i> inicio_rep_5[3]  and i<= fim_rep_5[3])  or
        (i> inicio_rep_5[4]  and i<= fim_rep_5[4])  or
        (i> inicio_rep_5[5]  and i<= fim_rep_5[5])  or
        (i> inicio_rep_5[6]  and i<= fim_rep_5[6])  or 
        (i> inicio_rep_5[7]  and i<= fim_rep_5[7])  or
        (i> inicio_rep_5[8]  and i<= fim_rep_5[8])  or
        (i> inicio_rep_5[9]  and i<= fim_rep_5[9])  or
        (i> inicio_rep_5[10]  and i<= fim_rep_5[10])  or
        (i> inicio_rep_5[11]  and i<= fim_rep_5[11])  or
        (i> inicio_rep_5[12]  and i<= fim_rep_5[12])  or
        (i> inicio_rep_5[13]  and i<= fim_rep_5[13])  or
        (i> inicio_rep_5[14]  and i<= fim_rep_5[14])  or
        (i> inicio_rep_5[15]  and i<= fim_rep_5[15])  or
        (i> inicio_rep_5[16]  and i<= fim_rep_5[16])  or
        (i> inicio_rep_5[17]  and i<= fim_rep_5[17])  or
        (i> inicio_rep_5[18]  and i<= fim_rep_5[18])  or
        (i> inicio_rep_5[19]  and i<= fim_rep_5[19])  or
        (i> inicio_rep_5[20]  and i<= fim_rep_5[20])  or
        (i> inicio_rep_5[21]  and i<= fim_rep_5[21])  or

        (i> inicio_rep_5[22]  and i<= fim_rep_5[22])  or
        (i> inicio_rep_5[23]  and i<= fim_rep_5[23])  or
        (i> inicio_rep_5[24]  and i<= fim_rep_5[24])  or
        (i> inicio_rep_5[25]  and i<= fim_rep_5[25])  or
        (i> inicio_rep_5[26]  and i<= fim_rep_5[26])  or
        (i> inicio_rep_5[27]  and i<= fim_rep_5[27])  or
        (i> inicio_rep_5[28]  and i<= fim_rep_5[28])  or
        (i> inicio_rep_5[29]  and i<= fim_rep_5[29])  or
        (i> inicio_rep_5[30]  and i<= fim_rep_5[30])  or
        (i> inicio_rep_5[31]  and i<= fim_rep_5[31])  or
        (i> inicio_rep_5[32]  and i<= fim_rep_5[32])  or
        (i> inicio_rep_5[33]  and i<= fim_rep_5[33])  or
        (i> inicio_rep_5[34]  and i<= fim_rep_5[34])  or
        (i> inicio_rep_5[35]  and i<= fim_rep_5[35])  or
        (i> inicio_rep_5[36]  and i<= fim_rep_5[36])  or
        (i> inicio_rep_5[37]  and i<= fim_rep_5[37])  or
        (i> inicio_rep_5[38]  and i<= fim_rep_5[38])  or
        (i> inicio_rep_5[39]  and i<= fim_rep_5[39])  or
        
        (i> inicio_rep_5[40]  and i<= fim_rep_5[40])  or
        (i> inicio_rep_5[41]  and i<= fim_rep_5[41])  or
        (i> inicio_rep_5[42]  and i<= fim_rep_5[42])  or
        (i> inicio_rep_5[43]  and i<= fim_rep_5[43])  or
        (i> inicio_rep_5[44]  and i<= fim_rep_5[44])  or
        (i> inicio_rep_5[45]  and i<= fim_rep_5[45])  or
        (i> inicio_rep_5[46]  and i<= fim_rep_5[46])  or
        (i> inicio_rep_5[47]  and i<= fim_rep_5[47])  or
        (i> inicio_rep_5[48]  and i<= fim_rep_5[48])):
        
        index_rep_5.append(i)

    if  ((i> inicio_rep_6[0] and i<= fim_rep_6[0])  or 
        (i> inicio_rep_6[1]  and i<= fim_rep_6[1])  or 
        (i> inicio_rep_6[2]  and i<= fim_rep_6[2])  or
        (i> inicio_rep_6[3]  and i<= fim_rep_6[3])  or
        (i> inicio_rep_6[4]  and i<= fim_rep_6[4])  or
        (i> inicio_rep_6[5]  and i<= fim_rep_6[5])  or
        (i> inicio_rep_6[6]  and i<= fim_rep_6[6])  or 
        (i> inicio_rep_6[7]  and i<= fim_rep_6[7])  or
        (i> inicio_rep_6[8]  and i<= fim_rep_6[8])  or
        (i> inicio_rep_6[9]  and i<= fim_rep_6[9])  or
        (i> inicio_rep_6[10]  and i<= fim_rep_6[10])  or
        (i> inicio_rep_6[11]  and i<= fim_rep_6[11])  or
        (i> inicio_rep_6[12]  and i<= fim_rep_6[12])  or
        (i> inicio_rep_6[13]  and i<= fim_rep_6[13])  or
        (i> inicio_rep_6[14]  and i<= fim_rep_6[14])  or
        (i> inicio_rep_6[15]  and i<= fim_rep_6[15])  or
        (i> inicio_rep_6[16]  and i<= fim_rep_6[16])  or
        (i> inicio_rep_6[17]  and i<= fim_rep_6[17])  or
        (i> inicio_rep_6[18]  and i<= fim_rep_6[18])  or
        (i> inicio_rep_6[19]  and i<= fim_rep_6[19])  or
        (i> inicio_rep_6[20]  and i<= fim_rep_6[20])  or
        (i> inicio_rep_6[21]  and i<= fim_rep_6[21])  or
        
        (i> inicio_rep_6[22]  and i<= fim_rep_6[22])  or
        (i> inicio_rep_6[23]  and i<= fim_rep_6[23])  or
        (i> inicio_rep_6[24]  and i<= fim_rep_6[24])  or
        (i> inicio_rep_6[25]  and i<= fim_rep_6[25])  or
        (i> inicio_rep_6[26]  and i<= fim_rep_6[26])  or
        (i> inicio_rep_6[27]  and i<= fim_rep_6[27])  or
        (i> inicio_rep_6[28]  and i<= fim_rep_6[28])  or
        (i> inicio_rep_6[29]  and i<= fim_rep_6[29])  or
        (i> inicio_rep_6[30]  and i<= fim_rep_6[30])  or
        (i> inicio_rep_6[31]  and i<= fim_rep_6[31])  or
        (i> inicio_rep_6[32]  and i<= fim_rep_6[32])  or
        (i> inicio_rep_6[33]  and i<= fim_rep_6[33])  or
        (i> inicio_rep_6[34]  and i<= fim_rep_6[34])  or
        (i> inicio_rep_6[35]  and i<= fim_rep_6[35])  or
        (i> inicio_rep_6[36]  and i<= fim_rep_6[36])  or
        (i> inicio_rep_6[37]  and i<= fim_rep_6[37])  or
        (i> inicio_rep_6[38]  and i<= fim_rep_6[38])  or
        (i> inicio_rep_6[39]  and i<= fim_rep_6[39])  or
        
        (i> inicio_rep_6[40]  and i<= fim_rep_6[40])  or
        (i> inicio_rep_6[41]  and i<= fim_rep_6[41])  or
        (i> inicio_rep_6[42]  and i<= fim_rep_6[42])  or
        (i> inicio_rep_6[43]  and i<= fim_rep_6[43])  or
        (i> inicio_rep_6[44]  and i<= fim_rep_6[44])  or
        (i> inicio_rep_6[45]  and i<= fim_rep_6[45])  or
        (i> inicio_rep_6[46]  and i<= fim_rep_6[46])  or
        (i> inicio_rep_6[47]  and i<= fim_rep_6[47])  or
        (i> inicio_rep_6[48]  and i<= fim_rep_6[48])):
        index_rep_6.append(i)


label_rep_1 = E1_label[index_rep_1,:]
emg_rep_1 = E1_emg[index_rep_1,:]
#acc_rep_1 = E1_acc[index_rep_1,:]

label_rep_3 = E1_label[index_rep_3,:]
emg_rep_3 = E1_emg[index_rep_3,:]
#acc_rep_3 = E1_acc[index_rep_3,:]

label_rep_4 = E1_label[index_rep_4,:]
emg_rep_4 = E1_emg[index_rep_4,:]
#acc_rep_4 = E1_acc[index_rep_4,:]

label_rep_6 = E1_label[index_rep_6,:]
emg_rep_6 = E1_emg[index_rep_6,:]
#acc_rep_6 = E1_acc[index_rep_6,:]

label_rep_2 = E1_label[index_rep_2,:]
emg_rep_2 = E1_emg[index_rep_2,:]
#acc_rep_2 = E1_acc[index_rep_2,:]

label_rep_5 = E1_label[index_rep_5,:]
emg_rep_5 = E1_emg[index_rep_5,:]
#acc_rep_5 = E1_acc[index_rep_5,:]


# Plota as classes de treinamento
print('\nGráficos de treinamento sem ajuste da classe 0')
plt.subplot(311)
plt.plot(label_rep_1)
plt.ylabel('Repeticao 1')

plt.subplot(312)
plt.plot(label_rep_3)
plt.ylabel('Repeticao 3')
plt.show()

plt.subplot(311)
plt.plot(label_rep_4)
plt.ylabel('Repeticao 4')

plt.subplot(312)
plt.plot(label_rep_6)
plt.ylabel('Repeticao 6')
plt.show()


# Plota as classes de teste
print('\nGráficos de teste sem ajuste da classe 0')
plt.subplot(311)
plt.plot(label_rep_2)
plt.ylabel('Repeticao 2')

plt.subplot(312)
plt.plot(label_rep_5)
plt.ylabel('Repeticao 5')
plt.show()


#####################################################
#          DADOS PARA TREINAMENTO REPETIÇÃO 1
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
# default: C1(7500)     C2(9000)      C3(9000)         C4(12000)      C5(7500)   
#          C6(11000)    C7(11000)     C8(7500/7000)    C9(8000)       C10(9000/8000)
index1 =[]
flag=0
for i in range(len(label_rep_1)):
   if label_rep_1[i]==0:
       if flag <= 11000:                
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_1 = label_rep_1[index1,:]
emg_1 = emg_rep_1[index1,:]
#acc_1 = acc_rep_1[index1,:]


#####################################################
#          DADOS PARA TREINAMENTO REPETIÇÃO 3
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
index1 =[]
flag = 0
for i in range(len(label_rep_3)):
   if label_rep_3[i]==0:
       if flag <= 11000:            
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_3 = label_rep_3[index1,:]
emg_3 = emg_rep_3[index1,:]
#acc_3 = acc_rep_3[index1,:]


#####################################################
#          DADOS PARA TREINAMENTO REPETIÇÃO 4
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
index1 =[]
flag = 0
for i in range(len(label_rep_4)):
   if label_rep_4[i]==0:
       if flag <= 11000:            
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_4 = label_rep_4[index1,:]
emg_4 = emg_rep_4[index1,:]
#acc_4 = acc_rep_4[index1,:]


#####################################################
#          DADOS PARA TREINAMENTO REPETIÇÃO 6
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
index1 =[]
flag = 0
for i in range(len(label_rep_6)):
   if label_rep_6[i]==0:
       if flag <= 11000:            
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_6 = label_rep_6[index1,:]
emg_6 = emg_rep_6[index1,:]
#acc_6 = acc_rep_6[index1,:]


#####################################################
#       JUNTA AS 4 REPETIÇÕES PARA TREINAMENTO
#####################################################

emg_train = np.vstack((emg_1, emg_3, emg_4, emg_6))
label_train = np.vstack((label_1, label_3, label_4, label_6))
#acc_train = np.vstack((acc_1, acc_3, acc_4, acc_6))


# Converte para df[x]: apresenta todos os valores do canal EMG[x] x de 0 a 9
df= pd.DataFrame(emg_train)
#dfa= pd.DataFrame(acc_train)
X_val, Y_val = janelamento(df, label_train, 400, 200)

# Plota as classes de treinamento
print('\nGráficos de treinamento com ajuste da classe 0')
plt.subplot(311)
plt.plot(Y_val)
plt.ylabel('Repeticões')
plt.show()

print('\nBalanço do Dataset de Treinamento')
balanco_classes(Y_val)


#####################################################
#          DADOS PARA TESTE REPETIÇÃO 2
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
index1 =[]
flag=0
for i in range(len(label_rep_2)):
   if label_rep_2[i]==0:
       if flag <= 11000:           
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_2 = label_rep_2[index1,:]
emg_2 = emg_rep_2[index1,:]
#acc_2 = acc_rep_2[index1,:]


#####################################################
#          DADOS PARA TESTE REPETIÇÃO 5
#####################################################

# Considera o movimento de descanso com amostras próximas dos demais gestos
index1 =[]
flag=0
for i in range(len(label_rep_5)):
   if label_rep_5[i]==0:
       if flag <= 11000:            
           index1.append(i)
           flag=flag+1;
   else:
       index1.append(i)
       
label_5 = label_rep_5[index1,:]
emg_5 = emg_rep_5[index1,:]
#acc_5 = acc_rep_5[index1,:]

#####################################################
#          JUNTA AS 2 REPETIÇÕES PARA TESTE
#####################################################

emg_test = np.vstack((emg_2, emg_5))
label_test = np.vstack((label_2, label_5))
#acc_test = np.vstack((acc_2, acc_5))

# Converte para df[x]: apresenta todos os valores do canal EMG[x] x de 0 a 11
df_test= pd.DataFrame(emg_test)
#dfa_test= pd.DataFrame(acc_test)
X_test, Y_test = janelamento(df_test, label_test, 400, 200)

# Plota as classes de teste
print('\nGráficos de teste com ajuste da classe 0')
plt.subplot(311)
plt.plot(Y_test)
plt.ylabel('Repeticões')
plt.show()

print('\nBalanço do Dataset de Teste')
balanco_classes(Y_test)

# Padronização
scaler=StandardScaler()
scaler.fit(X_val)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Redução com LDA
lda = LinearDiscriminantAnalysis(n_components=49)       # n_components = n_class - 1 = 49
X_val = lda.fit_transform(X_val, Y_val)
X_test = lda.transform(X_test)


"""
# Redução com PCA
pca = PCA(n_components=49)
X_val = pca.fit_transform(X_val, Y_val)
X_test = pca.transform(X_test)

"""

###################################################### 
#        Classificador RF - VALIDAÇÃO
###################################################### 
RF_model_val = RandomForestClassifier() 
parameter_space = {
        'n_estimators': [10, 50, 100, 150],   
        'max_depth': [50, 100, 150, 200],                             
        'max_features': ['auto', 'sqrt']}  

for z in range(0, 10):
####################################################
#                   GridSearchCV
####################################################
    print('Tuning by GridSearchCV' )
    grid = GridSearchCV(RF_model_val, parameter_space, cv = 4)

# Fit the  model
    grid.fit(X_val, Y_val)
    print('\nMelhor acurácia:', grid.best_score_)
    print('\nMelhores parâmetros:', grid.best_params_)

###################################################### 
#      Classificador RF - TREINAMENTO E TESTE
###################################################### 
    RF_model = RandomForestClassifier(n_estimators = grid.best_params_['n_estimators'], 
                                      max_depth = grid.best_params_['max_depth'], 
                                      max_features = grid.best_params_['max_features'])  
      
# Treina o sistema (repetições 1, 3, 4 e 6)
    RF_model.fit(X_val, Y_val)

# Função Predição  (repetições 2 e 5) 
    y_pred_test = RF_model.predict(X_test)

# Matriz de Confusão
    show_confusion_matrix(Y_test, y_pred_test)
    
# Mostra a acurácia de cada classe
    show_acuracias_classes(Y_test, y_pred_test)
    
# Calcula a acurácia
    show_acuracia_final(Y_test, y_pred_test)

# Plota sinais EMG, Classes Reais e Preditas
    plotagem(emg_test, Y_test, y_pred_test)