import os
import librosa
import sklearn
import numpy as np
from os import listdir
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

def SegmentarAudio(audio,fs,n,tamanho,tempo):
    D = np.abs(librosa.stft(audio))**2
    S = librosa.feature.melspectrogram(S=D)
    db = librosa.amplitude_to_db(S)
    bounds = librosa.segment.agglomerative(db, 40)
    bound_times = librosa.frames_to_time(bounds, sr=fs)
    bound_times = np.vstack((bound_times[2:], [0]*len(bound_times[2:]))).T
    kmeans = KMeans(n_clusters=n)  
    kmeans.fit(bound_times[2:])  
    func1 = lambda x: int(tamanho*(x/tempo))
    centros =  np.array([func1(xi) for xi in kmeans.cluster_centers_[:,0]])
    centros =  np.append(centros,tamanho)
    centros.sort()
    ultimo  = 0
    retorno = []
    for i,j in zip(centros,centros[1:]):
        sl = int((i+j)/2)
        retorno.append(audio[ultimo:sl])
        ultimo = sl
    return np.array(retorno)


# Treina o o modelo
os.chdir('base_treinamento_I')

nItens      = 0
nLetras     = 4
n_mfcc      = 64
atributos1  = []
atributos2  = []
atributos3  = []
rotulos1    = []
rotulos2    = []
rotulos3    = []
nArq        = 0


print('Semente...')
np.random.seed(seed = 42)


scaler = sklearn.preprocessing.StandardScaler()

for arquivo in listdir():
    if 'wav' in arquivo: 
        audio, fs = librosa.load(arquivo,None)
        tamanho = audio.shape[0]
        tempo   = audio.shape[0]/fs
        retorno = SegmentarAudio(audio, fs, nLetras, tamanho, tempo)
        nItens = nItens + 1
        print(nItens)   
        print(arquivo)
        for i in range(0,nLetras):
            S       = np.abs(librosa.stft(retorno[i,],hop_length=128))**2
            y_harm, y_per = librosa.effects.hpss(retorno[i,])
            mfcc1   =  librosa.feature.mfcc(y_harm,  sr=fs, n_mfcc=n_mfcc,hop_length=128).T
            mfccD   = librosa.feature.mfcc(y=retorno[i,], sr=fs, dct_type=2,n_mfcc=1,hop_length=64).T

            #Atributo Secundario

            atr1    = scaler.fit_transform(mfcc1)
            atributos1.append(atr1)
            rotulos1=rotulos1+[(arquivo[i])]*len(atr1)


atributos = np.vstack(atributos1)
rotulos   = np.array(rotulos1)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

model1 = RandomForestClassifier(n_estimators=10,criterion="entropy")
model1.fit(atributos, rotulos)


os.chdir("..")
os.chdir('base_validacao_I')

vReais    = []
vPreditos = []
nAcertos  = 0
nItem     = 0
nErros    = 0
nItens    = 0
for arquivo in listdir():
    if 'wav' in arquivo:
        nItem = nItem + 1 
        print(nItem)
        print(nAcertos)
        audio, fs = librosa.load(arquivo,None)
        tamanho = audio.shape[0]
        tempo   = audio.shape[0]/fs
        retorno = SegmentarAudio(audio,fs,nLetras,tamanho,tempo)
        nAcerto = 0
        for i in range(0,nLetras):
             S       = np.abs(librosa.stft(retorno[i,],hop_length=128))**2
             y_harm, y_per = librosa.effects.hpss(retorno[i,])
             mfcc1   =  librosa.feature.mfcc(y_harm,  sr=fs, n_mfcc=n_mfcc,hop_length=128).T
             mfccD   = librosa.feature.mfcc(y=retorno[i,], sr=fs, dct_type=2,n_mfcc=1,hop_length=64).T

             atr1    = scaler.fit_transform(mfcc1)
             pred    = model1.predict(atr1) 

             (values,counts) = np.unique(pred,return_counts=True)
             ind = np.argmax(counts)

             if values[ind] == arquivo[i]:
                nAcerto = nAcerto + 1
                print("ACERTOU_NOVO")
             else:
                print("ERROU NOVO / Valor Real:",arquivo[i],"/ Valor Predito: ",values[ind])

             vPreditos.append(values[ind])
             vReais.append(arquivo[i])
             if nAcerto == 4:
                 nAcertos = nAcertos + 1
             else:
                 nErros = nErros + 1


print("Total de Acertos:",nAcertos)
print("Total de Erros:",nErros)

Areais=np.array(vReais)
Apreditos=np.array(vPreditos)

import pandas as pd
print (pd.crosstab(Areais,Apreditos,rownames=['REAL'],colnames=['PREDITO'],margins=True))
