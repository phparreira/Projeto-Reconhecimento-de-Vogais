from os import listdir
import os
import librosa
import sklearn
import sklearn
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import exp
from sklearn.cluster import KMeans  
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
nLetras   = 4
n_mfcc    = 12

def rotuloGrupo(rotulo):
    if ('a'  in rotulo) or ('b' in rotulo)  or ('d' in rotulo):
        return  'g1'
    elif ('m' in rotulo) or ('n' in rotulo) or ('h' in rotulo):
        return  'g2'
    elif ('x' in rotulo) or ('6' in rotulo) or ('7' in rotulo) or ('c' in rotulo):
        return  'g3'

def SegmentarAudio(audio,fs,n,tamanho,tempo):
    #y_percussive = librosa.effects.percussive(audio)
    #D = np.abs(librosa.stft(y_percussive))**2
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


scaler = sklearn.preprocessing.StandardScaler()
atributos = []
rotulos   = []
nItens = 0
for arquivo in listdir():
    if 'wav' in arquivo: 
        audio, fs = librosa.load(arquivo,None)
        tamanho = audio.shape[0]
        tempo   = audio.shape[0]/fs
        retorno = SegmentarAudio(audio,fs,nLetras,tamanho,tempo)
        nItens = nItens + 1
        print(nItens)
        for i in range(0,nLetras):
             S       = np.abs(librosa.stft(retorno[i,]))
             y_harm, y_per = librosa.effects.hpss(retorno[i,])
             #p2      = librosa.feature.poly_features(S=S, order=2)
             #atr1    = librosa.feature.chroma_stft(y=retorno[i,], sr=fs).T
             mfcc1   =  librosa.feature.mfcc(y_harm, sr=fs, n_mfcc=n_mfcc).T
             #atr1    = scaler.fit_transform(mfcc1)
             mfcc2   =  librosa.feature.mfcc(y_per, sr=fs, n_mfcc=n_mfcc).T
             #atr2    = scaler.fit_transform(mfcc2)
             chrm1   = librosa.feature.chroma_stft(y=retorno[i,], sr=fs).T
             t = np.concatenate([mfcc1,chrm1])
             atr1    = scaler.fit_transform(t)
             #print(atr1.shape)
             atributos.append(atr1)
             #rotulos=rotulos+[(arquivo[i])]*len(atr1)
             rotulos=rotulos+[rotuloGrupo(arquivo[i])]*len(atr1)

atributos = np.vstack((atributos))
rotulos   = np.array(rotulos)
#Naive Bayses
#gnb = GaussianNB()
#model = sklearn.ensemble.GradientBoostingClassifier() #OVA(one vs all)
model = RandomForestClassifier(n_estimators=10)
model.fit(atributos, rotulos)



#Testa
os.chdir('base_validacao_I')


scaler = sklearn.preprocessing.StandardScaler()
atributos = []
rotulos   = []
nAudios   = 0
nAcertos  = 0
nErros    = 0 
nItens    = 0
pred      = []
vdd       = []
for arquivo in listdir():
    if 'wav' in arquivo: 
        print("Novo")
        print(nErros)
        print(nAcertos)
        audio, fs = librosa.load(arquivo,None)
        tamanho = audio.shape[0]
        tempo   = audio.shape[0]/fs
        retorno = SegmentarAudio(audio,fs,nLetras,tamanho,tempo)
        nAudios = nAudios + 4
        nItens = nItens + 1
        print(nItens)
        for i in range(0,nLetras):
             S       = np.abs(librosa.stft(retorno[i,]))
             y_harm, y_per = librosa.effects.hpss(retorno[i,])
             #p2      = librosa.feature.poly_features(S=S, order=2)
             #mfcc   =  librosa.feature.mfcc(retorno[i,], sr=fs, n_mfcc=n_mfcc).T
             #atr1    = librosa.feature.chroma_stft(y=retorno[i,], sr=fs).T
             mfcc1   =  librosa.feature.mfcc(y_harm, sr=fs, n_mfcc=n_mfcc).T
             #atr1    = scaler.fit_transform(mfcc1)
             mfcc2   =  librosa.feature.mfcc(y_per, sr=fs, n_mfcc=n_mfcc).T
             #atr2    = scaler.fit_transform(mfcc2)
             chrm1   = librosa.feature.chroma_stft(y=retorno[i,], sr=fs).T
             t = np.concatenate([mfcc1,chrm1])
             atr1    = scaler.fit_transform(t)

             pred    = model.predict(atr1)   
             (values,counts) = np.unique(pred,return_counts=True)
             ind = np.argmax(counts)
             #vdd = np.append(vdd,(arquivo[i],values[ind]))
             if values[ind] == rotuloGrupo(arquivo[i]):
             #if values[ind] == arquivo[i]:
                nAcertos = nAcertos + 1
                print("ACERTOU")
             else:
                nErros = nErros + 1
                print("ERROU")
                print(arquivo[i])
                print(values[ind])

print(nAudios)
print(nAcertos)
print(nErros)

_ = joblib.dump(model, 'modelo_FT_Grupo.bb', compress=9)

#BASELINE
#---------------------------------------------------------------------------------------------------------------------
#TESTE1 -> NB + CHROMA_STFT        12.8% [90/702]
#TESTE2 -> NB + MFCC               12.8% [90/702]
#TESTE4 ->GradientBoostingClassifier + MFCC        45.96% [364/792]
#TESTE5 ->GradientBoostingClassifier + CHROMA_STFT 14.39% [114/792]
#TESTE7 ->RandomForestClassifier     + MFCC        56.18% [445/792] 
#TESTE7 ->RandomForestClassifier     + MFCC[h/p]   57.19% [453/792] 
#---------------------------------------------------------------------------------------------------------------------
#TESTE6 ->GradientBoostingClassifier + MFCC 69.82% [553/792] [GRUPOS]
#TESTE6 ->RandomForestClassifier     + MFCC 77.39% [613/792] [GRUPOS]
#TESTE6 ->RandomForestClassifier     + MFCC[h/p] 79.92% [633/792] [GRUPOS]
