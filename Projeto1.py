import librosa
import sklearn
import numpy as np
from os import listdir
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans  

n_mfcc    = 12
rotulos   = []
atributos = []
rotulo    = ""

#Extrai todos as matrizes dos MFCC dos aúdios percursivos de treinamento 
for arquivo in listdir():
    if 'wav' in arquivo:
        audio, fs = librosa.load(arquivo,None)
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        mfcc   =  librosa.feature.mfcc(y_percussive, sr=fs, n_mfcc=n_mfcc).T
        scaler = sklearn.preprocessing.StandardScaler()
        mfcc_scaled = scaler.fit_transform(mfcc)
        atributos.append(mfcc_scaled)
        if ('a'  in [arquivo[0:1]]) or ('b' in [arquivo[0:1]])  or ('d' in [arquivo[0:1]]):
            rotulo = 'g1'
        elif ('m' in [arquivo[0:1]]) or ('n' in [arquivo[0:1]]) or ('h' in [arquivo[0:1]]):
            rotulo = 'g2'
        elif ('x' in [arquivo[0:1]]) or ('6' in [arquivo[0:1]]) or ('7' in [arquivo[0:1]]) or ('c' in [arquivo[0:1]]):
            rotulo = 'g3'
        rotulos=rotulos+[rotulo]*len(mfcc_scaled)

#Constroi o modelo
atributos = np.vstack((atributos))
rotulos   = np.array(rotulos)
model = sklearn.ensemble.GradientBoostingClassifier() #OVA(one vs all)
model.fit(atributos, rotulos)


# Abre o arquivo de teste
audio, fs = librosa.load('nbdd.c',None)
tamanho = audio.shape[0]
tempo   = audio.shape[0]/fs
#Retira as características para a segmentação
y_harmonic, y_percussive = librosa.effects.hpss(audio)
D = np.abs(librosa.stft(y_percussive))**2
S = librosa.feature.melspectrogram(S=D)
db = librosa.amplitude_to_db(S)

bounds = librosa.segment.agglomerative(db, 40)
bound_times = librosa.frames_to_time(bounds, sr=fs)
bound_times = np.vstack((bound_times[2:], [0]*len(bound_times[2:]))).T


kmeans = KMeans(n_clusters=4)  
kmeans.fit(bound_times[2:])  

func1 = lambda x: int(tamanho*(x/tempo))
centros =  np.array([func1(xi) for xi in kmeans.cluster_centers_[:,0]])
centros =  np.append(centros,tamanho)
centros.sort()

ultimo = 0
for i,j in zip(centros,centros[1:]):
    sl = int((i+j)/2)
    dado = audio[ultimo:sl]
    y_harmonic, y_percussive = librosa.effects.hpss(dado)
    mfcc   =  librosa.feature.mfcc(y_percussive, sr=fs, n_mfcc=n_mfcc).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)
    preditor = model.predict(mfcc_scaled)
    print(([(preditor == c).sum() for c in ('g1','g2','g3')]))
    librosa.output.write_wav('%s.wav' %i, dado,  fs)
    ultimo = sl

