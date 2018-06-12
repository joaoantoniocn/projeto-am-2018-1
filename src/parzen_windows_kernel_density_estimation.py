import numpy as np
import operator
from sklearn.preprocessing import LabelEncoder
import math
import pickle
import  scipy
from functools import reduce

class Parzen(object):
    

    def train(self, X):
        self.classes = np.arange(len(X))
        self.prior = np.zeros((len(self.classes),))
        for c in self.classes:
            self.prior[c] = len(X[c])/len(X[c]*len(X)) 
        return
    def predict(self, x_samples, center, h2):
        posterior = []
        #Density with parzen
        p_x=[]
        dimensions = len(x_samples[0][0])
        p = []
        for c in range(len(x_samples)):
            sum_x=0
            for x in range(len(x_samples[c])):
                k = []
                
                for axis,center_point in zip(x_samples[c][x], center):
                    left = 1/(math.sqrt(2*np.pi))
                    u = (center_point - axis)/h2
                    right = math.exp(-math.pow(u,2)/2)
                    k.append(left*right)
                prod_k = reduce(lambda x,y:x*y,k)
                sum_x+=prod_k
                p_x = (sum_x)/ len(x_samples)*(math.pow(h2,dimensions))
            posterior.append(p_x*self.prior[c])# / len(x_samples)*(math.pow(h,dimensions))))
        
        posterior = posterior/sum(posterior)
        
        p.append(np.argmax(posterior))
        
        return p, posterior
        
        
        
##
##    def parzen_gaussian(self, x_samples, center, h):
##        p_x=[]
##        dimensions = len(x_samples[0][0])
##        
##        for c in range(len(x_samples)):
##            for x in range(1,len(x_samples[c])):
##                _phi = 1
##                for axis,center_point in zip(x_samples[c][x], center):
##                    _phi *= self.GaussianKernel(axis, center_point, h)
##            p_x.append((_phi / len(x_samples) / (h**dimensions)))
##            
##        return p_x                    
##
## 

        
    def Gaussian(x,z,sigma,axis=None):
        return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))

    def GaussianKernel(self, v1, v2, sigma):
        return np.exp(-np.linalg.norm(v1-v2, None)**2/(2.*sigma**2))

#gerar dois numeros entre 0 e 9 diferentes de f
        
def separa_treino_teste_val(folders, indice_teste, remove_col, indice_validacao):

    treino = {}
    teste = {}
    validacao = {}
    
    if (remove_col == 1):
        for folder in range(len(folders)):
            for classe in range(len(folders[folder])):

                if (classe not in treino):
                    treino[classe] = []
                    teste[classe] = []
                    validacao[classe] = []

                for amostra in range(len(folders[folder][classe])):
                    if (folder == indice_teste):
                        # np.delete(a, 1, 0)
                        teste[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))

                    elif((folder == indice_validacao[0]) or (folder == indice_validacao[1])):
                        validacao[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))
                    else:
                        treino[classe].append(np.delete(folders[folder][classe][amostra], 2, 0))
    else:
        for folder in range(len(folders)):
            for classe in range(len(folders[folder])):

                if (classe not in treino):
                    treino[classe] = []
                    teste[classe] = []
                    validacao[classe] = []

                for amostra in range(len(folders[folder][classe])):
                    if (folder == indice_teste):
                        # np.delete(a, 1, 0)
                        teste[classe].append(folders[folder][classe][amostra])
                    elif ((folder == indice_validacao[0]) or (folder == indice_validacao[1])):
                        validacao[classe].append(folders[folder][classe][amostra])
                    else:
                        treino[classe].append(folders[folder][classe][amostra])

    return treino, teste, validacao

def separa_treino_teste( folders, indice_teste, remove_col):

    treino = {}
    teste = {}
    if(remove_col == 1):
        for folder in range(len(folders)):
            for classe in range(len(folders[folder])):

                if (classe not in treino):
                    treino[classe] = []
                    teste[classe] = []

                for amostra in range(len(folders[folder][classe])):
                    if (folder != indice_teste):
                        #np.delete(a, 1, 0)
                        treino[classe].append(np.delete(folders[folder][classe][amostra],2,0))
                    else:
                        teste[classe].append(np.delete(folders[folder][classe][amostra],2,0))
    else:
        for folder in range(len(folders)):
            for classe in range(len(folders[folder])):

                if (classe not in treino):
                    treino[classe] = []
                    teste[classe] = []

                for amostra in range(len(folders[folder][classe])):
                    if (folder != indice_teste):
                        #np.delete(a, 1, 0)
                        treino[classe].append(folders[folder][classe][amostra])
                    else:
                        teste[classe].append(folders[folder][classe][amostra]) 
    return treino, teste


def remove_coluna(X, index):

    part1 = X[:, :index]
    part2 = X[:, index+1:]

    return np.concatenate([part1, part2], axis=1)


#create a dict where each class has your vector samples
def separateByClass(_dataset, _labels):
        separated = {}
        for i in range(len(_dataset)):
                vector = _dataset[i]
                if (_labels[i] not in separated):
                        separated[_labels[i]] = []
                separated[_labels[i]].append(vector)
        return separated

def normaliza(X):
    from sklearn import preprocessing
    from scipy import stats
    for c in range(7):
        #X[c] = preprocessing.scale(X[c])

        #X[c] = (X[c] - np.mean(X[c])) / np.std(X[c])
        X[c] =  (X[c] - np.min(X[c]))/np.ptp(X[c])
    return X
    #X = stats.zscore(complet_view)#, axis=1, ddof=1)
    #X_test = stats.zscore(complet_view_test)#, axis=1, ddof=1)
        

def get_val_idx(idx_test):

    gen1 = np.random.randint(10)
    while(gen1 == idx_test):
        gen1 = np.random.randint(10)
    gen2 = np.random.randint(10)
    while(gen2 == idx_test or gen2 == gen1):
        gen2 = np.random.randint(10)
    return [gen1, gen2]
    

posterioris_complet=[]
means_complet = []
for i in range(30):
    arq = open('../folders/complet_view_'+str(0)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
    dic = pickle.load(arq)#Ler 

    print("Fold CompletView:", i)
    fold_acc = 0
    means_folds = 0
    
    for f in range(10):
        
        X, X_test, X_val = separa_treino_teste_val(dic, f,1,get_val_idx(f))
        acc = 0
        
        parzen = Parzen()
        parzen.train(X)
        best_h = 1
        best_acc = 0
        ##Selecionando o melhor H
        for h in range(1,10):
            _h = h
            y_val=[]
            p, posterioris = [], []
            
            for c in range(len(X_val)):
                for obs in range(len(X_val)):
                    sample = X_val[c][obs]
                     
                    y_val.append(c)
                    predicted, posterior =parzen.predict(X,np.array(sample),_h)
                    p.append(predicted)
                    
            
            for prediction, actual in zip(p, y_val):
                if(actual == prediction[0]):
                    acc+=1
            if(best_acc < acc/len(y_val)*100):
                best_acc = acc/len(y_val)*100
                best_h = h
            #print("h:",_h,"    | Accuracy:", acc/len(y_val)*100, "%")
            acc = 0
        acc = 0
        _h = best_h
        y_test=[]
        p, posterioris = [], []
        
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                 
                y_test.append(c)
                predicted, posterior =parzen.predict(X,np.array(sample),_h)
                p.append(predicted)
                posterioris_complet.append([posterior, c])
                
        
        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                acc+=1   
        fold_acc+= acc/len(p)
    means_complet.append(fold_acc/10)


posterioris_rgb=[]
means_rgb = []
for i in range(30):
    arq = open('../folders/rgb_view_'+str(0)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
    dic = pickle.load(arq)#Ler 

    print("Fold RGB View:", i)
    fold_acc = 0
    means_folds = 0
    
    for f in range(10):
        
        X, X_test, X_val = separa_treino_teste_val(dic, f,1,get_val_idx(f))
        acc = 0
        
        parzen = Parzen()
        parzen.train(X)
        best_h = 1
        best_acc = 0
        ##Selecionando o melhor H
        for h in range(1,10):
            _h = h
            y_val=[]
            p, posterioris = [], []
            
            for c in range(len(X_val)):
                for obs in range(len(X_val)):
                    sample = X_val[c][obs]
                     
                    y_val.append(c)
                    predicted, posterior =parzen.predict(X,np.array(sample),_h)
                    p.append(predicted)
                    
            
            for prediction, actual in zip(p, y_val):
                if(actual == prediction[0]):
                    acc+=1
            if(best_acc < acc/len(y_val)*100):
                best_acc = acc/len(y_val)*100
                best_h = h
            #print("h:",_h,"    | Accuracy:", acc/len(y_val)*100, "%")
            acc = 0
        acc = 0
        _h = best_h
        y_test=[]
        p, posterioris = [], []
        
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                 
                y_test.append(c)
                predicted, posterior =parzen.predict(X,np.array(sample),_h)
                p.append(predicted)
                posterioris_rgb.append([posterior, c])
                
        
        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                acc+=1   
        fold_acc+= acc/len(p)
    means_rgb.append(fold_acc/10)

posterioris_shape=[]
means_shape = []
for i in range(30):
    arq = open('../folders/shape_view_'+str(0)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
    dic = pickle.load(arq)#Ler 

    print("Fold Shape_View:", i)
    fold_acc = 0
    means_folds = 0
    
    for f in range(10):
        
        X, X_test, X_val = separa_treino_teste_val(dic, f,1,get_val_idx(f))
        acc = 0
        
        parzen = Parzen()
        parzen.train(X)
        best_h = 1
        best_acc = 0
        ##Selecionando o melhor H
        for h in range(1,10):
            _h = h
            y_val=[]
            p, posterioris=[], []
            
            for c in range(len(X_val)):
                for obs in range(len(X_val)):
                    sample = X_val[c][obs]
                     
                    y_val.append(c)
                    predicted, posterior =parzen.predict(X,np.array(sample),_h)
                    p.append(predicted)
                    
            
            for prediction, actual in zip(p, y_val):
                if(actual == prediction[0]):
                    acc+=1
            if(best_acc < acc/len(y_val)*100):
                best_acc = acc/len(y_val)*100
                best_h = h
            #print("h:",_h,"    | Accuracy:", acc/len(y_val)*100, "%")
            acc = 0
        acc = 0
        _h = best_h
        y_test=[]
        p, posterioris =[], []
        
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                 
                y_test.append(c)
                predicted, posterior =parzen.predict(X,np.array(sample),_h)
                p.append(predicted)
                posterioris_shape.append([posterior, c])
                
        
        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                acc+=1   
        fold_acc+= acc/len(p)
    means_shape.append(fold_acc/10)


print("Saving means")
np.savetxt("parzen_mean_shapes.txt", means_shape,fmt='%s')
np.savetxt("parzen_mean_rgbs.txt", means_rgb,fmt='%s')
np.savetxt("parzen_mean_complets.txt", means_complet,fmt='%s')

print("Saving post")
np.save("parzen_posterioris_shape.npy", posterioris_shape)
np.save("parzen_posterioris_rgb.npy", posterioris_rgb)
np.save("parzen_posterioris_complet.npy", posterioris_complet)


            
        
     
##
