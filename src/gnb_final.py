import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import LabelEncoder
import math
from sklearn import preprocessing
import pickle

class Naive_Bayes_Classifier(object):

    def train (self, X):

        _X = []
        y = []
        for c in range(len(X)):
            for sample in range(len(X[c])):
                sam = np.array(X[c][sample])
                _X.append(sam)
                y.append(c)
        y = np.array(y)
        X = np.array(_X)
       
        self.classes = set(y)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.c_mean = np.zeros((len(self.classes), X.shape[1]))
        self.c_std = np.zeros((len(self.classes), X.shape[1]))
        self.c_cov = np.zeros((len(self.classes), X.shape[1], X.shape[1]))

        self.prior = np.zeros((len(self.classes),))
        for c in self.classes:
            indices = np.where(y == c)
            self.prior[c] = indices[0].shape[0] / float(y.shape[0])
            self.c_mean[c] = np.mean(X[indices], axis=0)
            self.c_std[c] = np.std(X[indices], axis=0)
            self.c_cov[c] = np.cov(X[indices].T)
            
        return



    def predict (self, obs):
        
        p = []
        posterioris = []
        posterior=[]

        
        for c in range(len(self.classes)):
##            cov = np.diag(np.diag(self.c_cov[c])) 
##            left = math.pow(2*np.pi,(-obs.shape[0]/2))
##            inv = np.linalg.inv(cov)
##            left = left * math.pow(np.linalg.det(inv),1/2)
##
##            up = (-1/2*(obs - self.c_mean[c])).reshape(1,obs.shape[0])
##            inv2 =np.linalg.inv(cov)
##            up_left = np.dot(up,inv2)
##            upp = np.dot(up_left, (obs - self.c_mean[c]))
##            exp =  math.exp(upp)
##            likelihood =  left * exp
            likelihood = multivariate_normal.pdf(obs,self.c_mean[c],np.diag(np.diag(self.c_cov[c])) , allow_singular=True)
            
            posterior.append(likelihood*self.prior[c])
        posterior = posterior/sum(posterior)
        posterioris.append(posterior)
        p.append(np.argmax(posterior))
        return p, posterioris
 
# ------------------
# ------------------
def normaliza(X):
    maximos = []
    minimos = []

    for i in range(len(X[0])):
        maximos.append(max(X[:, i]))
        minimos.append(min(X[:, i]))

    for i in range(len(X)):

        for j in range(len(X[0])):
            X[i, j] = (X[i, j] - minimos[j]) / (maximos[j] - minimos[j])


    return X 
# ------------------
# ------------------
def remove_coluna(X, index):

    part1 = X[:, :index]
    part2 = X[:, index+1:]

    return np.concatenate([part1, part2], axis=1)
# ------------------

# ------------------ Preparando Dados ------------------

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

import numpy as np



from sklearn.preprocessing import LabelEncoder


base = np.genfromtxt('../base/segmentation.test',delimiter=',', dtype=np.str)
base_test = np.genfromtxt('../base/segmentation.data',delimiter=',', dtype=np.str)

labels = base[1:,0]
labels_test = base_test[1:,0]


complet_view = base[1:,1:]

complet_view = complet_view.astype(float)
complet_view_test = base_test[1:,1:]
complet_view_test = complet_view_test.astype(float)

shape_view = complet_view[:,:9]
shape_view = shape_view.astype(float)
shape_view = remove_coluna(shape_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
#shape_view = normaliza(shape_view)

shape_view_test = complet_view_test[:,:9]
shape_view_test = shape_view_test.astype(float)
shape_view_test = remove_coluna(shape_view_test, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
#shape_view_test = normaliza(shape_view_test)


rgb_view = complet_view[:,9:]
rgb_view = rgb_view.astype(float)
#rgb_view = normaliza(rgb_view)

rgb_view_test = complet_view_test[:,9:]
rgb_view_test = rgb_view_test.astype(float)
#rgb_view_test = normaliza(rgb_view_test)


complet_view = remove_coluna(complet_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras

##complet_view = normaliza(complet_view)

complet_view_test = remove_coluna(complet_view_test, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
##complet_view_test = normaliza(complet_view_test)





encoder = LabelEncoder() #Codifica as classes (String) em valores inteiros.
encoder.fit(labels)
y = encoder.transform(labels)

posterioris_complet=[]
means_complet = []
for i in range(30):
    fold_acc = 0
    means_folds = 0
    for f in range(10):
        arq = open('../folders/complet_view_'+str(i)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
        dic = pickle.load(arq)#Ler 

        X, X_test = separa_treino_teste(dic, f,1)
        gnb = Naive_Bayes_Classifier()
        gnb.train(X)
        acc =0
        p, posterioris = [], []
        y_test  = []
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                y_test.append(c)
                predicted, posterior = gnb.predict(np.array(sample))
                p.append(predicted)
                posterioris_complet.append(posterior)
                

        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                    acc+=1
        #print("rep "+ str(i) + " fold "+ str(f) + " Accuracy:"+str(acc/len(p)))
        
        fold_acc+= acc/len(p)
    means_complet.append(fold_acc/10)




means_rgb = []
posterioris_rgb=[]
for i in range(30):
    fold_acc = 0
    mean_folds = 0
    for f in range(10):
        arq = open('../folders/rgb_view_'+str(i)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
        dic = pickle.load(arq)#Ler 

        X, X_test = separa_treino_teste(dic, f,0)
        gnb = Naive_Bayes_Classifier()
        gnb.train(X)
        acc =0
        p, posterioris = [], []
        y_test  = []
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                y_test.append(c)
                predicted, posterior = gnb.predict(np.array(sample))
                p.append(predicted)
                posterioris_rgb.append(posterior)


        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                    acc+=1
        #print("rep "+ str(i) + " fold "+ str(f) + " Accuracy:"+str(acc/len(p)))
        
        fold_acc+= acc/len(p)
    means_rgb.append(fold_acc/10)



means_shape = []
posterioris_shape =[]
for i in range(30):
    fold_acc = 0
    mean_folds = 0
    for f in range(10):
        arq = open('../folders/shape_view_'+str(i)+'.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
        dic = pickle.load(arq)#Ler 

        X, X_test = separa_treino_teste(dic, f,1)
        gnb = Naive_Bayes_Classifier()
        gnb.train(X)
        acc =0
        p, posterioris = [], []
        y_test  = []
        for c in range(len(X_test)):
            for obs in range(len(X_test)):
                sample = X_test[c][obs]
                y_test.append(c)
                predicted, posterior = gnb.predict(np.array(sample))
                p.append(predicted)
                posterioris_shape.append(posterior)


        for prediction, actual in zip(p, y_test):
            if(actual == prediction[0]):
                    acc+=1
        #print("rep "+ str(i) + " fold "+ str(f) + " Accuracy:"+str(acc/len(p)))
        
        fold_acc+= acc/len(p)
    means_shape.append(fold_acc/10)

        

print("Complet")
print(means_complet)
print(np.mean(means_complet))

print("shape")
print(means_shape)
print(np.mean(means_shape))

print("rgb")
print(means_rgb)
print(np.mean(means_rgb))

print("Saving means")
np.save("gnb_mean_complets.npy", means_complet)
np.save("gnb_mean_shapes.npy", means_shape)
np.save("gnb_mean_rgbs.npy", means_rgb)

print("Saving post")

np.save("gnb_posterioris_complet.npy", posterioris_complet)
np.save("gnb_posterioris_shape.npy", posterioris_shape)
np.save("gnb_posterioris_rgb.npy", posterioris_rgb)



##
##encoder = LabelEncoder() #Codifica as classes (String) em valores inteiros.
##encoder.fit(labels_test)
##y_test = encoder.transform(labels_test)
##
##
##
##X = complet_view
##X_test = complet_view_test
##
##
##X = preprocessing.scale(X)
##X_test = preprocessing.scale(X_test)
##  
##gnb = Naive_Bayes_Classifier()
##gnb.train(X, y)
##acc =0
##p, posterioris = [], []
##for obs in X_test:
##    predicted, posterior = gnb.predict(obs)
##    p.append(predicted)
##    posterioris.append(posterior)
##
##for prediction, actual in zip(p, y_test):
##    if(actual == prediction):
##        acc+=1
##     
##print("Accuracy:"+str(acc/len(p)))


 
