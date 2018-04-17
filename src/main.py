import numpy as np
import pandas as pd
from scipy.spatial import distance

##
##def distance(x_i, x_j):
##    return pow(x_i - x_j,2)
##

#dst = distance.euclidean(a,b)

# ------------------ Load database ---------------
base = np.genfromtxt("C:\\Users\\jlpl\\Desktop\\segmentation.data.txt",delimiter=',', dtype=np.str)
# ------------------ 

# ------------------ Separate database ---------------
labels = base[1:,0]
complet_view = base[1:,1:]
complet_view = complet_view.astype(float)

shape_view = complet_view[:,:9]
shape_view = shape_view.astype(float)

rgb_view =  complet_view[:,9:]
rgb_view =  rgb_view.astype(float)
# ------------------


def get_s2(X):
    # calcula o s2 da base de dados X
    
    distancias=[]

    for i in range(X.shape[0]-1):
        for j in range(i+1, X.shape[0]):
            distancias.append(distance.euclidean(X[i],X[j]))
            #print(str(i) + str(" - ") + str(j))

    distancias = np.sort(distancias)
    
    s2 = (distancias[int(len(distancias)*.1)] + distancias[int(len(distancias)*.9)])/2
    
    return s2


def k(x_i, x_j, s2):
    # Função de distância do artigo
    
    distancia = sum(pow(x_i - x_j, 2))
    k = -1 * (distancia/(s2))
    k = pow(np.e, k)
    
    return k

def gerar_centroides(X, classes):
    # centroides é uma matriz onde as linhas são os centroides e as colunas os atributos
    # linhas = centroides
    # colunas = atributos
    
    centroides = np.random.rand(classes, X.shape[1])

    return centroides

def calcula_p(X, centroides):
    # P é uma matriz 3d onde
    # P[i] retorna a i-ésima região
    # P[i][j] retorna a j-ésima amostra da i-ésima região
    # P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região
    
    p = []

   
 
