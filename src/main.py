import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------ Load database ---------------
base = np.genfromtxt('../base/segmentation.test',delimiter=',', dtype=np.str)
# ------------------ 

# ------------------ Separate database ---------------
labels = base[1:,0]
complet_view = base[1:,1:]
complet_view = complet_view.astype(float)

shape_view = complet_view[:,:9]
shape_view = shape_view.astype(float)

rgb_view = complet_view[:,9:]
rgb_view = rgb_view.astype(float)
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
    
    return np.ndarray.tolist(centroides)

def calcula_p(X, centroides):
    # X é a base de dados
    # P é uma matriz 3d onde
    # P[i] retorna a i-ésima região
    # P[i][j] retorna a j-ésima amostra da i-ésima região
    # P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região
    # p[i][0] = centroide da i-ésima região
    
    p = {}
    s2 = get_s2(X)

    for i in range(len(centroides)):
        p[i] = [centroides[i]]
    
    for i in range(len(X)):
        
        valor_minimo = -1
        indice_valor_minimo = -1

        for j in range(len(centroides)):
            
            distancia = 2*(1 - k(X[i], centroides[j], s2))

            if (j == 0):
                
               valor_minimo = distancia
               indice_valor_minimo = j
               
            elif (distancia<valor_minimo):
                
                valor_minimo = distancia
                indice_valor_minimo = j

        p[indice_valor_minimo].append(np.ndarray.tolist(X[i]))
        
    return p

# --------------------------------------------------------------


centroides = gerar_centroides(shape_view, 7)
p = calcula_p(shape_view, centroides)

pca = PCA(n_components=2)
pca = pca.fit(shape_view)

for i in range(len(p)):
    p_i = pca.transform(p[i])
    p_a = np.array(p_i)
    plt.plot(p_a[:,0], p_a[:,1], 'ro', linewidth=0.03)
    plt.plot(p_a[0][0], p_a[0][1], 'g^', linewidth=0.03)
    
plt.show()
