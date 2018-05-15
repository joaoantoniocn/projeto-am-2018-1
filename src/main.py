import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from random import randint

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

# ------------------ Funções ------------------

# ------------------
def atualiza_s2(p, centroides, s2,y):
# calcula o novo vetor de hyper-parametros s2

    novo_s2=[]

    for j in range(len(s2)):
    # calculando o j-ésimo s2
        s2_j = 0
        nominador = 0 # parte de cima da fração
        denominador = 0 # parte de baixo da fração
        prod_centroide_h = 1
        soma_centroide_j = 0
    
        for h in range(len(s2)):
        # Produtorio do h-ésimo atributo
        # len(p) = len(s2), p = número de atributos
            soma_centroide_h = 0
            soma_centroide_j = 0
            
            for i in range(len(centroides)):
            # passando por todos os centroides
                soma_dist_h = 0
                soma_dist_j = 0
                
                for k in range(len(p[i])):
                # passando por todas as amostras da região p_i
                    
                    dist_centroide = calcula_k(p[i][k], centroides[i], s2)
                    
                    dist_atributo_h = pow(p[i][k][h] - centroides[i][h], 2)                    
                    dist_atributo_j = pow(p[i][k][j] - centroides[i][j], 2)                    
                    soma_dist_h += dist_centroide * dist_atributo_h                    
                    soma_dist_j += dist_centroide * dist_atributo_j
                    

                soma_centroide_h += soma_dist_h
                soma_centroide_j += soma_dist_j
            
            prod_centroide_h *= soma_centroide_h
            #print("soma_centroide_h")
            #print(soma_centroide_h)
        parte_cima = pow(prod_centroide_h, 1/(len(s2))) * pow(y,  1/(len(s2)))
        parte_baixo = soma_centroide_j
       # print("DIVIDINDO")
       # print(parte_cima)
       # print(parte_baixo)
        s2_j = parte_cima/parte_baixo
        novo_s2.append(1/s2_j)
       # print("add s2")
        #input()
    return novo_s2
# ------------------    
# ------------------
def inicializa_s2(X):
    # calcula o s2 da base de dados X
    
    distancias=[]
    s2 = []

    for i in range(X.shape[0]-1):
        for j in range(i+1, X.shape[0]):
            distancias.append(distance.euclidean(X[i],X[j]))
            #print(str(i) + str(" - ") + str(j))

    distancias = np.sort(distancias)
    
    result = (distancias[int(len(distancias)*.1)] + distancias[int(len(distancias)*.9)])/2

    for i in range(len(X[0])):
        s2.append(result)
            
    return s2
# ------------------
# ------------------
def get_y(s2):

    y = 1
    print(type(s2))
    for i in range(len(s2)):
        y *= 1/s2[i]

    return y
        
# ------------------
# ------------------
def calcula_k(x_i, x_j, s2):
    # Função (9) de distância do artigo 

    distancia =0.0
    
    for i in range(len(x_i)):
        distancia += pow((x_i[i] - x_j[i]), 2)/s2[i]
        
##    distancia = sum(np.ndarray.tolist(np.divide(pow(x_i - x_j, 2),s2)))
    k = -1 * (distancia)
    k = pow(np.e, k)
    
    return k
# ------------------
# ------------------
def gerar_centroides(X, classes):
    # centroides é uma matriz onde as linhas são os centroides e as colunas os atributos
    # linhas = centroides
    # colunas = atributos
 
##    centroides=[]
##    arange = np.arange(len(X))
##    arange = np.ndarray.tolist(arange)
##    for i in range(classes):
##        idx = random.choice(arange)
##        centroides.append(np.ndarray.tolist(X[idx]))
##        arange.remove(idx)    
##    return centroides
    centroides = []

    maximos = []
    minimos = []
    
    for i in range(len(X[0])):
        maximos.append(max(X[:,i]))
        minimos.append(min(X[:,i]))

    for i in range(classes):
        centroide=[]
        for j in range(len(maximos)):

            centroide.append(randint(int(minimos[j]), int(maximos[j])))

        centroides.append(centroide)
    
    #centroides = np.random.rand(classes, X.shape[1])
    
    return centroides
# ------------------
# ------------------
def calcula_p(X, centroides):
    # X é a base de dados
    # P é uma matriz 3d onde
    # P[i] retorna a i-ésima região
    # P[i][j] retorna a j-ésima amostra da i-ésima região
    # P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região
    # p[i][0] = centroide da i-ésima região
    
    p = {}
    s2 = inicializa_s2(X)

    # Iniciando cada região com seu centroide
    for i in range(len(centroides)):
        p[i] = [centroides[i]]
    
    for i in range(len(X)):
        
        valor_minimo = -1
        indice_valor_minimo = -1

        for j in range(len(centroides)):
            
            distancia = 2*(1 - calcula_k(X[i], centroides[j], s2))

            if (j == 0):
                
               valor_minimo = distancia
               indice_valor_minimo = j
               
            elif (distancia<valor_minimo):
                
                valor_minimo = distancia
                indice_valor_minimo = j

        p[indice_valor_minimo].append(np.ndarray.tolist(X[i]))

    # Retirando o centroide de cada região
    for i in range(len(centroides)):
        p[i].remove(p[i][0])
        
    return p
# ------------------
# ------------------
def normaliza(X):

    media = []
    desvio = []

    for i in range(len(X[0])):
        media.append(np.mean(X[:, i]))
        desvio.append(np.std(X[:, i]))

    for i in range(len(X)):

        for j in range(len(X[0])):
            X[i, j] = (X[i, j] - media[j]) / desvio[j]


    return X
# ------------------
# ------------------
def remove_coluna(X, index):

    part1 = X[:, :index]
    part2 = X[:, index+1:]

    return np.concatenate([part1, part2], axis=1)
# ------------------

# ------------------ Preparando Dados ------------------
shape_view = remove_coluna(shape_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
shape_view = normaliza(shape_view)

rgb_view = normaliza(rgb_view)

complet_view = remove_coluna(complet_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
complet_view = normaliza(complet_view)

# --------------------------------------------------------------



centroides = gerar_centroides(shape_view, 7)
p = calcula_p(shape_view, centroides)
s2 = inicializa_s2(shape_view)
y = get_y(s2)
novo_s2 = atualiza_s2(p, centroides, s2, y)

print(s2)
print(novo_s2)
##centroides = gerar_centroides(shape_view, 7)
##p = calcula_p(shape_view, centroides)
##
##pca = PCA(n_components=2)
##pca = pca.fit(shape_view)
##
##for i in range(len(p)):
##    p_i = pca.transform(p[i])
##    p_a = np.array(p_i)
##    plt.plot(p_a[:,0], p_a[:,1], 'ro', linewidth=0.03)
##    plt.plot(p_a[0][0], p_a[0][1], 'g^', linewidth=0.03)
##    
##plt.show()

