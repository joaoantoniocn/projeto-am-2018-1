import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from random import randint
from sklearn import preprocessing
import time

# ------------------ Load database ---------------
print("carregando base de dados")
base = np.genfromtxt('../base/segmentation.test',delimiter=',', dtype=np.str)
print("base de dados carregada")
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
# ------------------
def funcao_objetivo(p, centroides, s2):
    # define quão boa foi nossa solução
    # quanto menor a função objetivo melhor a solução
    
    result = 0

    for i in range(len(centroides)):
        for k in range(len(p[i])):
            result += 2*(1 - calcula_k(p[i][k], centroides[i], s2))
     

    return result
# ------------------
# ------------------
def atualiza_s2_somatorio(p, centroides, s2, caracteristica):

    result = 0

    for i in range(len(centroides)):
        for k in range(len(p[i])):
            result += (calcula_k(p[i][k], centroides[i], s2) * pow((p[i][k][caracteristica] - centroides[i][caracteristica]), 2))
            
    return result
# ------------------
# ------------------
def atualiza_s2(p, centroides, s2,y):
# calcula o novo vetor de hyper-parametros s2

    novo_s2=[]

    for j in range(len(s2)):
    # calculando o j-ésimo s2
        s2_j = 0
        parte_baixo = atualiza_s2_somatorio(p, centroides, s2, j)
        produtorio = 1
        
        for h in range(len(s2)):
            produtorio *= atualiza_s2_somatorio(p, centroides, s2, h)   
        
        parte_cima = pow(y, (1/len(s2))) * pow(produtorio, (1/len(s2)))

        if((parte_baixo != 0) and (parte_cima!= 0)):         
            s2_j = 1/(parte_cima/parte_baixo)
        else:
            s2_j = s2[j]
            
        novo_s2.append(s2_j)
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
    k = (-1/2) * (distancia)
    k = pow(np.e, k)
    
    return k
# ------------------
# ------------------
def gerar_centroides(X, classes):
    # gerar_centroides é usado para inicializar os centroides de forma aleatória
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

            centroide.append(random.uniform(int(minimos[j]), int(maximos[j])))

        centroides.append(centroide)

    #centroides = np.random.rand(classes, X.shape[1])
    
    return centroides
# ------------------
# ------------------
def atualiza_centroides(p, antigos_centroides, s2):
# atualizar_centroides vai calcular a nova posição dos centroides em relação a sua região p[i]
# p = matriz 3d onde
# P[i] retorna a i-ésima região
# P[i][j] retorna a j-ésima amostra da i-ésima região
# P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região

    centroides = []

    for i in range(len(antigos_centroides)):

        g_i = antigos_centroides[i]

        parte_cima = 0
        parte_baixo = 0

        for k in range(len(p[i])):
            x_k = p[i][k]
            distancia = calcula_k(x_k, g_i, s2)
            parte_cima += distancia * np.array(x_k)
            parte_baixo += distancia


        # esse if não está no artigo, mas foi uma adaptação que achei pra resolver o caso em que a partição P não tem amostras suficientes para atualizar a posição do seu centroide
        if(parte_baixo == 0):
            centroides.append(g_i)
        else:
            centroides.append(np.ndarray.tolist(parte_cima/parte_baixo))



    return centroides
# ------------------
# ------------------
def calcula_p(X, centroides, s2):
    # X é a base de dados
        # P é uma matriz 3d onde
        # P[i] retorna a i-ésima região
        # P[i][j] retorna a j-ésima amostra da i-ésima região
        # P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região

    
    p = {}

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
def atualiza_p(p, centroides, s2):
    # atualiza_p atualiza as regiões p de acordo com as novas posições de cada centroide
    # o retorno dessa função é a nova matriz de regiões p e a quantidade de vezes que ela foi atualizada em comparação a antiga matriz p

    # P é uma matriz 3d onde
    # P[i] retorna a i-ésima região
    # P[i][j] retorna a j-ésima amostra da i-ésima região
    # P[i][j][k] retorna a k-ésima característica da j-ésima amostra da i-ésima região

    p_novo = {}
    atualizacoes = 0

    # Iniciando cada região com seu centroide
    for i in range(len(centroides)):
        p_novo[i] = [centroides[i]]

    # todas as regiões
    for i in range(len(p)):

        # todos as amostras de cada região
        for j in range(len(p[i])):

            valor_minimo = -1
            indice_valor_minimo = -1

            # calcula a distancia da amostra com todos os novos centroides
            for k in range(len(centroides)):

                distancia = 2*(1 - calcula_k(p[i][j], centroides[k], s2))

                if (k == 0):

                    valor_minimo = distancia
                    indice_valor_minimo = k

                elif (distancia < valor_minimo):

                    valor_minimo = distancia
                    indice_valor_minimo = k

            if(indice_valor_minimo != i):
                atualizacoes += 1

            p_novo[indice_valor_minimo].append(p[i][j])


    # Retirando o centroide de cada região
    for i in range(len(centroides)):
        p_novo[i].remove(p_novo[i][0])

    return p_novo, atualizacoes
# ------------------
# ------------------
def imprime_p(p):

    texto = "Quantidade de elementos por grupo"
    #print(texto)

    for i in range(len(p)):
        #print("Particao ", i, ": ", len(p[i]))
        texto += "\n Partição " + str(i) + ": " + str(len(p[i]))

    return texto
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
# ------------------
def inicializa_modelo(base, numero_classes):
    centroides = gerar_centroides(base, numero_classes)
    s2 = inicializa_s2(base)
    y = get_y(s2)
    p = calcula_p(base, centroides, s2)

    return centroides, s2, y, p
# ------------------
# ------------------
def treinar_modelo(base, numero_classes, numero_holdouts):

    centroides_result = []
    p_result = {}
    s2_result = []
    objetivo_result = 0
    tempo = [] # array com o tempo de execução em segundos de cada holdout

    # Holdouts
    for i in range(numero_holdouts):

        # contando tempo em segundos
        inicio = time.time()
        # --- Inicializando ---
        centroides, s2, y, p = inicializa_modelo(base, numero_classes)

        # --- Treinando ---
        atualizacoes = 1

        while (atualizacoes > 0):
            centroides = atualiza_centroides(p, centroides, s2)
            s2 = atualiza_s2(p, centroides, s2, y)
            p, atualizacoes = atualiza_p(p, centroides, s2)
            objetivo = funcao_objetivo(p, centroides, s2)
        # ------------

        if (i == 0):
            centroides_result = centroides
            p_result = p
            s2_result = s2
            objetivo_result = objetivo
        elif ( objetivo < objetivo_result):
            centroides_result = centroides
            p_result = p
            s2_result = s2
            objetivo_result = objetivo

        final = time.time()
        tempo.append(final-inicio)

    return centroides_result, p_result, s2_result, objetivo_result, tempo

# ------------------
# ------------------
def combinacao(n):
    return (n*(n-1))/2

def ari(p, base, labels_n):
    result = 0

    base = np.ndarray.tolist(base)
     # exemplo de como pegar a classe

    contingency_table = np.zeros([len(p), len(p)])

    for cluster in range(len(p)):
        for amostra in range(len(p[cluster])):
            classe = labels_n[base.index(p[cluster][amostra])]
            contingency_table[classe][cluster] += 1

    index = 0
    expected_index = 0
    max_index = 0
    
    for i in range(len(contingency_table)):
        for j in range(len(contingency_table)):
            index += combinacao(contingency_table[i][j]) 

    linha = 0
    coluna = 0
    for i in range(len(contingency_table)):
        linha += combinacao(sum(contingency_table[i]))
        coluna += combinacao(sum(contingency_table[:, i]))

    expected_index = (linha * coluna)/ combinacao(len(base))
    max_index = (linha + coluna) / 2

    result = (index - expected_index)/(max_index - expected_index)
    
    return result
# ------------------
# ------------------
def get_lista_objetos(p, base):
    lista = {}
    base = np.ndarray.tolist(base)

    for i in range(len(p)):
        lista[i] = []
        for j in range(len(p[i])):
            lista[i].append(base.index(p[i][j]))


    return lista

# ------------------
# ------------------
def run(base, path, holdouts):
# treina os modelos e escreve o resultado no arquivo com endereço indicado na variavel 'path'

    # Treinando
    centroides_result, p_result, s2_result, objetivo_result, tempo_execucao = treinar_modelo(base, 7, holdouts)

    # Escrevendo Resultados
    arq = open(path,  'w')
    arq.write("centroides: " + str(centroides_result) + "\n \n")
    arq.write(imprime_p(p_result))
    arq.write("\n \n hyperparametros: " + str(s2_result) + "\n \n")
    arq.write("Lista de objetos por grupo: " + str(get_lista_objetos(p_result, base)) + "\n \n")
    arq.write("Função Objetivo: " + str(objetivo_result) + "\n \n")
    arq.write("Tempo de Execução: " + str(tempo_execucao) + "\n \n")
    arq.write("Indice de Rand Corrigido: " + str(ari(p_result, base, labels_n)) + "\n")
    arq.close()

# ------------------
# ------------------ Preparando Dados ------------------
#shape_view = remove_coluna(shape_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
#print("normalizando shape view")
#shape_view = normaliza(shape_view)

#print("normalizando rgb_view")
#rgb_view = normaliza(rgb_view)

complet_view = remove_coluna(complet_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
#print("normalizando complet view")
complet_view = normaliza(complet_view)

labels_transform = preprocessing.LabelEncoder()
labels_transform.fit(labels)
labels_n = labels_transform.transform(labels) # Labels numericos
# ------------------- Treinando -------------------------

holdouts = 100

#print("Começando a rodar a base Shape_View")
#run(shape_view, '../resultados/shape_view_nao_normalizado2.txt', holdouts)
#print("Finalizado Shape_view")

#print("Começando a rodar a base RGB_View")
#run(rgb_view, '../resultados/rgb_view_nao_normalizado2.txt',holdouts)
#print("Finalizado RGB_View")

print("Começando a rodar a base complet view")
run(complet_view, '../resultados/complet_view2.txt', holdouts)
print("Finalizado Complet_view")