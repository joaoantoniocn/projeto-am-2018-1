from CrossFoldValidation import CrossFoldValidation
import numpy as np
from sklearn import preprocessing
import pickle


# ------------------ Load database ---------------
base = np.genfromtxt('../base/segmentation.test',delimiter=',', dtype=np.str)
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
def remove_coluna(X, index):

    part1 = X[:, :index]
    part2 = X[:, index+1:]

    return np.concatenate([part1, part2], axis=1)
# ------------------
# ------------------ Separate database ---------------
labels = base[1:,0]
complet_view = base[1:,1:]
complet_view = complet_view.astype(float)
complet_view = remove_coluna(complet_view, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
complet_view = normaliza(complet_view)

shape_view = complet_view[:,:9]
shape_view = shape_view.astype(float)


rgb_view = complet_view[:,9:]
rgb_view = rgb_view.astype(float)


labels_transform = preprocessing.LabelEncoder()
labels_transform.fit(labels)
labels_n = labels_transform.transform(labels) # Labels numericos
# ------------------


#cross = CrossFoldValidation(shape_view, labels_n, 10)
#folders = cross.gerar_folders('../folders/test.txt')

#indice_validacao = [1, 2]

#treino, teste, validacao = cross.separa_treino_teste(folders, 0, indice_validacao)


for i in range(30):

    cross = CrossFoldValidation(complet_view, labels_n, 10)
    folders = cross.gerar_folders('../folders/complet_view_' + str(i) + '.txt')


