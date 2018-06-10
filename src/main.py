from CrossFoldValidation import CrossFoldValidation
import numpy as np
from sklearn import preprocessing
import pickle


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

labels_transform = preprocessing.LabelEncoder()
labels_transform.fit(labels)
labels_n = labels_transform.transform(labels) # Labels numericos
# ------------------



for i in range(30):

    cross = CrossFoldValidation(shape_view, labels_n, 10)
    folders = cross.gerar_folders('../folders/shape_view_' + str(i) + '.txt')


# como ler os arquivos gerados
#arq = open('../folders/test.txt','rb') #abrir o arquivo para leitura - o "b" significa que o arquivo é binário
#dic = pickle.load(arq)#Ler a stream a partir do arquivo e reconstroi o objeto original.
#arq.close()