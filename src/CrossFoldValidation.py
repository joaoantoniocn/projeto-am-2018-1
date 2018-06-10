import random
import numpy as np
import pickle

class CrossFoldValidation:

    def __init__(self, base, labels, numero_folders):

        self.base = base
        self.labels = labels
        self.numero_folders = numero_folders



    def separar_classe(self):
        separated = {}
        for i in range(len(self.base)):
            vector = self.base[i]
            if (self.labels[i] not in separated):
                separated[self.labels[i]] = []
            separated[self.labels[i]].append(vector)
        return separated


    def gerar_folders(self, path):

        base_separada = self.separar_classe()
        folders = {}            # folders[numero_folder][classe][amostra][atributo]
        amostras_por_classe = int(len(base_separada[0]) / self.numero_folders)


        # folder
        for folder in range(self.numero_folders):

            if (folder not in folders):
                folders[folder] = {}

            # classe
            for classe in range(len(base_separada)):

                if (classe not in folders[folder]):
                    folders[folder][classe] = []

                # amostra
                for i in range(amostras_por_classe):

                    indice = random.randint(0, len(base_separada[classe])-1)
                    amostra = base_separada[classe][indice]

                    # adicionando a amostra no folder
                    folders[folder][classe].append(np.ndarray.tolist(amostra))

                    # removendo a amostra da base separada
                    del base_separada[classe][indice]

        # gravando folder em arquivo
        arq = open(path, 'wb')
        #arq.write(str(folders))
        pickle.dump(folders, arq)
        arq.close()

        return folders