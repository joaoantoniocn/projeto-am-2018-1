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




def calculateS2(X):

    distancias=[]

    for i in range(X.shape[0]-1):
        for j in range(i+1, X.shape[0]):
            distancias.append(distance.euclidean(X[i],X[j]))
            #print(str(i) + str(" - ") + str(j))

    distancias = np.sort(distancias)
    
    s2 = (distancias[int(len(distancias)*.1)] + distancias[int(len(distancias)*.9)])/2
    
    return s2


s2 = calculateS2(shape_view)

       
