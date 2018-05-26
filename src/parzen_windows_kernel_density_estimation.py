import numpy as np
import operator
from sklearn.preprocessing import LabelEncoder

 
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
 



def parzen(x_samples, h=1, center=[0,0,0]):
    x_samples = separateByClass(x_samples, y )   
    p_x=[]
    p_xx = {}
    dimensions = len(x_samples[0][0])
     
    for c in range(len(x_samples)):
        k = 0
         
        #dist=[]
         
        for x in range(1,len(x_samples[c])):
            is_inside = 1
            #sample_dist=[]
            for axis,center_point in zip(x_samples[c][x], center):
                if np.abs(axis-center_point) > (h/2):
                    is_inside = 0
                #sample_dist.append(np.abs(axis-center_point))
            #dist.append(sample_dist)
            k += is_inside 
        p_x.append((k / len(x_samples) / (h**dimensions)))
        
    return p_x#,p_xx


 

# ------------------ Load database ---------------
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
#complet_view = normaliza(complet_view)

complet_view_test = remove_coluna(complet_view_test, 2) # Removendo coluna 2, os valores dessa coluna são os mesmos para todas as amostras
#complet_view_test = normaliza(complet_view_test)





encoder = LabelEncoder() #Codifica as classes (String) em valores inteiros.
encoder.fit(labels)

y = encoder.transform(labels)

encoder.fit(labels_test)
y_test = encoder.transform(labels_test)


 
X = rgb_view
X_test = rgb_view_test

 
idx = np.arange(len(X_test))
np.random.shuffle(idx)
acc = 0
for example_idx in idx:
    
    p_x = parzen(X, h=1, center=X_test[example_idx])
    if(p_x.index(max(p_x)) == y_test[example_idx]):
       acc+=1
    
    #print(p_x.index(max(p_x)), y_test[example_idx])

print("Accuracy:", acc/len(y_test)*100, "%")

 
