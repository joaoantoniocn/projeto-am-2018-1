import numpy as np
import pandas as pd


base = np.genfromtxt('segmentation.data.txt',delimiter=',', dtype=np.str)


labels = base[1:,0]
complet_view = base[1:,1:]

shape_view = complet_view[:,:9]

rgb_view =  complet_view[:,9:]
