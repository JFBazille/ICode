#! /usr/bin/python3.4
import matplotlib.pyplot as plt
from nilearn.region import signals_to_img_maps
from nilearn.plotting import plot_roi
from nilearn.plotting import plot_img, plot_stat_map
import sys, os, time
import numpy as np
sys.path.append('/volatile/hubert/Code/post_learning_analysis')
sys.path.append('/volatile/hubert/HCode')
import time
from loader import load_dynacomp, dict_to_list
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker
from extract_signals.extract import extract_one_vpv_signal
from ProgressBar import ProgressBar
from Estimators.welchnitime import WelchNitimeNormT
execfile('/volatile/hubert/HCode/Estimators/DFA.py')
#execfile('/volatile/hubert/HCode/extract_signals.py')



##Chargement des donnees prend du temps a voir si besoin
dataset2 = load_dynacomp(preprocessing_folder ='pipeline_1', prefix= 'swr')

print 'chargement des donnees' 
#on recupere notre signal et le maps_img qui va bien
[x2, masker2] = extract_one_vpv_signal(dataset2)


H2 = np.zeros(x2.shape[1])
##la progressBar
Bar = ProgressBar(x2.shape[1], 60, 'Work in progress')
##Avant d'essayer cette fonction matriciellement on la test en utilisant
#les boucles
#Cette loop calcule les exposant de hurst du signal par deux methodes
print 'lancement'
for j in np.arange(0,x2.shape[1]):
  DFANormt(x2[:,j],H2,j,CS = 0)
  Bar.update(j)

print 'fin boucle'

#on cree l'image qui va bien avec la fonction qui va bien
img8 = masker.inverse_transform(H2)
#on plug le tout dans une fonction qui plot
plt.figure(1)
plot_stat_map(img8)




plt.show()