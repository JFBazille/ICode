#! /usr/bin/python3.4
#fast has been implemented to test Hurstexp_Welchper
#	(Hurst exponent estmator using Wlech periodogramme)
#	on real datas
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
from Estimators.Wavelet.HDW_plagiate import *
from ProgressBar import ProgressBar
##Chargement des donnees prend du temps a voir si besoin
dataset2 = load_dynacomp(preprocessing_folder ='pipeline_1', prefix= 'swr')

print 'chargement des donnees' 
#on recupere notre signal et le maps_img qui va bien
[x, masker] = extract_one_vpv_signal(dataset2)
Bar = ProgressBar(x.shape[1], 60, 'Work in progress')
H = np.zeros(x.shape[1])

for j in np.arange(0,x.shape[1]):
  H[j] = HDW_p(x[:,j],2,1)
  Bar.update(j)


print 'min value    '+str(np.min(H))+'\n'
print 'max value    '+str(np.max(H))+'\n' 
##la progressBar


#on cree l'image qui va bien avec la fonction qui va bien
img8 = masker.inverse_transform(H)
#on plug le tout dans une fonction qui plot
plt.figure(1)
plot_stat_map(img8)

plt.show()