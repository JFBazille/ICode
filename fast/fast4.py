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

from Estimators.Hexp_Welchp import Hurstexp_Welchper_2 as HW
#from Estimators.Hexp_Welchp import Hurstexp_Welchper_scipy as HWs


##Chargement des donnees prend du temps a voir si besoin
dataset2 = load_dynacomp(preprocessing_folder ='pipeline_1', prefix= 'swr')

print 'chargement des donnees' 
#on recupere notre signal et le maps_img qui va bien
[x2, masker2] = extract_one_vpv_signal(dataset2)


H2 = HW(x2.T, f_max =0.023)
print 'min value    '+str(np.min(H2))+'\n'
print 'max value    '+str(np.max(H2))+'\n' 
##la progressBar

##Avant d'essayer cette fonction matriciellement on la test en utilisant
#les boucles
#Cette loop calcule les exposant de hurst du signal par deux methodes


#on cree l'image qui va bien avec la fonction qui va bien
img8 = masker2.inverse_transform(H2)
#on plug le tout dans une fonction qui plot
plt.figure(1)
plot_stat_map(img8)




plt.show()