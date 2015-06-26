from nilearn.region import signals_to_img_maps
from nilearn.plotting import plot_roi
from nilearn.plotting import plot_img, plot_stat_map
import sys, os, time
import numpy as np

from ICode.loader import load_dynacomp, dict_to_list
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker
##Le module Thread ne permet pas de faire un parallelisme veritable
#	preferer le module multiprocessing
#from threading import Thread
from multiprocessing import Process

##Chargement des donnees prend du temps a voir si besoin
dataset = load_dynacomp(preprocessing_folder ='pipeline_2', prefix= 'resampled_wr')

print 'chargement des donnees' 
#on recupere notre signal et le maps_img qui va bien
[x, masker] = extract_one_vpv_signal(dataset)


H = np.zeros(x.shape[1])
##pour Whittle
#whittlei = np.zeros(x.shape[1])

#thread_list = list()
process_list = list()
##Avant d'essayer cette fonction matriciellement on la test en utilisant
#les boucles
#Cette loop calcule les exposant de hurst du signal par deux methodes
print 'lancement des thread'
for j in np.arange(0,x.shape[1]):
  ##pour DFA
  #p = Process(target = DFAt, args =(x[:,j],dfai,j,3,6))
  #p.start()
  #process_list.append(p)
  ##pour Whittle
  #p = Process(target = Whittlet, args = (x[:,j],whittlei,j))
  #p.start()
  #process_list.append(p)
  p = Thread(target = WelchNitimeNormT, args =(x[:,j],H,j))
  p.start()
  process_list.append(p)
  #pour Whittle
  #t = Thread(target = Whittlet, args = (x[:,j],whittlei,j))
  #t.start()
  #thread_list.append(p)
print 'retour thread'
#on refait une boucle pour attendre les threads
for p in process_list:
  p.join()

print 'fin de retour thread'
#on cree l'image qui va bien avec la fonction qui va bien
img4 = masker.inverse_transform(dfai)
#on plug le tout dans une fonction qui plot
plt.figure(1)
plot_stat_map(img4)

##pour Whittle
#img3 = masker.inverse_transform(whittlei)
#plt.figure(2)
#plot_stat_map(img3)


plt.show()