#les charge et les affiche
import sys
import matplotlib.pyplot as plt
sys.path.append('/volatile/hubert/HCode')
import data_class as dat
import pickle
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi
from nilearn.plotting import plot_img, plot_stat_map
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## Ce premier Code est adapte au fichier mysession qui sauvegarde les calculs
##	dans une liste de data_class sont tous les champs sont remplis
#with open('/volatile/hubert/datas/mysession', 'rb') as fichier:
  #monpickler = pickle.Unpickler(fichier)
  #donnees = monpickler.load()


#for data in donnees:
  #p= plot_stat_map(data.image)
  #p.title(data.title)
  #sys.stdout.write(data.title + '\n\t' + data.comments +'\n')
  
#plt.show()
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## Ce second Code est adapte au fichier mysession2 qui est un format un peu
#plus leger de sauvegarde save
#with open('/volatile/hubert/datas/mysession2', 'rb') as myfile:
  #monunpickler = pickle.Unpickler(myfile)
  #on peut aussi utiliser ce fichier
with open('/volatile/hubert/datas/session_reussie', 'rb') as myfile:
  monunpickler = pickle.Unpickler(myfile)
    

  image=list()
  first = True
  while True:
    try:
      data = monunpickler.load()
      sys.stdout.write('dataloaded')
      if first:
	masker = data.masker
	signal = data.signal
	first = False
      
      img = masker.inverse_transform(data.H)
      image.append(img)
      p= plot_stat_map(img)
      p.title(data.title)
      sys.stdout.write(data.title + '\n\t' + data.comments +'\n')
    except(EOFError):
      break
      
plt.show()