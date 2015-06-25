
import sys
sys.path.append('/volatile/hubert/HCode')
import saver as sav
import pickle
with open('/volatile/hubert/datas/sessionbinary', 'rb') as fichier:
  monpickler = pickle.Unpickler(fichier)
  donnees = monpickler.load()
d = sav.save_session('coefficient de Hurst calcule avec la methode DFA sur donnees non normalisees',
	      donnees[3],donnees[2],donnees[0],donnees[1],'No comments',True)

