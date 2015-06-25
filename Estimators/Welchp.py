import numpy as np
import scipy.optimize as so
from scipy.signal import welch
import nitime.algorithms.spectral as nas
from math import gamma

def Welchp(data):
 

  #datanorm = (data- np.mean(data))/np.std(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,p =  welch(data)
  #f,p = nas.get_spectra(data)
  mask = np.all([(f>0),(f<0.1)],axis =0)
  
  if any(mask):
    tmp = np.polyfit(np.log2(f[mask]), np.log2(np.abs(p[mask])), deg = 1)
    H = (-1-tmp[0])/2
    return H
  else:
    return 0

def WelchpS(data,idx):
 

  #datanorm = (data- np.mean(data))/np.std(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,p =  welch(data[idx,:])

  #f,p = nas.get_spectra(data[idx,:])

  mask = np.all([(f>0.001)],axis =0)
  if any(mask):
    tmp = np.polyfit(np.log2(f[mask]), np.log2(np.abs(p[mask])), deg = 1)
    H = (-1-tmp[0])/2
    ##?? devrait etre H = (1-tmp[0])/2 mais l'estimateur est shifter de 1
    return H
  
  else:
    return 0