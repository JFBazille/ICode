##these functions hab been made according to whittlenew,
# wittlefunc and fspecFGN of Matlab Biyu_code
# It is a maximum likelyhood estimator using a welch periodogram

import numpy as np
import scipy.optimize as so
from scipy.signal import welch
from math import gamma
def Welch(data):
 

  #datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,p =  welch(data)


  nhalfm = len(f)
  gammahat =  np.exp(2*np.log(p[1:nhalfm]))/(2*np.pi*((nhalfm)))

  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  
  return so.fminbound(func,0,10)
  ##probleme de division par zero rencontre lors de la minimisation
  #pour la fonction scipy.optimize.minimize_scalar()
  

def WelchS(data,idx):
  #datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
 
  f,p =  welch(data[idx,:])



  nhalfm = len(f)


  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  
  return so.fminbound(func,0,1)

def WelchNormS(data,idx):
  datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
 
  f,p =  welch(datanorm[idx,:])



  nhalfm = len(f)


  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  
  return so.fminbound(func,0,1)

def whittlefunc(H,gammahat,nbpoints):
  gammatheo = fspecFGN(H, nbpoints)
  yf = gammahat/gammatheo
  return 2*(2*np.pi/nbpoints)*np.sum(yf)

def fspecFGN(hest, nbpoints):
  #Cette fonction semble bien equivalente a son homonyme en matlab
  #procedure de calcul de la densite spectrale
  ##essayer d'en savoir plus sur cette procedure

  hhest=-((2*hest)+1)
  const = np.sin(np.pi*hest)*gamma(-hhest)/np.pi
  nhalfm=int((nbpoints-1)/2)
  l=2*np.pi*np.arange(1,nhalfm+1)/nbpoints
  fspec = np.ones(nhalfm)
  for i in np.arange(0,nhalfm):
    fi=np.arange(0,200)
    fi=2*np.pi*fi;
    gi=(np.abs(l[i]+fi))**hhest
    hi=(np.abs(l[i]-fi))**hhest
    fi=gi+hi
    fi[0]=fi[0]/2
    fi=(1-np.cos(l[i]))*const*fi
    fspec[i]=np.sum(fi)


  fspec=fspec/np.exp(2*np.sum(np.log(fspec))/nbpoints)
  return fspec