##these functions hab been made according to whittlenew,
# wittlefunc and fspecFGN of Matlab Biyu_code

import numpy as np
import scipy.optimize as so
import math

def Whittle(data):
  nbpoints = len(data)
  #datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees'?
  nhalfm = int((nbpoints - 1)/2)
  tmp =  np.abs(np.fft.fft(data))
  gammahat =  np.exp(2*np.log(tmp[1:nhalfm+1]))/(2*np.pi*nbpoints)

  func = lambda H : whittlefunc(H, gammahat, nbpoints)
  ##probleme de division par zero rencontre lors de la minimisation
  #pour la fonction scipy.optimize.minimize_scalar()
  
  return so.fminbound(func,0,1)

def Whittlet(data,pointeur,idx):
  nbpoints = len(data)
  #datanorm = (data- np.mean(data))/np.var(data)
  nhalfm = int((nbpoints - 1)/2)
  tmp =  np.abs(np.fft.fft(data))
  gammahat =  np.exp(2*np.log(tmp[1:nhalfm+1]))/(2*np.pi*nbpoints)

  func = lambda H : whittlefunc(H, gammahat, nbpoints)
  ##probleme de division par zero rencontre lors de la minimisation
  #pour la fonction scipy.optimize.minimize_scalar()
  
  pointeur[idx] = so.fminbound(func,0,1)

def WhittleS(data,idx):
  x = data[idx,:]
  nbpoints = len(x)
  #datanorm = (data- np.mean(data))/np.var(x)
  nhalfm = int((nbpoints - 1)/2)
  tmp =  np.abs(np.fft.fft(x))
  gammahat =  np.exp(2*np.log(tmp[1:nhalfm+1]))/(2*np.pi*nbpoints)

  func = lambda H : whittlefunc(H, gammahat, nbpoints)
  ##probleme de division par zero rencontre lors de la minimisation
  #pour la fonction scipy.optimize.minimize_scalar()
  
  return so.fminbound(func,0,1)

def WhittleNormS(data,idx):
  x = data[idx,:]
  nbpoints = len(x)
  datanorm = (x- np.mean(x))/np.var(x)
  nhalfm = int((nbpoints - 1)/2)
  tmp =  np.abs(np.fft.fft(datanorm))
  gammahat =  np.exp(2*np.log(tmp[1:nhalfm+1]))/(2*np.pi*nbpoints)

  func = lambda H : whittlefunc(H, gammahat, nbpoints)
  ##probleme de division par zero rencontre lors de la minimisation
  #pour la fonction scipy.optimize.minimize_scalar()
  
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
  const = np.sin(np.pi*hest)*math.gamma(-hhest)/np.pi
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