##these functions hab been made according to whittlenew,
# wittlefunc and fspecFGN of Matlab Biyu_code

import numpy as np
import scipy.optimize as so
import nitime.algorithms.spectral as nas
from math import gamma

def Welchnitime(data):
 

  #datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,pw =  nas.get_spectra(data)
  p= abs(pw)

  nhalfm = len(f)
  gammahat =  np.exp(2*np.log(p[1:nhalfm]))/(2*np.pi*((nhalfm)))

  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  
  return so.fminbound(func,0,1)

def WelchnitimeNorm(data):
  datanorm = (data- np.mean(data))/np.std(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,pw =  nas.get_spectra(datanorm)
  p= abs(pw)
  nhalfm = len(f)
  gammahat =  np.exp(2*np.log(p[1:nhalfm]))/(2*np.pi*((nhalfm)))
  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  
  return so.fminbound(func,0,1)

def WelchNitimeS(data,idx):
  #datanorm = (data- np.mean(data))/np.var(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,pw =  nas.get_spectra(data[idx,:])
  p= abs(pw)

  nhalfm = len(f)
  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  return so.fminbound(func,0,1)

def WelchNitimeNormS(data,idx):
  datanorm = (data- np.mean(data[idx,:]))/np.std(data[idx,:])
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,pw =  nas.get_spectra(datanorm)
  p= abs(pw)

  nhalfm = len(f)
  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  return so.fminbound(func,0,1)

def WelchNitimeNormT(data,pointeur,idx):
  datanorm = (data- np.mean(data))/np.std(data)
  #cette ligne provient de la version matlab du code,
  #faut-il effectuer la transformation de fourier sur des donnees 'normalisees' ?
  f,pw =  nas.get_spectra(datanorm)
  p= abs(pw)

  nhalfm = len(f)
  func = lambda H : whittlefunc(H, p[1:nhalfm], 2*(nhalfm))

  pointeur[idx]=so.fminbound(func,0,10)


def whittlefunc(H,gammahat,nbpoints):
  gammatheo = fspecFGN(H, nbpoints)
  yf = gammahat/gammatheo
  return 2*(2*np.pi/nbpoints)*np.sum(yf)

def fspecFGN(hest, nbpoints):
  #Cette fonction semble bien equivalente a son homonyme en matlab
  #procedure de calcul de la densite spectrale d'un fGn
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