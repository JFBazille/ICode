#this function doesn't work but still it can be kept and fixed later

import numpy as np
from scipy import signal

def Hurst_by_Wt(data):
  #Pour 1 et 31 je ne sais pas on va essayer comme ca apres on essaiera de comprendre
  widths = np.arange(1,31)
  #le top ce serait de pouvoir rentrer des tableaux 2D
  #mais je ne suis pas sur que cela soit possible,2.**(-widths))* on verra
  cwtmatr = signal.cwt(data,signal.ricker, widths)
  #on moyenne notre cwtmatr sur la dimension temporelle il faut que ce soit la derniere
  cwtmatrmean = np.mean(cwtmatr, axis =1)
  #plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
  #         vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

  H = np.mean((np.log(np.abs(cwtmatrmean)) - np.log(np.abs(cwtmatrmean[10]))) / widths) -0.5
  #on prie pour que tout marche
  return {'H':H, 'widths':widths,'cwtmatrmean':cwtmatrmean,'cwtmatr':cwtmatr}
  
def Hurst_by_Wt2(data):
  #Pour 1 et 31 je ne sais pas on va essayer comme ca apres on essaiera de comprendre
  widths = np.arange(1,31)
  #le top ce serait de pouvoir rentrer des tableaux 2D
  #mais je ne suis pas sur que cela soit possible, on verra
  cwtmatr = signal.cwt(data,signal.ricker, widths)
  #on moyenne notre cwtmatr sur la dimension temporelle il faut que ce soit la derniere
  cwtmatrmean = np.mean(cwtmatr, axis =1)
  #plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
  #         vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
  
  tmp = np.polyfit((np.log(np.abs(cwtmatrmean[6:]))),widths[6:],deg =1)
  H = tmp[0]
  #on prie pour que tout marche
  return {'H':H, 'widths':widths,'cwtmatrmean':cwtmatrmean,'cwtmatr':cwtmatr}
  
def Hurst_by_Wt3(data):
  #Pour 1 et 31 je ne sais pas on va essayer comme ca apres on essaiera de comprendre
  widths = np.arange(1,31)
  #le top ce serait de pouvoir rentrer des tableaux 2D
  #mais je ne suis pas sur que cela soit possible, on verra
  cwtmatr = signal.cwt(data,signal.ricker, widths)
  #on moyenne notre cwtmatr sur la dimension temporelle il faut que ce soit la derniere
  A = np.sum(cwtmatr**2, axis =1)
  F = np.sqrt(np.cumsum(A)) 
  #plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
  #         vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
  

  tmp = np.polyfit(np.log(F/F[5]),np.log(widths),deg =1)
  H = tmp[0]
  #on prie pour que tout marche
  return {'H':H, 'widths':widths,'F':F,'A':A}

def Hurst_by_Wt4(data):
  #Pour 1 et 31 je ne sais pas on va essayer comme ca apres on essaiera de comprendre
  J= 2**np.arange(3,6)
  #le top ce serait de pouvoir rentrer des tableaux 2D
  #mais je ne suis pas sur que cela soit possible, on verra
  cwtmatr = signal.cwt(data,signal.ricker, J)
  #on moyenne notre cwtmatr sur la dimension temporelle il faut que ce soit la derniere
  S = np.mean(cwtmatr**2, axis =1)

  #plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
  #         vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
  

  tmp = np.polyfit(np.log2(S),np.arange(3,6),deg =1)
  H = tmp[0]
  #on prie pour que tout marche
  return {'H':H, 'J':J,'S':S}

def Hurst_by_Wt5(data):
  #Pour 1 et 31 je ne sais pas on va essayer comme ca apres on essaiera de comprendre
  widths = np.arange(1,31)
  #le top ce serait de pouvoir rentrer des tableaux 2D
  #mais je ne suis pas sur que cela soit possible, on verra
  cwtmatr = signal.cwt(data,signal.ricker, widths)
  #on moyenne notre cwtmatr sur la dimension temporelle il faut que ce soit la derniere
  S = np.mean(cwtmatr**2, axis =1)

  #plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
  #         vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
  

  tmp = np.polyfit(np.log2(S),widths,deg =1)
  H = tmp[0]
  #on prie pour que tout marche
  return {'H':H, 'widths':widths,'S':S}