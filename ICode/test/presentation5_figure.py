
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
from nitime.analysis.spectral import SpectralAnalyzer
from nitime.timeseries import TimeSeries
import matplotlib.pyplot as plt
from Estimators.Wavelet.HDW_plagiate import *
from matplotlib.colors import Normalize
f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
## Plotter signal 1D pour H = 0.8
plt.figure()
signal = simulations[7,1]
signalcs = np.cumsum(signal)
plt.plot(signal)
plt.title('original signal')
plt.figure()
plt.plot(signalcs)
plt.title('cumsum signal')

## Plot estimateur Welch du signal
plt.figure()
t1 = TimeSeries(data = signal, sampling_rate = 0.2)
s1 = SpectralAnalyzer(t1)
f,pw = s1.psd
masker = np.all([(f>0),(f<0.1)],axis=0)
tmp = np.polyfit(np.log2(f[masker]),np.log2(pw.T[masker]),deg =1)
plt.plot(np.log2(f),np.log2(pw),'b')
plt.plot(np.log2(f),np.log2(f)*tmp[0] + tmp[1],'r')
plt.xlabel('log2 frequency')
plt.ylabel('log2 pw')
plt.title('Hurst Exponent by Welch Estimator\nSignal length = 4096 \n estimated H = %.2f'%((1-tmp[0])/2.))
#Plot l'estimateur en ondelette
##Une facon de calculer les coefficient de Hurst plus longue mais plus precise
l=4096
nbvoies = int( np.log2(l) )
nbvanishmoment  = 4 
j1=3
j2 =6
wtype =1
nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)

dico = wtspecq_statlog3(signalcs, nbvanishmoment,1,np.array(2),nbvoies,0,0)
sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
estimate=sortie['Zeta']/2. #normalement Zeta
aest = sortie['aest']
Elog = dico['Elogmuqj'][0]
Varlog = dico['Varlogmuqj'][0]
nj = dico['nj']
jj=np.arange(Elog.shape[0])+1
plt.figure()
plt.plot(jj,Elog,'b')
plt.plot(jj,2*estimate *jj +aest,'r')
plt.xlabel('log2 frequency')
plt.ylabel('log2 pw')
plt.title('Hurst Exponent by Wavelet Estimator\nSignal length = 4096 \n estimated H = %.2f'%estimate)

fig, axes = plt.subplots(nrows= 2)
fig.suptitle('Hurst Exponent, Wavelet and Welch Estimator,\nSignal length = 4096')
axes[0].plot(np.log2(f),np.log2(f)*tmp[0] + tmp[1],'r')
axes[0].plot(np.log2(f),np.log2(pw),'b')
axes[0].set_xlabel('log2 frequency')

axes[0].set_title('Hurst Exponent by Welch Estimator\n estimated H = %.2f'%((1-tmp[0])/2.))
axes[1].plot(jj,2*estimate*jj +aest,'r')
axes[1].plot(jj,Elog,'b')
axes[1].set_xlabel('log2 frequency')
axes[1].set_ylabel('scale')
axes[1].set_title('Hurst Exponent by Wavelet Estimator estimated H = %.2f'%estimate)


##Meme chose avec des signaux d'une longueur de 514
plt.figure()
t1 = TimeSeries(data = signal[:514], sampling_rate = 0.20)
s1 = SpectralAnalyzer(t1)
f,pw = s1.psd
masker = np.all([(f>0),(f<0.1)],axis=0)
tmp = np.polyfit(np.log2(f[masker]),np.log2(pw.T[masker]),deg =1)
plt.plot(np.log2(f),np.log2(pw),'b')
plt.plot(np.log2(f),np.log2(f)*tmp[0] + tmp[1],'r')
plt.xlabel('log2 frequency')
plt.ylabel('log2 pw')
plt.title('Hurst Exponent by Welch Estimator\nSignal length = 514 \n estimated H = %.2f'%((1-tmp[0])/2.))


l=514
nbvoies = int( np.log2(l) )
nbvanishmoment  = 4 
j1=3
j2 =6
nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)

dico = wtspecq_statlog3(signalcs[:l], nbvanishmoment,1,np.array(2),nbvoies,0,0)
sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
estimate=sortie['Zeta']/2. #normalement Zeta
aest = sortie['aest']
Elog = dico['Elogmuqj'][0]
Varlog = dico['Varlogmuqj'][0]
nj = dico['nj']
jj=np.arange(Elog.shape[0])+1

plt.figure()
plt.plot(jj,Elog,'b')
plt.plot(jj,2*estimate *jj +aest,'r')
plt.xlabel('log2 frequency')
plt.ylabel('log2 pw')
plt.title('Hurst Exponent by Wavelet Estimator\nSignal length = 514 \n estimated H = %.2f'%estimate)

fig, axes = plt.subplots(nrows= 2)
fig.suptitle('Hurst Exponent, Wavelet and Welch Estimator,\nSignal length = 514')
axes[0].plot(np.log2(f),np.log2(f)*tmp[0] + tmp[1],'r')
axes[0].plot(np.log2(f),np.log2(pw),'b')
axes[0].set_xlabel('log2 frequency')
axes[0].set_ylabel('log2 pw')
axes[0].set_title('Hurst Exponent by Welch Estimator estimated H = %.2f'%((1-tmp[0])/2.))
axes[1].plot(jj,2*estimate*jj +aest,'r')
axes[1].plot(jj,Elog,'b')
axes[1].set_xlabel('log2 frequency')
axes[1].set_ylabel('scale')
axes[1].set_title('Hurst Exponent by Wavelet Estimator estimated H = %.2f'%estimate)


## Deux images
j1=3
j2=6
wtype = 1
l=514
title = '  '
s = opas.square2(10)
#s = opas.smiley()
#mask = s>0.3

signal = opas.get_simulation_from_picture(s,lsimul=l)
signalshape = signal.shape
shape = signalshape[:-1]
sig = np.reshape(signal,(signalshape[0]*signalshape[1],signalshape[2]))
N= sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation=np.cumsum(sig,axis =1 )

nbvoies = int( np.log2(l) )
nbvanishmoment  = 4 
nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)
Bar = ProgressBar(N, 60, 'Work in progress')
Elog = np.zeros((N,nbvoies))
Varlog = np.zeros((N,nbvoies))


for i in np.arange(0,N):
  Bar.update(i)
  dico = wtspecq_statlog3(simulation[i], nbvanishmoment,1,np.array(2),nbvoies,0,0)
  sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
  estimate[i]=sortie['Zeta']/2. #normalement Zeta
  Elog[i] = dico['Elogmuqj'][0]
  Varlog[i] = dico['Varlogmuqj'][0]
  Bar.update(i+1)
nj = dico['nj']


plt.figure()
im= np.reshape(estimate,s.shape)
imnorm = Normalize(vmin = np.min(estimate),vmax = np.max(estimate))
plt.imshow(im,norm = imnorm)
plt.colorbar()
plt.title('Lenght of the simulation %d, rmse = %.3f'%(l, np.sqrt(np.mean((im-s)**2))))




## ## ##
l=2048
signal = opas.get_simulation_from_picture(s,lsimul=l)
signalshape = signal.shape
shape = signalshape[:-1]
sig = np.reshape(signal,(signalshape[0]*signalshape[1],signalshape[2]))
N= sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation=np.cumsum(sig,axis =1 )

nbvoies = int( np.log2(l) )
nbvanishmoment  = 4 
nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)
Bar = ProgressBar(N, 60, 'Work in progress')
Elog = np.zeros((N,nbvoies))
Varlog = np.zeros((N,nbvoies))


for i in np.arange(0,N):
  Bar.update(i)
  dico = wtspecq_statlog3(simulation[i], nbvanishmoment,1,np.array(2),nbvoies,0,0)
  sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
  estimate[i]=sortie['Zeta']/2. #normalement Zeta
  Elog[i] = dico['Elogmuqj'][0]
  Varlog[i] = dico['Varlogmuqj'][0]
  Bar.update(i+1)
nj = dico['nj']

plt.figure()
im= np.reshape(estimate,s.shape)
plt.imshow(im,norm = imnorm)
plt.colorbar()
plt.title('Lenght of the simulation %d, rmse = %.3f'%(l, np.sqrt(np.mean((im-s)**2))))
##


## ## ## ## ##
l=4096
signal = opas.get_simulation_from_picture(s,lsimul=l)
signalshape = signal.shape
shape = signalshape[:-1]
sig = np.reshape(signal,(signalshape[0]*signalshape[1],signalshape[2]))
N= sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation=np.cumsum(sig,axis =1 )

nbvoies = int( np.log2(l) )
nbvanishmoment  = 4 
nbvoies =min( int(np.log2(l/(2*nbvanishmoment+1)))  , nbvoies)
Bar = ProgressBar(N, 60, 'Work in progress')
Elog = np.zeros((N,nbvoies))
Varlog = np.zeros((N,nbvoies))


for i in np.arange(0,N):
  Bar.update(i)
  dico = wtspecq_statlog3(simulation[i], nbvanishmoment,1,np.array(2),nbvoies,0,0)
  sortie = regrespond_det2(dico['Elogmuqj'][0],dico['Varlogmuqj'][0],2,dico['nj'],j1,j2,wtype)
  estimate[i]=sortie['Zeta']/2. #normalement Zeta
  Elog[i] = dico['Elogmuqj'][0]
  Varlog[i] = dico['Varlogmuqj'][0]
  Bar.update(i+1)
nj = dico['nj']
plt.figure()
im= np.reshape(estimate,s.shape)
plt.imshow(im,norm =imnorm)
plt.colorbar()
plt.title('Lenght of the simulation %d, rmse = %.3f'%(l, np.sqrt(np.mean((im-s)**2))))


plt.figure()
plt.imshow(s,norm =imnorm)
plt.colorbar()

plt.show()




