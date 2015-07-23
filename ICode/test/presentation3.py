##Present the differences between matlab and Python wavelet estimators tested on same simulations

import scipy.io as scio
import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('resultat_test_estimators','rb') as fichier:
  unpickler = pickle.Unpickler(fichier)
  donnees = unpickler.load()

pwave_514 = donnees['Wavelet_514']
pwave_4096 = donnees['Wavelet_4096']


mdata = scio.loadmat('matlab_wavelet_estimations.mat')
mwave_4096 = mdata['matlab_wavelet_4096']
mwave_514 = mdata['matlab_wavelet_514']


i=0
k=0
      
fig = plt.figure(i)
f, myplotswavelet = plt.subplots(1,2,sharey=True)
myplotswavelet[0].set_title('matlab function')
myplotswavelet[1].set_title('python function')
f.suptitle('Estimation of Hurst coefficient of fGn\nof length 514 by Wavelet method\n comparison Matlab on Python function')

bp = myplotswavelet[0].boxplot(mwave_514.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswavelet[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mwave_514[k,:]),
					  np.var(mwave_514[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswavelet[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mwave_514[k,:]),
					  np.var(mwave_514[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
bp = myplotswavelet[1].boxplot(pwave_514.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswavelet[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(pwave_514[k,:]),
					  np.var(pwave_514[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswavelet[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(pwave_514[k,:]),
					  np.var(pwave_514[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
i=i+1

     
fig = plt.figure(i)
f, myplotswavelet = plt.subplots(1,2,sharey=True)
myplotswavelet[0].set_title('matlab function')
myplotswavelet[1].set_title('python function')
f.suptitle('Estimation of Hurst coefficient of fGn\nof length 4096 by Wavelet method\n comparison Matlab on Python function')

bp = myplotswavelet[0].boxplot(mwave_4096.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswavelet[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mwave_4096[k,:]),
					  np.var(mwave_4096[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswavelet[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mwave_4096[k,:]),
					  np.var(mwave_4096[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
bp = myplotswavelet[1].boxplot(pwave_4096.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswavelet[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(pwave_4096[k,:]),
					  np.var(pwave_4096[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswavelet[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(pwave_4096[k,:]),
					  np.var(pwave_4096[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
i=i+1      