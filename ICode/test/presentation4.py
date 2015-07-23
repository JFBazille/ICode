import scipy.io as scio
import pickle
import numpy as np
import matplotlib.pyplot as plt

mdata = scio.loadmat('matlab_estimations_4096.mat')
mdfal = mdata['dfa']
mwhittlel = mdata['whittle']

mdata = scio.loadmat('matlab_estimations_514.mat')
mdfas = mdata['dfa']
mwhittles = mdata['whittle']

with open('resultat_test_estimators','rb') as fichier:
  unpickler = pickle.Unpickler(fichier)
  donnees = unpickler.load()
  
welch_514 = donnees['Welch_514']
welch_4096 = donnees['Welch_4096']
dfa_514 = welch_514
dfa_4096 = welch_4096
whittle_514 = welch_514
whittle_4096 = welch_4096
with open('ICode/Results/resultat_test_dfa','rb') as fichier:
  unpickler = pickle.Unpickler(fichier)
  donnees = unpickler.load()

dfa_514 = donnees['dfa_514']
dfa_4096 = donnees['dfa_4096']

#with open('/volatile/hubert/HCode/Results/resultat_test','rb') as fichier:
  #unpickler = pickle.Unpickler(fichier)
  #donnees = unpickler.load()


#whittle_514 = donnees['whittle_514']
#whittle_4096 = donnees['whittle_4096']

i=0
k=0
whelchfig = -1


fig = plt.figure(i)
f, myplots = plt.subplots(1,2,sharey=True)
myplots[0].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 4096  by Whittle method \n Matlab estimator')
myplots[1].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 514  by Whittle method \n Matlab estimator')

bp = myplots[0].boxplot(mwhittlel.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittlel[k,:]),
					  np.var(mwhittlel[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittlel[k,:]),
					  np.var(mwhittlel[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
bp = myplots[1].boxplot(mwhittles.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittles[k,:]),
					  np.var(mwhittles[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittles[k,:]),
					  np.var(mwhittles[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
i=i+1




fig = plt.figure(i)
f, myplots = plt.subplots(1,2,sharey=True)
myplots[0].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 514  by DFA method \nPython estimator')
myplots[1].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 514  by DFA method\nMatlab estimator')
plt.title('Estimation of Hurst\ncoeffician of fGn\nof length 514  by DFA method')
bp = myplots[0].boxplot(dfa_514.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(dfa_514[k,:]),
					  np.var(dfa_514[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(dfa_514[k,:]),
					  np.var(dfa_514[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
bp = myplots[1].boxplot(mdfas.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mdfas[k,:]),
					  np.var(mdfas[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mdfas[k,:]),
					  np.var(mdfas[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
i=i+1
      

fig = plt.figure(i)
f, myplots = plt.subplots(1,2,sharey=True)
myplots[0].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 4096  by DFA method \n Python estimator')
myplots[1].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 4096  by DFA method \n Matlab estimator')
plt.title('Estimation of Hurst\ncoeffician of fGn\nof length 4096  by DFA method')
bp = myplots[0].boxplot(dfa_4096.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(dfa_4096[k,:]),
					  np.var(dfa_4096[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(dfa_4096[k,:]),
					  np.var(dfa_4096[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
bp = myplots[1].boxplot(mdfal.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:# get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplots[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mdfal[k,:]),
					  np.var(mdfal[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplots[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mdfal[k,:]),
					  np.var(mdfal[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
i=i+1


fig = plt.figure(i)

f, myplotswelch = plt.subplots(1,2,sharey=True)
#plt.title('Estimation of Hurst\ncoeffician of fGn\nof length 514 and 4096  by Whelch method')
myplotswelch[0].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 514  by Whelch method')
myplotswelch[1].set_title('Estimation of Hurst\ncoeffician of fGn\nof length 4096  by Whelch method')


bp = myplotswelch[0].boxplot(welch_514.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:
  # get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswelch[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(welch_514[k,:]),
					  np.var(welch_514[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswelch[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(welch_514[k,:]),
					  np.var(welch_514[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0
  

bp = myplotswelch[1].boxplot(welch_4096.T, labels=np.arange(1,10)/10.)
for line in bp['medians']:
  # get position data for median line
  x, y = line.get_xydata()[1] # top of median line
  # overlay median value
  if(k <6):
    myplotswelch[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(welch_4096[k,:]),
					  np.var(welch_4096[k,:])),
    horizontalalignment='center') # draw above, centered
  else:
    myplotswelch[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(welch_4096[k,:]),
					  np.var(welch_4096[k,:])),
	horizontalalignment='center') # draw above, centered
  k = k+1
k=0

i=i+1
      
      
      