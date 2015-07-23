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
i=0
k=0
whelchfig = -1
for cle, valeur in donnees.items():
  if 'Whittle' in cle:
    if '514' in cle:
      fig = plt.figure(i)
      f, myplots = plt.subplots(1,2,sharey=True)
      f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length 514  by Whittle method')
      bp = myplots[0].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      myplots[0].set_title('Python estimator')
      myplots[1].set_title('Matlab estimator')
      for line in bp['medians']:
	# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      bp = myplots[1].boxplot(mwhittles.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:
	# get position data for median line
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
      
    if '4096' in cle:
      fig = plt.figure(i)
      f, myplots = plt.subplots(1,2,sharey=True)
      myplots[0].set_title('Python estimator')
      myplots[1].set_title('Matlab estimator')
      f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length 4096  by DFA method')
      bp = myplots[0].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:
	# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      bp = myplots[1].boxplot(mwhittles.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:
	# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplots[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittlel[k,:]),
						np.var(mwhittlel[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplots[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(mwhittlel[k,:]),
						np.var(mwhittlel[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      
  if 'DFA' in cle:
    if '514' in cle:
      fig = plt.figure(i)
      f, myplots = plt.subplots(1,2,sharey=True)
      myplots[0].set_title('Python estimator')
      myplots[1].set_title('Matlab estimator')
      f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length 514  by DFA method')
      bp = myplots[0].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      bp = myplots[1].boxplot(mwhittles.T, labels=np.arange(1,10)/10.)
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
      
    if '4096' in cle:
      fig = plt.figure(i)
      f, myplots = plt.subplots(1,2,sharey=True)
      myplots[0].set_title('Python estimator')
      myplots[1].set_title('Matlab estimator')
      f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length 4096  by DFA method')
      bp = myplots[0].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplots[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplots[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      bp = myplots[1].boxplot(mwhittles.T, labels=np.arange(1,10)/10.)
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
  if 'Welch' in cle:
    if whelchfig == -1:
      fig = plt.figure(i)
      f, myplotswelch = plt.subplots(1,2,sharey=True)
      myplotswelch[0].set_title('length 514')
      myplotswelch[1].set_title('legnth4096')
      f.suptitle('Estimation of Hurst\ncoefficient of fGn\nof length 514 and 4096  by Welch method')

      whelchfig = 1
      
    if '514' in cle:
      bp = myplotswelch[0].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:
	# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplotswelch[0].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplotswelch[0].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0
      
    if '4096' in cle:
      bp = myplotswelch[1].boxplot(valeur.T, labels=np.arange(1,10)/10.)
      for line in bp['medians']:
	# get position data for median line
	x, y = line.get_xydata()[1] # top of median line
	# overlay median value
	if(k <6):
	  myplotswelch[1].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
	else:
	  myplotswelch[1].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
						np.var(valeur[k,:])),
	      horizontalalignment='center') # draw above, centered
	k = k+1
      k=0

  i=i+1
      
      
      