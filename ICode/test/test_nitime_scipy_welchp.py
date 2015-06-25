#This Code test our Hexp_Welchp function gives mean and var
#of the estimate Hurst exponent
import sys
sys.path.append('/volatile/hubert/HCode/')
import scipy.io as scio
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Estimators.Hexp_Welchp import Hurstexp_Welchper as HW
#from Estimators.Hexp_Welchp import Hurstexp_Welchper_scipy as HWs
f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]

estimate01 = np.zeros((n,N))
##estimate_s01 = np.zeros((n,N))
estimate005 = np.zeros((n,N))
##estimate_s005 = np.zeros((n,N))
for i in np.arange(0,n):
  estimate01[i]=HW(simulations[i,:,:])
  #estimate_s01[i] = HWs(simulations[i,:,:514]) 
  estimate005[i]=HW(simulations[i,:,:])
  #estimate_s005[i] = HWs(simulations[i,:,:514],f_max = 0.01) 
donnees = {'Welch_nitime_per01' : estimate01,'Welch_,nitime_per005' :estimate005}#,
	   #'Welch_scipy_per01' : estimate_s01, 'Welch_scipy_per005' :estimate_s005} 
#s ='Welch_per'
#with open('/volatile/hubert/datas/simulations/Python_'+s+'_results514_norm','w') as fichier:
  #monpickler = pickle.Pickler(fichier)
  #monpickler.dump(donnees)  

i=0
j=0
k=0


f, myplots = plt.subplots(1,2,sharey=True)
for cle,valeur in donnees.items():
  bp = myplots[j].boxplot(valeur.T, labels=np.arange(1,10)/10.)
  myplots[j].set_title('Estimation of Hurst\ncoeffician of fGn by\n'+cle+'method')
  
  for line in bp['medians']:
    # get position data for median line
    x, y = line.get_xydata()[1] # top of median line
    # overlay median value
    if(k <6):
      myplots[j].text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
      horizontalalignment='center') # draw above, centered
    else:
      myplots[j].text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					    np.var(valeur[k,:])),
	  horizontalalignment='center') # draw above, centered
    k = k+1
  k=0
  j = j+1

f.show()