#This function give several estimations of Hurst exponent from a matlab simulation
import sys
sys.path.append('/volatile/hubert/HCode')
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from Estimators.DFA import DFAS, DFANormS
from Estimators.Whittle import WhittleS, WhittleNormS
from Estimators.Welch import WelchS, WelchNormS
from Estimators.welchnitime import WelchNitimeS, WelchNitimeNormS
from Estimators.Welchp import WelchpS
import ProgressBar
import pickle
import math


def test_simulation2(fichier = '/volatile/hubert/datas/simulations/simulationsfBn.mat',
		    sortie =None,titre=None, progressbar = True):
  f = scio.loadmat(fichier)
  """Cette fonction permet de tester les differentes fonctions de calcul
  d'exposant de Hurst et de presenter les resultats en boite a moustache
  pour se faire il faut au prealable effectuer des simulations de processus
  fractionnaire avec un outil matlab
  ces simulations devront etre presenter dans une structures a trois dimensions
  appelees simulations dont le modele se trouve dans le
  fichier Simulation_matlab/simul2.m
  """
  simulations = f['simulations']
  #number of different h
  n = simulations.shape[0]
  #number of simulation for a given h
  N = simulations.shape[1]
  #length of simulation
  l = simulations.shape[2]
  
  #dfai = np.zeros((n,N))
  #whittlei = np.zeros((n,N))
  #welchi = np.zeros((n,N))
  #welchni = np.zeros((n,N))
  welchpi = np.zeros((n,N))
  
  for i in np.arange(0,n):
    #We define anonymous functions that will be usefull
    #wh = lambda idx : WhittleS(simulations[i,:,:],idx)
    #d = lambda idx : DFAS(simulations[i,:,:],idx)
    #we = lambda idx : WelchS(simulations[i,:,:],idx)
    wp = lambda idx : WelchpS(simulations[i,:,:],idx)
    #wni = lambda idx : WelchNitimeS(simulations[i,:,:],idx)
    
    
    if progressbar:
      progressBar = ProgressBar.ProgressBar(N,30,'Work '+str(i+1)+' in progress')
      for j in range(N):
	progressBar.update(j)
	#we compute the return value
	#dfai[i,j] = d(j)
	#whittlei[i,j] = wh(j)
	#welchi[i,j] = we(j)
	#welchni[i,j] = wni(j)
	welchpi[i,j] = wp(j)
		
    else:
      for j in range(N):
	#dfai[i,j] = d(j)
	#whittlei[i,j] = wh(j)
	#welchi[i,j] = we(j)
	#welchni[i,j] = wni(j)
	welchpi[i,j] = wni(j)
	
  #donnees = {'dfa' : dfai , 'whittle' : whittlei,'welch' : welchi, 'welchnitime' : welchni}
  donnees = {'welchp' : welchpi}
  k=0
  if(sortie != None):
    with open(sortie,'wb') as fichier_sortie:
      monpickler = pickle.Pickler(fichier_sortie)
      monpickler.dump(donnees)
      
  for cle,valeur in donnees.items():
    bp = plt.boxplot(valeur.T, labels=np.arange(1,10)/10.)
    plt.title('Estimation of Hurst\ncoeffician by\n'+cle+' method')
    
    for line in bp['medians']:
      # get position data for median line
      x, y = line.get_xydata()[1] # top of median line
      # overlay median value
      if(k <6):
	plt.text(x+1.5, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
	horizontalalignment='center') # draw above, centered
      else:
	plt.text(x-2, y-0.02, '%.3f\n%.3e' % (np.mean(valeur[k,:]),
					      np.var(valeur[k,:])),
	    horizontalalignment='center') # draw above, centered
      k = k+1
    k=0
  
  return donnees,f