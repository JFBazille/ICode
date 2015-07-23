##This Code test our DFA and Whittle function gives mean and var
##of the estimate Hurst exponent
import scipy.io as scio
import numpy as np
from ICode.Estimators import WhittleS
from ICode.Estimators import DFAS
from ICode.opas import get_simulation

simulations = get_simulation()
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]

dfai2 = np.zeros((n,N))
whittlei2= np.zeros((n,N))
for i in np.arange(0,n):
  #We define anonymous functions that will be usefull
  w2 = lambda idx : WhittleS(simulations[i,:,:514],idx)
  d2 = lambda idx : DFAS(simulations[i,:,:514],idx)
  #we lauch the threads !
  for j in range(N):
    dfai2[i,j] = d2(j)
    whittlei2[i,j] = w2(j)
  
for s in ['DFA', 'Whittle']:
  f = open('ICode/simulations/Python_'+s+'_results514','w')
  
  if s == 'Whittle':
    f.write('\n\n------------Whittle-----------------------\n \n')
    f.write('number of simulations for a given h : %d \n'%n)
    f.write('lengh of each simulation : %d \n'%l)
    f.write('Htheo\tH mean estimate for a fGn \t Var estimation\n')
    for i in np.arange(0,9):
      f.write(str((i+1)/10) + '\t')

      f.write(str(np.mean(whittlei2[i,:])) + '\t')
      
      f.write(str(np.var(whittlei2[i,:])) + '\n')
      
    
  if s == 'DFA':
    f.write('\n\n----------DFA-----------------------\n \n')
    f.write('number of simulations for a given h : %d \n'%n)
    f.write('lengh of each simulation : %d \n'%l)
    f.write('Htheo\tH mean estimate for a fGn \t Var estimation\n')

    for i in np.arange(0,9):
      f.write(str((i+1)/10) + '\t')

      f.write(str(np.mean(dfai2[i,:])) + '\t')
      
      f.write(str(np.var(dfai2[i,:])) + '\n')
      

    
  f.close()
