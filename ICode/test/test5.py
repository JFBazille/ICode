#This Code test our DFA and Whittle function gives mean and var
#of the estimate Hurst exponent
import scipy.io as scio
import numpy as np
execfile('/volatile/hubert/HCode/Welch.py')

f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn514.mat')

simulations = f['simulations']
#number of different h

#number of simulation for a given h
N = simulations.shape[0]
#length of simulation
l = simulations.shape[1]

welchi = np.zeros(N)

#We define anonymous functions that will be usefull
we = lambda simul : Welch(simul)
#we lauch the threads !
for j in range(N):
  welchi[j] = we(simulations[j])

  

f = open('/volatile/hubert/datas/simulations/Python_Welche_results','w')


    
  

f.write('\n\n----------welch-----------------------\n \n')
f.write('Htheo\tH mean estimate for a fGn \t Var estimation\n')

f.write(str(np.mean(welchi)) + '\t') 
f.write(str(np.var(welchi)) + '\n')
    

  
f.close()
