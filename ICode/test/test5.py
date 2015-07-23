#This Code test our DFA and Whittle function gives mean and var
#of the estimate Hurst exponent
import scipy.io as scio
import numpy as np
from ICode.Estimators import Welch
from ICode.opas import get_simulation

simulations = get_simulation()
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

  

f = open('Python_Welche_results','w')

f.write('\n\n----------welch-----------------------\n \n')
f.write('Htheo\tH mean estimate for a fGn \t Var estimation\n')

f.write(str(np.mean(welchi)) + '\t') 
f.write(str(np.var(welchi)) + '\n')
    

  
f.close()
