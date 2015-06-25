#This Code test our DFA and Whittle function gives mean and var
#of the estimate Hurst exponent
import scipy.io as scio
import numpy as np
execfile('/volatile/hubert/HCode/Welch.py')

f = scio.loadmat('/volatile/hubert/datas/simulations/simulationsfGn2.mat')

simulations = f['simulations']
#number of different h
n = simulations.shape[0]
#number of simulation for a given h
N = simulations.shape[1]
#length of simulation
l = simulations.shape[2]

welchi4096 = np.zeros((n,N))

for i in np.arange(0,n):
  #We define anonymous functions that will be usefull
  d = lambda idx : WelchS(simulations[i,:,:],idx)
  #we lauch the threads !
  for j in range(N):
    welchi4096[i,j] = d(j)

  
s ='Welch'
f = open('/volatile/hubert/datas/simulations/Python_'+s+'_results4096','w')



f.write('\n\n-----------------Welch----------------------\n \n')
f.write('Htheo\tH mean estimate for a fGn \t Var estimation\n')
for i in np.arange(0,9):
  f.write(str((i+1)/10.) + '\t')
  f.write(str(np.mean(welchi4096[i,:])) + '\t') 
  f.write(str(np.var(welchi4096[i,:])) + '\n')
  


f.close()

plt.boxplot(welchi4096.T, labels=np.arange(1,10)/10)
plt.xlabel('Htheo')
plt.ylabel('H estimate')
plt.title('Estimation of Hurst coeffician by Welch method for a fGn of lenght 514 differente value of H')
plt.show()