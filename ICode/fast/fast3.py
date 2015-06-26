import sys
import nitime.algorithms.spectral as nas
from scipy.signal import welch
import matplotlib.pyplot as plt
import ICode.Estimators as es
import numpy as np
import pickle
import ICode.data_class as dat


#with open('/volatile/hubert/datas/mysession2', 'rb') as myfile:
   #monunpickler = pickle.Unpickler(myfile)
   #while True:
    #try:
      #data = monunpickler.load()
      #sys.stdout.write('dataloaded\n')
     
    #except(EOFError):
      #break

  
  
x = data.signal
y = x[:,10000]
ynorm = (y - np.mean(y)) / np.var(y)
f,pw =  nas.get_spectra(ynorm)
f2,pw2 = welch(ynorm)
plt.figure(0)
#plt.plot(f,np.abs(pw),'blue')
#plt.plot(f2,np.abs(pw2),'red')
#plt.xlabel('freq')
#plt.ylabel('pw')
#plt.legend()
#plt.figure(1)
#plt.plot(np.log10(f),np.log10(np.abs(pw)),'blue')
#plt.xlabel('freq log10')
#plt.ylabel('pw log10')
#plt.plot(np.log10(f2),np.log10(pw2),'red')
mask = np.all([(f>0.001), (f<1)], axis =0)
tmp = np.polyfit(np.log10(f[mask]), np.log10(np.abs(pw[mask])), deg = 1)
#plt.plot(np.log10(f),tmp[0]*np.log10(f)+tmp[1] )
#plt.legend()
print tmp[0]

print DFA(ynorm)
print Welchp(ynorm)
plt.show()
