import sys
sys.path.append('/volatile/hubert/HCode/')

import numpy as np
import Optimize as o
from scipy.optimize import approx_fprime as approx
import matplotlib.pyplot as plt
import opas

s = opas.square()
shape = s.shape
l= shape[0]*shape[1]
sline = np.reshape(s,l)
#on cree nos fonctions norme de gradient a base des fonctions gradient de numpy et de hgrad
ngnp = lambda x : np.sum(np.array(np.gradient(np.reshape(x,shape)))**2)
nghp = lambda x : np.sum(np.array(o.hgrad(np.reshape(x,shape)))**2)
#elles prennent toutes les deux en entree des array de une dimension

#on cree la fonction qui va approximer la norme du gradient
apxngnp = lambda x: approx(x,ngnp,np.ones(l))
apxnghp = lambda x: approx(x,nghp,np.ones(l))
#apxngnp = lambda x: approx(x,ngnp, 0.00001*np.ones(l))
#apxnghp = lambda x: approx(x,nghp, 0.00001*np.ones(l))
#On va regarder ce que nous donne nos fonctions et les remettres sous le bon format
apnp = apxngnp(sline)
aphp = apxnghp(sline)

imapflap = o.hflapd(s,epsilon =1)
imapnp = np.reshape(apnp,shape)
imaphp = np.reshape(aphp,shape)

plt.figure(1)
plt.title('my derivate of the norm of the gradient :: hflapd')
plt.imshow(imapflap)
plt.colorbar()

plt.figure(2)
plt.title('approximate gradient (scipy.optimize.approx_fprime) \n of the norm of numpy.gradient')
plt.imshow(imapnp)
plt.colorbar()

plt.figure(3)
plt.title('approximate gradient(scipy.optimize.approx_fprime) \n of the norm of hgrad')
plt.imshow(imaphp)
plt.colorbar()

plt.figure(4)
plt.title('approx npgrad - approx hpgrad')
plt.imshow(imapnp - imaphp)
plt.colorbar()

plt.figure(5)
plt.title('hflapd - approx hpgrad')
plt.imshow(imapflap - imaphp)
plt.colorbar()

plt.figure(6)
plt.title('hflapd - approx npgrad')
plt.imshow(imapflap - imapnp)
plt.colorbar()

plt.show()