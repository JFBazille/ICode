##This function has been implemented to test wepl2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ICode.estimators import penalyzed
import ICode.opas as opas
from ICode.optimize.objective_functions import _unmask
from ICode.estimators.wavelet.hdwpab import *
from math import ceil
from scipy.optimize import approx_fprime as approx
from ICode.estimators.penalyzed import lipschitz_constant_gradf
j1 = 3
j2 = 6
wtype = 1
l = 514
title = '  '
#s = opas.square2(10)
#mask = np.ones(s.shape, dtype=bool)
s = opas.smiley()
mask = s > 0.1

signal = opas.get_simulation_from_picture(s, lsimul=l)
signalshape = signal.shape
shape = signalshape[: - 1]
sig = np.reshape(signal, (signalshape[0] * signalshape[1], signalshape[2]))
N = sig.shape[0]

estimate = np.zeros(N)
aest = np.zeros(N)
simulation = np.cumsum(sig, axis=1)

####Another way to compute Hurst exponent more precise but longer
#estimate = np.zeros(N)
#nbvoies = int(np.log2(l))
#nbvanishmoment = 4
#j2 = 5
#nbvoies = min(int(np.log2(l / (2 * nbvanishmoment + 1))), nbvoies)
#Bar = ProgressBar(N, 60, 'Work in progress')
#Elog = np.zeros((N, nbvoies))
#Varlog = np.zeros((N, nbvoies))


#for i in np.arange(0, N):
    #Bar.update(i)
    #dico = wtspecq_statlog3(simulation[i], nbvanishmoment, 1,
    #                        np.array(2), nbvoies, 0, 0)
    #sortie = regrespond_det2(dico['Elogmuqj'][0], dico['Varlogmuqj'][0], 2,
    #                         dico['nj'], j1, j2, wtype)
    #estimate[i] = sortie['Zeta'] / 2.
    #Elog[i] = dico['Elogmuqj'][0]
    #Varlog[i] = dico['Varlogmuqj'][0]
    #if i == 0:
        #nj = dico['nj']
        #Bar.update(i + 1)

#print 'computation time   ::  ' + str(time.time() - tic)+'\n'

#######################################################################
dico = wtspecq_statlog32(simulation, 2, 1, np.array(2), int(np.log2(l)), 0, 0)
Elog = dico['Elogmuqj'][:, 0]
Varlog = dico['Varlogmuqj'][:, 0]
nj = dico['nj']

for j in np.arange(0, N):
    sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
    estimate[j] = sortie['Zeta'] / 2.
    aest[j] = sortie['aest']
##########
lip = 2
#lip = 0 -> H_size*np.sum(4 * ((j1j2 + 1) ** 2) * wvarjj)
#lip = 1 -> H_size*np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)
#lip = 2 -> np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)
H_size = len(estimate[mask.ravel()])
j22 = np.min((j2, len(nj)))
j1j2 = np.arange(j1 - 1, j22)
njj = nj[j1j2]
N = sum(njj)
wvarjj = njj / N
lipschitz_constant =  np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)

if choice == 1:
    l1_ratio = 0
    f = lambda lbda: penalyzed.mtvsolver(estimate[mask.ravel()], aest[mask.ravel()],
                                        Elog[mask.ravel()], Varlog[mask.ravel()],
                                        nj, j1, j2,mask,
                                        lipschitz_constant=lipschitz_constant,
                                        l1_ratio = l1_ratio, l=lbda)
    title = 'wetvp'


if choice == 2:
    l1_ratio = 0
    f = lambda lbda: penalyzed.mtvsolverbis(np.concatenate((estimate[mask.ravel()],
                                                            aest[mask.ravel()])),
                                        Elog[mask.ravel()], Varlog[mask.ravel()],
                                        nj, j1, j2,mask, max_iter = 100,
                                        l1_ratio = l1_ratio,
                                        lipschitz_constant=0, l=lbda)
    title = 'wetvp_bis'

lmax = 15
minimiseurs = np.zeros((lmax,) + s.shape)
minfounded = list()
cg = np.zeros(lmax)
rmse = np.zeros(lmax)
r = np.arange(lmax)
lbda = np.array((0,) + tuple(1.5 ** r[:- 1]))


if choice % 2 == 1:
    for idx in r:
        monmin = f(lbda[idx])
        minimiseurs[idx] = _unmask(monmin[0], mask)
        rmse[idx] = np.sqrt(np.mean((minimiseurs[idx] - s) ** 2))
        minfounded.append(monmin[1])

else:
    for idx in r:
        monmin = f(lbda[idx])
        minimiseurs[idx] = _unmask(monmin[0][:len(monmin) / 2], mask)
        rmse[idx] = np.sqrt(np.mean((minimiseurs[idx] - s) ** 2))
        minfounded.append(monmin[1])

plt.figure(1)
plt.title(title)

fig, axes = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
fig2, axes2 = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
for idx, (dat, ax, ax2) in enumerate(zip(minimiseurs, axes.flat, axes2.flat)):
    im = ax.imshow(dat, norm=Normalize(vmin=np.min(minimiseurs),
                                       vmax=np.max(minimiseurs)))
    im.set_interpolation('nearest')
    im2 = ax2.imshow(dat)
    im2.set_interpolation('nearest')
    #image[int(idx/5.)*shape[0]:(int(idx/5.)+1)*shape[0],
    #(idx%5)*shape[1]:(idx%5+1)*shape[0]] = dat
    ax.set_title("l = %.1f " % (lbda[idx]))
    ax2.set_title("l= %.1f " % (lbda[idx]))

cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
fig.colorbar(im, cax=cax)
cax2 = fig2.add_axes([0.91, 0.1, 0.028, 0.8])
fig2.colorbar(im2, cax=cax2)

fig.savefig('/volatile/hubert/beamer/presentation_juin_2015/wetv/' + title + '_graph'+str(lip)+'.pdf')

fig3 = plt.figure()
plt.plot(lbda, rmse, 'r')
plt.ylabel('rmse', color='r')
plt.xlabel('lambda')

fig3.savefig('/volatile/hubert/beamer/presentation_juin_2015/wetv/' + title + '_rmse'+str(lip)+'.pdf')
print title
del fig, fig2, fig3

#choice += 1
#if choice < 22 and not choice == 15:
    #execfile('/volatile/hubert/ICode/ICode/test/test_welp2_opas_rmse.py')

plt.show()