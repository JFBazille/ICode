##This function has been implemented to test wepl2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
tic = time.time()
from scipy.optimize import fmin_l_bfgs_b, check_grad
from ICode.estimators import penalyzed
import ICode.opas as opas
from ICode.optimize.objective_functions import _unmask
from ICode.estimators.wavelet.hdwpab import *
from math import ceil
from scipy.optimize import approx_fprime as approx
j1 = 3
j2 = 6
wtype = 1
l = 514
title = '  '
#s = opas.square2(10)
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

#choice =4
#Ici pour tester J2 et GradJ2
if choice == 1:
    f = lambda x, lbda: penalyzed.j2(x, aest, Elog, Varlog, nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradj2(x, aest, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    H = estimate
    title = 'J2_GradJ2'

##Ici pour tester J2bis et GradJ2bis
if choice == 2:
    f = lambda x, lbda: penalyzed.j2bis(x, Elog, Varlog, nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradj2bis(x, Elog, Varlog,
                                            nj, j1, j2, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'J2bis_GradJ2bis'

##Ici pour tester JH2 et GradJH2
if choice == 3:
    f = lambda x, lbda: penalyzed.jh2(x, aest, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjh2(x, aest, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    H = estimate
    title = 'JH2_GradJH2'

##Ici pour tester JH2bis etGradJH2bis
if choice == 4:
    f = lambda x, lbda: penalyzed.jh2bis(x, shape, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjh2bis(x, shape, Elog, Varlog,
                                             nj, j1, j2, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JH2bis_GradJH2bis'

##Ici pour tester JH22 et GradJH22
if choice == 5:
    f = lambda x, lbda: penalyzed.jh22(x, aest, shape, Elog, Varlog,
                                       nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjh22(x, aest, shape, Elog, Varlog,
                                           nj, j1, j2, l=lbda)
    H = estimate
    title = 'JH22_GradJH22'

#Ici pour tester JH22bis et GradJH22bis
if choice == 6:
    f = lambda x, lbda: penalyzed.jh22bis(x, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjh22bis(x, shape, Elog, Varlog,
                                              nj, j1, j2, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JH22bis_GradJH22bis'

##Ici pour tester JHu et GradJHu
if choice == 7:
    f = lambda x, lbda: penalyzed.jhu(x, aest, shape, Elog, Varlog,
                                     nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhu(x, aest, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    H = estimate
    title = 'JHu_GradJHu'

##Ici pour tester JHubis et GradJHubis
if choice == 8:
    f = lambda x, lbda: penalyzed.jhubis(x, shape, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhubis(x, shape, Elog, Varlog,
                                             nj, j1, j2, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JHubis_GradJHubis'

##Ici pour tester JHu et GradJHud
if choice == 9:
    f = lambda x, lbda: penalyzed.jhu(x, aest, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhud(x, aest, shape, Elog, Varlog,
        nj, j1, j2, epsilon=np.sqrt(np.finfo(float).eps) / 100, l=lbda)
    H = estimate
    title = 'JHu_GradJHud'

##Ici pour test JHubis et JHudbis
if choice == 10:
    f = lambda x, lbda: penalyzed.jhubis(x, shape, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhudbis(x, shape, Elog, Varlog,
                nj, j1, j2, epsilon=np.sqrt(np.finfo(float).eps) / 100, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JHubis_GradJHudbis'

##Ici pour tester autre chose
if choice == 11:
    f = lambda x, lbda: penalyzed.jh2(x, aest, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = estimate
    title = 'JH2approx'

##Ici pour tester autre chose
if choice == 12:
    f = lambda x, lbda: penalyzed.jh2bis(x, shape, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = np.concatenate((estimate, aest))
    title = 'JH2bisapprox'

##Ici pour tester autre chose
if choice == 13:
    f = lambda x, lbda: penalyzed.jhu(x, aest, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = estimate
    title = 'JHUapprox'

##Ici pour tester autre chose
if choice == 14:
    f = lambda x, lbda: penalyzed.jh22bis(x, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = np.concatenate((estimate, aest))
    title = 'JH22bisapprox'

##Ici pour tester autre chose
if choice == 16:
    f = lambda x, lbda: penalyzed.jhubis(x, shape, Elog, Varlog,
                                         nj, j1, j2, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = np.concatenate((estimate, aest))
    title = 'JHubisapprox'


##Ici pour tester autre chose
if choice == 17:
    f = lambda x, lbda: penalyzed.jhumask(x, aest, shape, Elog, Varlog,
                                          nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = estimate
    title = 'JHumaskapprox'

##Ici pour tester autre chose
if choice == 18:
    f = lambda x, lbda: penalyzed.jhumaskbis(x, shape, Elog, Varlog,
                                             nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: approx(x, lambda y: f(y, lbda),
                               0.000001 * np.ones(len(x)))
    H = np.concatenate((estimate, aest))
    title = 'JHumaskbisapprox'

##Ici pour tester autre chose
if choice == 19:
    f = lambda x, lbda: penalyzed.jhumask(x, aest, shape, Elog, Varlog,
                                          nj, j1, j2, mask, l=lbda)
    #we set epsilon to 0
    g = lambda x, lbda: penalyzed.gradjhumask(x, aest, shape, Elog, Varlog,
                                        nj, j1, j2, mask, epsilon=0, l=lbda)
    H = estimate
    title = 'JHumaskGradJhudmask'

##Ici pour tester autre chose
if choice == 20:
    f = lambda x, lbda: penalyzed.jhumaskbis(x, shape, Elog, Varlog,
                                             nj, j1, j2, mask, l=lbda)
    #we set epsilon to 0
    g = lambda x, lbda: penalyzed.gradjhumaskbis(x, shape, Elog, Varlog,
                                                 nj, j1, j2, mask, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JHumaskbis_GradJhudmaskbis'

##Ici pour tester autre chose
if choice == 21:
    #In that configuration, datas out of the mask are not available
    Elog2 = np.reshape(Elog, shape + (Elog.shape[- 1],))[mask]
    Varlog2 = np.reshape(Varlog, shape + (Varlog.shape[- 1],))[mask]
    f = lambda x, lbda: penalyzed.jhm(x, aest, shape, Elog2, Varlog2,
                                      nj, j1, j2, mask, l=lbda)
    #we set epsilon to 0
    g = lambda x, lbda: penalyzed.gradjhm(x, aest, shape, Elog2, Varlog2,
                                          nj, j1, j2, mask, epsilon=0, l=lbda)
    H = estimate
    title = 'JHm_GradJHm'

##Ici pour tester autre chose
if choice == 22:
    #In that configuration, datas out of the mask are not available
    Elog2 = np.reshape(Elog, shape + (Elog.shape[- 1],))[mask]
    Varlog2 = np.reshape(Varlog, shape + (Varlog.shape[- 1],))[mask]
    f = lambda x, lbda: penalyzed.jhmbis(x, shape, Elog2, Varlog2,
                                         nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhmbis(x, shape, Elog2, Varlog2,
                                        nj, j1, j2, mask, epsilon=0, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JHmbis_GradJHmbis'

if choice == 23:
    f = lambda x, lbda: penalyzed.jhe(x, aest, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhe(x, aest, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    H = estimate
    title = 'JHe_GradJHe'

if choice == 24:
    f = lambda x, lbda: penalyzed.jhebis(x, shape, Elog, Varlog,
                                      nj, j1, j2, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhebis(x, shape, Elog, Varlog,
                                          nj, j1, j2, l=lbda)
    H = np.concatenate((estimate, aest))
    title = 'JHebis_GradJHebis'

if choice == 25:
    #In that configuration, datas out of the mask are not available
    Elog2 = np.reshape(Elog, shape + (Elog.shape[- 1],))[mask]
    Varlog2 = np.reshape(Varlog, shape + (Varlog.shape[- 1],))[mask]
    f = lambda x, lbda: penalyzed.jhem(x, aest[mask.ravel()], Elog2, Varlog2,
                                      nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhem(x, aest[mask.ravel()], Elog2, Varlog2,
                                          nj, j1, j2, mask, l=lbda)
    H = estimate[mask.ravel()]
    title = 'JHem_GradJHem'

if choice == 26:
    #In that configuration, datas out of the mask are not available
    Elog2 = np.reshape(Elog, shape + (Elog.shape[- 1],))[mask]
    Varlog2 = np.reshape(Varlog, shape + (Varlog.shape[- 1],))[mask]
    f = lambda x, lbda: penalyzed.jhembis(x,
                                          Elog2, Varlog2,
                                      nj, j1, j2, mask, l=lbda)
    g = lambda x, lbda: penalyzed.gradjhembis(x,
                                              Elog2, Varlog2,
                                          nj, j1, j2, mask, l=lbda)
    H = np.concatenate((estimate[mask.ravel()], aest[mask.ravel()]))
    title = 'JHembis_GradJHembis'

##if not 'bis'
#cg= check_grad(f,g,estimate)
#monmin = fmin_l_bfgs_b(fg, estimate)
#minimiseur  = monmin[0]

fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
#for each lambda this function compute the minimum starting from the estimation
fmin = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), H)
ckgrad = lambda lbda: check_grad(lambda x: f(x, lbda),
                                 lambda x: g(x, lbda), H)
                                 #, epsilon = )

lmax = 15
minimiseurs = np.zeros((lmax,) + s.shape)
cg = np.zeros(lmax)
rmse = np.zeros(lmax)
r = np.arange(lmax)
lbda = np.array((0,) + tuple(1.5 ** r[:- 1]))

if choice % 2 == 1:
    for idx in r:
        monmin = fmin(lbda[idx])
        cg[idx] = ckgrad(lbda[idx])
        if 'JHem' in title:
            minimiseurs[idx] = _unmask(monmin[0], mask)
        else:
            minimiseurs[idx] = _unmask(monmin[0], mask)
        rmse[idx] = np.sqrt(np.mean((minimiseurs[idx] - s) ** 2))

else:
    for idx in r:
        monmin = fmin(lbda[idx])
        cg[idx] = ckgrad(lbda[idx])
        minimiseurs[idx] = np.reshape(monmin[0][:monmin[0].shape[0] / 2],
                                      s.shape)
        rmse[idx] = np.sqrt(np.mean((minimiseurs[idx] - s) ** 2))

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

fig.savefig('/volatile/hubert/beamer/presentation_juin_2015/' + title + '_graph.pdf')

fig3, ax1 = plt.subplots()
ax1.plot(lbda, cg)
ax1.set_xlabel('\lambda')
ax1.set_ylabel('check_gradient', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
    #set the left graduation color in blue

ax2 = ax1.twinx()
ax2.plot(lbda, rmse, 'r')
ax2.set_ylabel('rmse', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    #set the left graduation color in blue


fig3.savefig('/volatile/hubert/beamer/presentation_juin_2015/' + title + '_rmse.pdf')
print title
del fig, fig2, fig3

#choice += 1
#if choice < 22 and not choice == 15:
    #execfile('/volatile/hubert/ICode/ICode/test/test_welp2_opas_rmse.py')
