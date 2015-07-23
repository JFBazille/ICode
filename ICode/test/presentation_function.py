import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import ICode.opas as opas
import scipy.io as scio
from ICode.estimators.wavelet import wtspecq_statlog32, regrespond_det2
from scipy.optimize import fmin_l_bfgs_b
from ICode.optimize.objective_functions import _unmask
from ICode.progressbar import ProgressBar
from ICode.estimators import penalized
from scipy.ndimage.filters import gaussian_filter
from math import ceil
from matplotlib.colors import Normalize

def diff_perf_boxplot():
    with open(os.path.join('./ICode','test','resultat_test_estimators'),'rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        data = unpickler.load()

    mlist = list()
    mlist.append(('Wavelet', data['Wavelet_514']))
    mlist.append(('Welch', data['Welch_514']))
    mlist.append(('DFA', data['DFA_514']))
    mlist.append(('Wavelet', data['Wavelet_4096']))
    mlist.append(('Welch', data['Welch_4096']))
    mlist.append(('DFA', data['DFA_4096']))


    for i,(title,stat) in enumerate(mlist):
        k = 0
        if i == 0:
            fig = plt.figure(0)
            f, myplotswavelet = plt.subplots(1, 3, sharey=True)
            f.suptitle('Estimation of Hurst coefficient of fGn\nof length 514 by different methods')

        if i == 3:
            fig = plt.figure(1)
            f, myplotswavelet = plt.subplots(1, 3, sharey=True)
            f.suptitle('Estimation of Hurst coefficient of fGn\nof length 4096 by differents methods')

        idx_subplot = i%3
        myplotswavelet[idx_subplot].set_title(title)

        bp = myplotswavelet[idx_subplot].boxplot(stat.T)
        for line in bp['medians']:# get position data for median line
            x, y = line.get_xydata()[1] # top of median line
            # overlay median value
            if(k <6):
                myplotswavelet[idx_subplot].text(x + 1.5, y - 0.02, '%.3f\n%.3e' % (np.mean(stat[k, :]),
                                                    np.var(stat[k,:])),
                horizontalalignment='center') # draw above, centered
            else:
                myplotswavelet[idx_subplot].text(x - 2, y - 0.02, '%.3f\n%.3e' % (np.mean(stat[k, :]),
                                                    np.var(stat[k, :])),
                    horizontalalignment='center') # draw above, centered
            k +=1
    plt.show()


def plot_syj_against_j(j1=2, j2=6, wtype=1, theoretical_Hurst=0.8, idx_simulation=0):
    idx = int(theoretical_Hurst * 10) - 1
    if(idx < 0 or idx > 10):
        idx = 7
    simulation = np.cumsum(opas.get_simulation(), axis=-1)
    
    dico = wtspecq_statlog3(simulation[idx,idx_simulation], 2, 1, np.array(2),
                            int(np.log2(simulation[idx,idx_simulation].shape[0])), 0, 0)
    Elog = dico['Elogmuqj'][0]
    Varlog = dico['Varlogmuqj'][0]
    nj = dico['nj']
    regression = regrespond_det2(Elog, Varlog, np.array(2), nj, j1, j2, wtype)
    
    jmax = len(Elog)

    j_indices = np.arange(0,jmax + 2)
    plt.plot(j_indices, j_indices * regression['Zeta'] + regression['aest'])
    plt.text(j_indices.mean() - 2 * j_indices.var() / jmax, Elog.mean() + Elog.var() / jmax, r'Hurst Exponent = %.2f'%(regression['Zeta']/2))

    j_indices = np.arange(0,jmax) + 1
    plt.plot(j_indices, Elog, 'ro')
    plt.xlabel('scale j')
    plt.ylabel('Sy(j,2) value')
    plt.show()


def plot_python_against_matlab_wavelet(theoretical_Hurst=0.8):
    idx = int(theoretical_Hurst * 10) - 1
    if(idx < 0 or idx > 10):
        idx = 7
    with open('./ICode/test/resultat_test_estimators','rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        donnees = unpickler.load()

    pwave_514 = donnees['Wavelet_514']
    pwave_4096 = donnees['Wavelet_4096']

    mdata = scio.loadmat('./ICode/test/matlab_wavelet_estimations.mat')
    mwave_4096 = mdata['matlab_wavelet_4096']
    mwave_514 = mdata['matlab_wavelet_514']

    mini = np.min((mwave_4096[idx], pwave_4096[idx])) - 0.01
    maxi = np.max((mwave_4096[idx], pwave_4096[idx])) + 0.01
    values = np.linspace(mini,maxi,1000)
    plt.plot(mwave_4096[idx], pwave_4096[idx], 'ro')
    plt.plot(values,values)
    plt.show()


def plot_python_against_matlab_wavelet_all():
    with open('./ICode/test/resultat_test_estimators','rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        donnees = unpickler.load()

    pwave_514 = donnees['Wavelet_514']
    pwave_4096 = donnees['Wavelet_4096']

    mdata = scio.loadmat('./ICode/test/matlab_wavelet_estimations.mat')
    mwave_4096 = mdata['matlab_wavelet_4096']
    mwave_514 = mdata['matlab_wavelet_514']

    mini = np.min((mwave_4096, pwave_4096)) - 0.01
    maxi = np.max((mwave_4096, pwave_4096)) + 0.01
    values = np.linspace(mini,maxi,1000)
    plt.plot(mwave_4096.ravel(), pwave_4096.ravel(), 'ro')
    plt.plot(values,values)
    plt.show()


def test_simulated_image(j1=3, j2=6, wtype=1, length_simul=514, title_prefix='test_simulated_image',
                         figure='smiley', size=10, mask=True):
    if figure=='smiley':
        s = opas.smiley(size)
    else:
        s = opas.square2(size)

    if mask:
        mask = s > 0.1
    else:
        mask = np.ones(s.shape, dtype=bool)

    signal = opas.get_simulation_from_picture(s, lsimul=length_simul)
    signalshape = signal.shape
    shape = signalshape[:- 1]
    sig = np.reshape(signal, (signalshape[0] * signalshape[1], signalshape[2]))
    N = sig.shape[0]

    estimate = np.zeros(N)
    aest = np.zeros(N)
    simulation=np.cumsum(sig, axis=1)

    #######################################################################

    dico = wtspecq_statlog32(simulation, 2, 1, np.array(2),
                                int(np.log2(length_simul)), 0, 0)
    Elog = dico['Elogmuqj'][:, 0]
    Varlog = dico['Varlogmuqj'][:, 0]
    nj = dico['nj']

    for j in np.arange(0, N):
        sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
        estimate[j] = sortie['Zeta'] / 2. #normalement Zeta
        aest[j]  = sortie['aest']

    #######################################################################

    f = lambda x, lbda: penalized.loss_l2_penalization_on_grad(x, aest[mask.ravel()],
                        Elog[mask.ravel()], Varlog[mask.ravel()], nj, j1, j2, mask, l=lbda)
    #We set epsilon to 0
    g = lambda x, lbda: penalized.grad_loss_l2_penalization_on_grad(x, aest[mask.ravel()],
                        Elog[mask.ravel()], Varlog[mask.ravel()], nj, j1, j2, mask, l=lbda)

    l2_title = title_prefix + 'loss_l2_penalisation_on_grad'

    fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
    #For each lambda we use blgs algorithm to find the minimum
    # We start from the
    l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), estimate[mask.ravel()])

    #######################################################################

    j22 = np.min((j2, len(nj)))
    j1j2 = np.arange(j1 - 1, j22)
    njj = nj[j1j2]
    N = sum(njj)
    wvarjj = njj / N
    lipschitz_constant =  np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)
    l1_ratio = 0
    tv_algo = lambda lbda: penalized.mtvsolver(estimate[mask.ravel()], aest[mask.ravel()],
                                        Elog[mask.ravel()], Varlog[mask.ravel()],
                                        nj, j1, j2,mask,
                                        lipschitz_constant=lipschitz_constant,
                                        l1_ratio = l1_ratio, l=lbda)
    tv_title = title_prefix + 'wetvp'

    #######################################################################

    lmax = 15
    l2_minimizor = np.zeros((lmax,) + s.shape)
    l2_rmse = np.zeros(lmax)
    tv_minimizor = np.zeros((lmax,) + s.shape)
    tv_rmse = np.zeros(lmax)

    r = np.arange(lmax)
    lbda = np.array((0,) + tuple(1.5 ** r[:- 1]))

    for idx in r:
        algo_min = l2_algo(lbda[idx])
        l2_minimizor[idx] = _unmask(algo_min[0], mask)
        l2_rmse[idx] = np.sqrt(np.mean((l2_minimizor[idx] - s) ** 2))

        if idx == 0:
            l2_min_rmse = l2_rmse[idx]
            l2_min_rmse_idx = 0
        else:
            if l2_min_rmse > l2_rmse[idx]:
                l2_min_rmse = l2_rmse[idx]
                l2_min_rmse_idx = idx

        algo_min = tv_algo(lbda[idx])
        tv_minimizor[idx] = _unmask(algo_min[0], mask)
        tv_rmse[idx] = np.sqrt(np.mean((tv_minimizor[idx] - s) ** 2))

        if idx == 0:
            tv_min_rmse = l2_rmse[idx]
            tv_min_rmse_idx = 0
        else:
            if tv_min_rmse > tv_rmse[idx]:
                tv_min_rmse = tv_rmse[idx]
                tv_min_rmse_idx = idx

    #######################################################################

    for minimizor_idx, (title, minimizor) in enumerate(zip([tv_title, l2_title],
                                    [tv_minimizor, l2_minimizor])):

        plt.figure(1)
        plt.title(title)

        fig, axes = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
        fig2, axes2 = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
        for idx, (dat, ax, ax2) in enumerate(zip(minimizor, axes.flat, axes2.flat)):
            im = ax.imshow(dat, norm=Normalize(vmin=np.min(minimizor),
                                            vmax=np.max(minimizor)), interpolation='nearest')
            ax.axis('off')
            ax.set_title("$\lambda$ = %.1f " % (lbda[idx]))

            im2 = ax2.imshow(dat, interpolation='nearest')
            ax2.axis('off')
            ax2.set_title("$\lambda$ = %.1f " % (lbda[idx]))

        cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
        fig.colorbar(im, cax=cax)
        cax2 = fig2.add_axes([0.91, 0.1, 0.028, 0.8])
        fig2.colorbar(im2, cax=cax2)
        fig.savefig('/volatile/hubert/beamer/graphics/juillet2015/' + title + '_graph.pdf')

    fig = plt.figure()
    plt.title('l2 minimizor of rmse $\lambda$ = %.1f ' % (lbda[l2_min_rmse_idx]))
    im = plt.imshow(l2_minimizor[l2_min_rmse_idx], norm=Normalize(vmin=np.min(l2_minimizor),
                                        vmax=np.max(l2_minimizor)), interpolation='nearest')
    plt.axis('off')
    cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig('/volatile/hubert/beamer/graphics/juillet2015/'+l2_title+'minimizor.pdf')

    fig = plt.figure()
    plt.title('tv minimizor of rmse $\lambda$ = %.1f ' % (lbda[tv_min_rmse_idx]))
    im = plt.imshow(tv_minimizor[tv_min_rmse_idx], norm=Normalize(vmin=np.min(tv_minimizor),
                                        vmax=np.max(tv_minimizor)), interpolation='nearest')
    plt.axis('off')
    cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig('/volatile/hubert/beamer/graphics/juillet2015/'+tv_title+'minimizor.pdf')

    ##image of difference
    l2_diff = l2_minimizor[l2_min_rmse_idx]-s
    tv_diff = tv_minimizor[tv_min_rmse_idx]-s
    normalize_vmax = np.max((l2_diff, tv_diff))
    normalize_vmin = np.min((l2_diff, tv_diff))

    fig = plt.figure()
    plt.title('l2 minimizor of rmse, difference with original image $\lambda$ = %.1f ' % (lbda[l2_min_rmse_idx]))
    im = plt.imshow(l2_diff, norm=Normalize(vmin=normalize_vmin,
                                        vmax=normalize_vmax), interpolation='nearest')
    plt.axis('off')
    cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig('/volatile/hubert/beamer/graphics/juillet2015/'+l2_title+'minimizordiff.pdf')

    fig = plt.figure()
    plt.title('tv minimizor of rmse, difference with original image $\lambda$ = %.1f ' % (lbda[tv_min_rmse_idx]))
    im = plt.imshow(tv_diff, norm=Normalize(vmin=normalize_vmin,
                                        vmax=normalize_vmax), interpolation='nearest')
    plt.axis('off')
    cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
    fig.colorbar(im, cax=cax)
    fig.savefig('/volatile/hubert/beamer/graphics/juillet2015/'+tv_title+'minimizordiff.pdf')

    fig3 = plt.figure()
    plt.plot(lbda, l2_rmse, 'r', label='l2 rmse')
    plt.plot(lbda, tv_rmse, 'b', label='tv rmse')
    plt.axvline(lbda[l2_min_rmse_idx], color='r')
    plt.axvline(lbda[tv_min_rmse_idx], color='b')
    plt.ylabel('rmse')
    plt.xlabel('lambda')
    plt.legend()

    fig3.savefig('/volatile/hubert/beamer/graphics/juillet2015/' + title + '_rmse.pdf')
    print title

    plt.show()