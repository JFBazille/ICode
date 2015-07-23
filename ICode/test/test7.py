#if figure=='smiley':
    #s = opas.smiley(size)
#else:
    #s = opas.square2(size)

#if mask:
    #mask = s > 0.1
#else:
    #mask = np.ones(s.shape, dtype=bool)

#signal = opas.get_simulation_from_picture(s, lsimul=length_simul)
#signalshape = signal.shape
#shape = signalshape[:- 1]
#sig = np.reshape(signal, (signalshape[0] * signalshape[1], signalshape[2]))
#N = sig.shape[0]

#estimate = np.zeros(N)
#aest = np.zeros(N)
#simulation=np.cumsum(sig, axis=1)

########################################################################

#dico = wtspecq_statlog32(simulation, 2, 1, np.array(2),
                            #int(np.log2(length_simul)), 0, 0)
#Elog = dico['Elogmuqj'][:, 0]
#Varlog = dico['Varlogmuqj'][:, 0]
#nj = dico['nj']

#for j in np.arange(0, N):
    #sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
    #estimate[j] = sortie['Zeta'] / 2. #normalement Zeta
    #aest[j]  = sortie['aest']

########################################################################

#f = lambda x, lbda: penalized.loss_l2_penalization_on_grad(x, aest[mask.ravel()],
                    #Elog[mask.ravel()], Varlog[mask.ravel()], nj, j1, j2, mask, l=lbda)
##We set epsilon to 0
#g = lambda x, lbda: penalized.grad_loss_l2_penalization_on_grad(x, aest[mask.ravel()],
                    #Elog[mask.ravel()], Varlog[mask.ravel()], nj, j1, j2, mask, l=lbda)

#l2_title = title_prefix + 'loss_l2_penalisation_on_grad'

#fg = lambda x, lbda, **kwargs: (f(x, lbda), g(x, lbda))
##For each lambda we use blgs algorithm to find the minimum
## We start from the
#l2_algo = lambda lbda: fmin_l_bfgs_b(lambda x: fg(x, lbda), estimate[mask.ravel()])

########################################################################

#j22 = np.min((j2, len(nj)))
#j1j2 = np.arange(j1 - 1, j22)
#njj = nj[j1j2]
#N = sum(njj)
#wvarjj = njj / N
#lipschitz_constant =  np.sum(8 * ((j1j2 + 1) ** 2) * wvarjj)
#l1_ratio = 0
#tv_algo = lambda lbda: penalized.mtvsolver(estimate[mask.ravel()], aest[mask.ravel()],
                                    #Elog[mask.ravel()], Varlog[mask.ravel()],
                                    #nj, j1, j2,mask,
                                    #lipschitz_constant=lipschitz_constant,
                                    #l1_ratio = l1_ratio, l=lbda)
#tv_title = title_prefix + 'wetvp'

########################################################################

#lmax = 15
#l2_minimizor = np.zeros((lmax,) + s.shape)
#l2_rmse = np.zeros(lmax)
#tv_minimizor = np.zeros((lmax,) + s.shape)
#tv_rmse = np.zeros(lmax)

#r = np.arange(lmax)
#lbda = np.array((0,) + tuple(1.5 ** r[:- 1]))

#for idx in r:
    #algo_min = l2_algo(lbda[idx])
    #l2_minimizor[idx] = _unmask(algo_min[0], mask)
    #l2_rmse[idx] = np.sqrt(np.mean((l2_minimizor[idx] - s) ** 2))

    #algo_min = tv_algo(lbda[idx])
    #tv_minimizor[idx] = _unmask(algo_min[0], mask)
    #tv_rmse[idx] = np.sqrt(np.mean((tv_minimizor[idx] - s) ** 2))

########################################################################

for idx_loop, (title, minimizor) in enumerate(zip([tv_title, l2_title],
                                [tv_minimizor, l2_minimizor])):

    plt.figure()
    plt.title(title)

    fig, axes = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
    fig2, axes2 = plt.subplots(nrows=3, ncols=int(ceil(lmax / 3.)))
    for idx, (dat, ax, ax2) in enumerate(zip(minimizor, axes.flat, axes2.flat)):
        im = ax.imshow(dat, norm=Normalize(vmin=np.min(minimizor),
                                        vmax=np.max(minimizor)), interpolation='nearest')
        ax.axis('off')
        im2 = ax2.imshow(dat, interpolation='nearest')
        ax2.axis('off')
        ax.set_title("l = %.1f " % (lbda[idx]))
        ax2.set_title("l= %.1f " % (lbda[idx]))

    cax = fig.add_axes([0.91, 0.1, 0.028, 0.8])
    fig.colorbar(im, cax=cax)
    cax2 = fig2.add_axes([0.91, 0.1, 0.028, 0.8])
    fig2.colorbar(im2, cax=cax2)

fig.savefig('/volatile/hubert/beamer/' + title + '_graph.pdf')

fig3 = plt.figure()
plt.plot(lbda, l2_rmse, 'r', label='l2 rmse')
plt.plot(lbda, tv_rmse, 'b', label='tv rmse')
plt.ylabel('rmse')
plt.xlabel('lambda')
plt.legend()

fig3.savefig('/volatile/hubert/beamer/' + title + '_rmse.pdf')
print title

plt.show()