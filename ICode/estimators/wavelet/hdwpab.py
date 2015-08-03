"""These functions compute an Estimation of Hurst Exponent using a Wavelet
 implemented by P. Abry in matlab code, I let P. Abry comments
 this Python version yields results that are very cloth the the results yield
 by the original matlab function (see test_plagiate for a comparison and
 test_preg for a test of several regularity)
"""

import sys
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from math import gamma
from scipy.special import gammainc
from scipy.ndimage.filters import convolve1d as convolve

__all__ = ['hdw_p', 'wtspecq_statlog3', 'wtspecq_statlog32',
           'regrespond_det2', 'regrespond_det32']

def hdw_p(appro, nb_vanishmoment=2, norm=1, q=np.array(2), nbvoies=None, distn=1, wtype=1, j1=2, j2=8, printout=0):
    '''
    This function compute the Hurst exponent of a signal 'appro' using Discrete
    Wavelet transform

    appro: data to be analysed
       N: number of Vanishing Moments of Daubechies Wavelets
    norm: wavelet normalisation convention:  1 = L1,  2 = L2
 nbvoies: number of octaves
       q: vectors of q values
   distn: Hypothesis on the data distribution or `singal type':
              0: Gauss,  1: Non-G finite var, 2: AlphaStable or other
    '''
    n = len(appro)
    if nbvoies is None:
        nbvoies = int(np.log2(n))
    q = np.array(q)

    dico = wtspecq_statlog32(appro, nb_vanishmoment, norm, q, nbvoies, distn, printout)

    if q.ndim:
        for idx, moment in enumerate(q): 
            tmp = regrespond_det2(dico['Elogmuqj'][idx, 0], dico['Varlogmuqj'][idx, 0], moment,
                             dico['nj'], j1, j2, wtype)
    else:
        tmp = regrespond_det32(dico['Elogmuqj'][:,0], dico['Varlogmuqj'][:,0], q,
                             dico['nj'], j1, j2, wtype)
    
    
    return {'Elogmuqj': dico['Elogmuqj'], 'Varlogmuqj': dico['Varlogmuqj'], 'nj': dico['nj'],
            'logmuqj': dico['logmuqj'], 'Zeta': tmp['Zeta'], 'Vzeta': tmp['Vzeta'],
            'aest': tmp['aest'], 'Q': tmp['Q'], 'jj': tmp['jj']}



def wtspecq_statlog3(appro, N, norm, q, nbvoies, distn, printout):
    """
    This function is a Python Clone of the eponyme Matlab function implemented
    by P. Abry
    It computes E(log(muqj)) and Var for each level j and each degre q
    """
    l = list()
    loge = np.log2(np.e)
    #The format of q is important it has to be checked
    q = np.array(q)
    if q.ndim == 0:
        lenq = 1
        tq = True
        qctab = np.zeros(2)
    else:
        lenq = len(q)
        qctab = np.zeros(lenq)
    # Initialize the wavelet filters
    n = len(appro)  # data length
    hh1 = rlistcoefdaub(N)[::-1]  # scaling function filter
    nl = len(hh1)  # length of filter, store to manage edge effect later
    gg1 = ((- 1) ** (np.arange(0, nl)) * hh1)[::- 1]  # wavelet filter
    #--- Predict the max # of octaves available given N,
    #and take the min with what asked for
    nbvoies = min(int(np.log2(n / (2 * N + 1))), nbvoies)
    gjj = np.zeros(nbvoies)
    nj = np.zeros(nbvoies)
    Elogmuqj = np.zeros((lenq, nbvoies))
    Varlogmuqj = np.zeros((lenq, nbvoies))
    logmuqj = np.zeros((lenq, nbvoies))
    #--- Distn  dependent initialisation
    if distn == 0:
        #    GAUSSIEN
        #--- Load tabulated values of Expectation and variances for |N(0,1)|^q.
        #for Gaussian case
        #load varndq_gauss_tab
        loadeddata = scio.loadmat('ICode/estimators/wavelet/varndq_gauss_tab.mat')
        Nlim = np.reshape(loadeddata['Nlim'], 11)
        Ntab = np.reshape(loadeddata['Ntab'], 12)
        biaisgauss_tab = loadeddata['biaisgauss_tab']
        qtab = np.reshape(loadeddata['qtab'], 11)
        vargauss_tab = loadeddata['vargauss_tab']
        # q control:  those asked for not neccesarily
        # tabulated, generate substitutes qctab
        for  k in np.arange(0, lenq):
            if lenq == 1:
                minindex = np.argmin(abs(q - qtab))
                #qctab = qtab[minindex]
            else:
                minindex = np.argmin(abs(q[k] - qtab))
                # find index of closest q value
            qctab[k] = qtab[minindex]  # use that value
        if sum(q == qctab) != lenq and printout > 1:
            print "*** Warning, some q values aren''t tabulated,substituting:  original q were: \n"
            print q
            print 'Substitutes for tabulated part are:\n'
            print qctab
    elif  distn == 1:  # Not-Gaussian, but finite variance,
        #using asymptotic forms  ~C/nj
        #--- Choose estimation method for the constant ( C_4(j) + 2 )
        parametric = 0
        # Linear Regression for the smart estimate of C_4 at each j
        # using subblocks
        # if  ~parametric     or just do a direct point estimate of C_4(j)
        # fprintf('*** Reminder, doing direct estimation of C_4 rather than
        # regression  ***\n')
        # end

    #--- Compute the WT, calculate statistics
    njtemp = n  # this is the the appro, changes at each scale
    #-- Phase 1, get the coefficients at this scale
    for j in np.arange(0, nbvoies):
        convolue = np.convolve(appro, gg1)
        l.append(convolue)
        decime = convolue[nl:njtemp:2]
        # details at scale j. decime always becomes empty near the end,
        nj[j] = len(decime)  # number of coefficients
        # forget this j and exit if no coefficients left!
        if nj[j] == 0 and printout > 1:
            print 'In wtspecq_statlog, oops!  no details left at scale ' + str(j) + '\n'
            break

        #- Phase 2, calculate and store relevant statistics, do this for each q
        for  k in np.arange(0, lenq):
            # Loop on the values of q
            if lenq == 1:
                qc = q + 0
            else:
                qc = q[k]
            # the absolute powers are the new base quantities
            absdqk = abs(decime) ** qc
            # Calculate  log[ S_q(j) ]    ( in L1 language )
            logmuqj[k, j] = np.log2(np.mean(absdqk))

            # Now calculate the output quantities:  Elogmuqj  and  Varlogmuqj
            # GAUSSIEN
            if distn == 0:
                k0 = np.where(qctab[k] == qtab)[0][0]
                # find index of qctab in qtab (will exist)
                # if nj too small, need to use tabulation
                if  nj[j] < Nlim[k0] and printout > 1:
                    print '! Using tabulated values for |Gaussian|^q distribution:  q,j = %4.1f  %d \n'%(qc,j)
                    # choose nearest tabulated nj
                    minindex = abs(np.log2(nj[j]) - np.log2(Ntab)).argmin()
                    # grab precalculated values at nearest (q,j)
                    Varlogmuqj[k, j] = vargauss_tab[k0, minindex] / nj[j]
                    # what about scaling this to real var??
                    gj = biaisgauss_tab[k0, minindex] / nj[j]
                # Large nj(q): For each q,j:
                # Asymptotic iid based analytic expressions
                # V = 2( 1+C_4/2 ) = 2 + C_4 = [ E[|d|^2q] / (E[|d|^q)^2 ] - 1,
                # can calculate if d Gaussian
                else:
                    V = np.sqrt(np.pi) * gamma(qc + 1 / 2.) / gamma((qc +
                    1) / 2.) ** 2 - 1
                    Varlogmuqj[k, j] = loge ** 2 * V / nj[j]
                    gj = - loge / 2. * V / nj[j]
                gjj[j] = gj
                Elogmuqj[k, j] = logmuqj[k, j] - gj
            # Not-Gaussian, but finite variance
            elif distn == 1:
                # use a parametric model for bias and variance,
                #and fit from data
                if parametric:
                    # try to estimate by splitting into blocks
                    [LogEj, Varj] = logstat(absdqk, 0)
                    # this can happen, is set by logstat as a sign of
                    # not enough data
                    if  Varj == 0:
                        if j > 1:
                            # if no variance info at j,
                            #guess it is twice that at j-1
                            Varj = 2 * Varlogmuqj[k, j - 1]
                        # Very small signal, j-1 doesn't exist,
                        # have to estimate directly
                        else:
                            # estimate C_4 + 2 directly
                            Varj = loge ** 2 * (np.std(absdqk) ** 2 / mean(absdqk) ** 2 ) / nj[j]
                            # this is true, an estimate of the Var of the log
                            Varlogmuqj[k, j] = Varj
                            # Beware confusing notation, if bias corrected,
                            # this is log(E[muqj])...
                            Elogmuqj[k, j] = LogEj
                # Asymptotic formula with direct estimate of C_4 from data
                else:
                    # estimate  C_4 + 2  directly
                    V = np.std(absdqk) ** 2 / np.mean(absdqk) ** 2
                    # as before, asymptotic formulae
                    Varlogmuqj[k, j] = loge ** 2 * V / nj[j]
                    gj = - loge / 2 * V / nj[j]
                    # Modif PA --- le 11/01/2004 pour forcer gj = 0
                    # Elogmuqj(k,j) = logmuqj - gj ;
                    Elogmuqj[k, j] = logmuqj[k, j]
            # Infinite variance, Alpha stable or other
            else:
                print '! Distribution type 2 not implemented! \n'
            # Adjust for wavelet normalisation ,  affects the E term only
            # L1 normalisation, = 1/a
            if norm == 1:
                Elogmuqj[k, j] = Elogmuqj[k, j] - j * qc / 2.
                # else, L2 normalisation, = 1/root(a)

        #-- prepare for Phase 1 at next coarser scale
        del convolue
        del decime
        del absdqk
        convolue = np.convolve(appro, hh1)
        # new approximation, filtre meme taille que pour detail
        #=> memes effets de bord
        appro = convolue[nl:njtemp:2]
        njtemp = len(appro)
        del convolue

    return {'Elogmuqj': Elogmuqj, 'Varlogmuqj': Varlogmuqj, 'nj': nj,
            'logmuqj': logmuqj}


def wtspecq_statlog32(appro,N,norm,q,nbvoies,distn,printout) :
    """
    This function is a Python Clone of the eponyme Matlab function
    implemented by P. Abry,
    It computes E(log(muqj)) and Var for each level j and each degre q.
    It is the same function as wtspecq_statlog3 but appro can be an array
    of signal. If it is the case time should be on second dimension
    """

    #print 'distn' + str(distn)
    l = list()
    loge = np.log2(np.e)
    q = np.array(q)
    if q.ndim == 0:
        lenq = 1
        tq = True
        qctab = np.zeros(2)
    else:
        lenq = len(q)
        qctab = np.zeros(lenq)

    apshape = appro.shape
    multi = (len(apshape) > 1)
    #-- Initialize the wavelet filters
    if multi:
        n = apshape[1]
    else:
        n = len(appro)  # data length
    hh1 = rlistcoefdaub(N)[:: - 1]  # scaling function filter
    nl = len(hh1)  # length of filter, store to manage edge effect later
    gg1 = ((- 1) ** (np.arange(0, nl)) * hh1)[:: - 1]  # wavelet filter
    #--- Predict the max # of octaves available given N,
    #and take the min with what asked for
    # safer, casadestime having problems
    nbvoies = min(int(np.log2(n / (2 * N + 1))), nbvoies)

    gjj = np.zeros(nbvoies)
    nj = np.zeros(nbvoies)

    #if len(apshape) ==1:
        #Elogmuqj = np.zeros((lenq,nbvoies))
        #Varlogmuqj = np.zeros((lenq,nbvoies))
        #logmuqj = np.zeros((lenq,nbvoies))
    #else:
    Elogmuqj = np.zeros((apshape[0], lenq, nbvoies))
    Varlogmuqj = np.zeros((apshape[0], lenq, nbvoies))
    logmuqj = np.zeros((apshape[0], lenq, nbvoies))

    #--- Distn  dependent initialisation
    # GAUSSIEN
    if distn == 0:
        #--- Load tabulated values of Expectation and
        # variances for |N(0,1)|^q. for Gaussian case
        # load varndq_gauss_tab
        loadeddata = scio.loadmat('ICode/estimators/wavelet/varndq_gauss_tab.mat')
        Nlim = np.reshape(loadeddata['Nlim'], 11)
        Ntab = np.reshape(loadeddata['Ntab'], 12)
        biaisgauss_tab = loadeddata['biaisgauss_tab']
        qtab = np.reshape(loadeddata['qtab'], 11)
        vargauss_tab = loadeddata['vargauss_tab']

        # q control:  those asked for not neccesarily tabulated,
        #generate substitutes qctab
        for  k in np.arange(0, lenq):
            if lenq == 1:
                minindex = np.argmin(abs(q - qtab))
                #qctab = qtab[minindex]
            else:
                minindex = np.argmin(abs(q[k] - qtab))
                #find index of closest q value

            qctab[k] = qtab[minindex]  # use that value

        if sum(q == qctab) != lenq and printout > 1:
            print '*** Warning, some q values aren''t tabulated, substituting:  original q were: \n'
            print q
            print 'Substitutes for tabulated part are:\n'
            print qctab
    # Not-Gaussian, but finite variance,  using asymptotic forms  ~C/nj
    #--- Choose estimation method for the constant ( C_4(j) + 2 )
    elif  distn == 1:
        parametric = 0
    #--- Compute the WT, calculate statistics
    #  this is the the appro, changes at each scale
    njtemp = n
    #-- Phase 1, get the coefficients at this scale
    for j in np.arange(0, nbvoies):
        convolue = convolve(appro, gg1)
        l.append(convolue)
        # details at scale j. decime always becomes empty near the end,
        decime = convolue[:, nl:njtemp:2]
        # number of coefficients
        nj[j] = decime.shape[1]
        # forget this j and exit if no coefficients left!
        if nj[j] == 0 and printout > 1:
            print 'In wtspecq_statlog, oops!  no details left at scale ' + str(j)+'\n'
            break

        #-- Phase 2, calculate and store relevant statistics,
        #do this for each q
        #  Loop on the values of q
        for  k in np.arange(0, lenq):
            if lenq == 1:
                qc = q + 0
            else:
                qc = q[k]
            absdqk = abs(decime) ** qc
            # the absolute powers are the new base quantities
            logmuqj[:, k, j] = np.log2(np.mean(absdqk, axis=1))
            # Calculate  log[ S_q(j) ]    ( in L1 language )

            # Now calculate the output quantities:  Elogmuqj  and  Varlogmuqj
            # GAUSSIEN
            if distn == 0:
                # find index of qctab in qtab (will exist)
                k0 = np.where(qctab[k] == qtab)[0][0]
                # if nj too small, need to use tabulation
                if  nj[j] < Nlim[k0] and printout > 1:
                    print '! Using tabulated values for |Gaussian|^q distribution:  q,j = %4.1f  %d \n'%(qc,j)
                    # choose nearest tabulated nj
                    minindex = abs(np.log2(nj[j]) - np.log2(Ntab)).argmin()
                    # grab precalculated values at nearest (q,j)
                    Varlogmuqj[k, j] = vargauss_tab[k0, minindex] / nj[j]
                    # what about scaling this to real var??
                    gj = biaisgauss_tab[k0, minindex] / nj[j]
                # Large nj(q): For each q,j:
                # Asymptotic iid based analytic expressions
                # V = 2( 1+C_4/2 ) = 2 + C_4 = [ E[|d|^2q] / (E[|d|^q)^2 ] - 1,
                # can calculate if d Gaussian
                else:
                    V = np.sqrt(np.pi) * gamma(qc + 1 / 2.) / gamma((qc +
                    1) / 2.) ** 2 - 1
                    Varlogmuqj[:, k, j] = (loge ** 2 * V / nj[j]) * np.ones(apshape[0])
                    gj = - loge / 2. * V / nj[j]
                gjj[j] = gj
                Elogmuqj[:, k, j] = logmuqj[:, k, j] - gj
            # Not-Gaussian, but finite variance
            elif  distn == 1:
                # use a parametric model for bias and variance,
                # and fit from data
                if  parametric:
                    # try to estimate by splitting into blocks
                    [LogEj, Varj] = logstat(absdqk, 0)
                    # this can happen,
                    # is set by logstat as a sign of not enough data
                    if  Varj == 0:
                        # if no variance info at j,
                        # guess it is twice that at j-1
                        if j > 1:
                            Varj = 2 * Varlogmuqj[k, j - 1]
                        # Very small signal, j-1 doesn't exist,
                        # have to estimate directly
                        else:
                            # estimate C_4 + 2 directly
                            Varj = loge ** 2 * (np.std(absdqk, axis=1) ** 2 / mean(absdqk, axis=1) ** 2) / nj[j]
                            # this is true, an estimate of the Var of the log
                    Varlogmuqj[:, k, j] = Varj * np.ones(apshape[1])
                    # Beware confusing notation, if bias corrected,
                    # this is log(E[muqj])...
                    Elogmuqj[:, k, j] = LogEj
                    # Asymptotic formula with direct estimate of C_4 from data
                else:
                    V = np.std(absdqk, axis=1) ** 2 / np.mean(absdqk, axis=1) ** 2
                    #estimate  C_4 + 2  directly
                    # as before, asymptotic formulae
                    Varlogmuqj[:, k, j] = (loge ** 2 * V / nj[j]) * np.ones(apshape[0])
                    gj = - loge / 2 * V / nj[j]
                    # Modif PA --- le 11/01/2004 pour forcer gj = 0
                    # Elogmuqj(k,j) = logmuqj - gj ;
                    Elogmuqj[:, k, j] = logmuqj[:, k, j]

            # Infinite variance, Alpha stable or other
            else:
                print '! Distribution type 2 not implemented! \n'
            # Adjust for wavelet normalisation ,  affects the E term only
            # L1 normalisation, = 1/a
            if norm == 1:
                Elogmuqj[:, k, j] = Elogmuqj[:, k, j] - j * qc / 2.
                #% else, L2 normalisation, = 1/root(a)

        #-- prepare for Phase 1 at next coarser scale
        del convolue
        del decime
        del absdqk
        convolue = convolve(appro, hh1)
        # new approximation, filtre meme taille que pour detail
        # => memes effets de bord
        appro = convolue[:, nl:njtemp:2]
        njtemp = appro.shape[1]
        del convolue

    return {'Elogmuqj': Elogmuqj, 'Varlogmuqj': Varlogmuqj, 'nj': nj,
            'logmuqj': logmuqj}


#---------------------------------------------------------------------------
#Input:     q:  the order of the statistic used
#           N:  the number of vanishing moments of the wavelet
#               ((here only used in the plot title)
#          nj: the vector[1..scalemax] of the actual
#              number of coefficients averaged at octave j. (these
#              numbers are not powers of two because of the border
#              effects at each scale).
#             yj: the vector[1..scalemax] of log( sum |d(j)|^q ) - g(j) :
#                 -the bias term has already been removed prior to calling.
#             varyj: The variances of yj,  precalculated.
#             wtype: The kind of weighting used
#                     0 -  no weigthing  (ie uniform weights)
#                     1 -  1/nj weights  (suitable for fully Gaussian data)
#                     2 -  use variance estimates varj
#
#             j1:   the lower limit of the scales chosen,  1<= j1 <= scalemax-1
#             j2:   the upper limit of the octaves chosen, 2<= j2 <= scalemax
#       printout:  0:  batch mode, no output
#                  1:  printout plut log-log plot and
#                      regression line and +- 2*std(yj) around each point
#
#  Output:    zetaest:   estimate of the spectral parameter  zeta
#               Vzeta:   Exact variance of zetaest assuming given variances
#                        are correct
#                   Q:   A goodness of fit measure,  Chi2 based probability
#                        that null hyp is true given the
#                        observed data over the scale range  (j1,j2)


def regrespond_det2(yj, varyj, q, nj, j1, j2, wtype):
    scalemax = len(yj)
    j=np.arange(0, scalemax)
    #--- Check and clean up the inputted j1 and j2
    j1 = max(1, j1)  # make sure j1 is not too small
    j2 = max(j1 + 1, min(j2, scalemax))  # make sure j2 is not chosen too large

    #--- Convenience variables in the chosen scaling range
    jj = np.arange(j1 - 1, j2)

    J = len(jj)
    njj = nj[jj]
    yjj = yj[jj]
    varyjj = varyj[jj]
    # -- Calculate the weighted linear regression over  j1..j2

    #Notes
    # wtype = 1:  Here use     varyj ~ C/nj  to create the weights,
    #rather than use the varyj, to avoid
    # poor varyj estimation from generating bad weights.
    # This captures the essence of the weights,
    #  and the result is independant of C.
    #Thus no need to tell this function which "version"
    #  is being used (see wtspecq_statlog for more details)

    # define the effective varjj's used, depending on weight type
    if wtype == 0:
        # uniform weights
        wvarjj = np.ones(J)
        wstr = 'Uniform'
    elif wtype == 1:
        # Gaussian type weights
        wvarjj = 1. / njj
        wstr = 'Gaussian'
    elif  wtype == 2:
        # weights from data
        wvarjj = varyjj
        wstr = 'Estimated'
    else:
        #% all other cases
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, J)
        wstr = 'Uniform'
    #% use weighted regression formula in all cases

    S0 = sum(1. / wvarjj)
    S1 = sum((jj + 1) / wvarjj)
    S2 = sum((jj + 1) ** 2. / wvarjj)
    wjj = (S0 * (jj + 1) - S1) / wvarjj / (S0 * S2 - S1 * S1)
    vjj = (S2 - S1 * (jj + 1)) / wvarjj / (S0 * S2 - S1 * S1)

    #  Estimate  zeta
    zetaest = sum(wjj * yjj)
    # zeta is just the slope, unbiased regardless of the weights
    # intercept  'a'
    aest = sum(vjj * yjj)

    #Calculation of the variance of zetahat
    Vzeta = sum(varyjj * wjj * wjj)
    #this is exact if the varyjj are

    #--- Goodness of fit, based on inputted variances
    #   If at least 3 points, Apply a Chi-2 test , no level chosen,
    #should be viewed as a function of j1

    if J > 2:
        J2 = (J - 2) / 2.
        X = sum(((yjj - zetaest * (jj + 1) - aest) ** 2.) / varyjj)
        Q = 1 - gammainc(J2, X / 2.)
    else:
        print '\n***** Cannot calculate Q, need at least 3 points.\n'
        Q = 0

    return {'Zeta': zetaest, 'Vzeta': Vzeta, 'aest': aest, 'Q': Q, 'jj': jj}


def regrespond_det32(yj, varyj, q, nj, j1, j2, wtype):
    scalemax = yj.shape[-1]
    j = np.arange(0, scalemax)
    #--- Check and clean up the inputted j1 and j2
    j1 = max(1, j1)  # make sure j1 is not too small
    j2 = max(j1 + 1, min(j2, scalemax))  # make sure j2 is not chosen too large

    #--- Convenience variables in the chosen scaling range
    jj = np.arange(j1 - 1, j2)

    J = len(jj)
    njj = nj[jj]
    if yj.ndim==1:
        yjj = yj[jj]
        varyjj = varyj[jj]
    else:
        yjj = yj[:,jj]
        varyjj = varyj[:,jj]
    # -- Calculate the weighted linear regression over  j1..j2

    #Notes
    # wtype = 1:  Here use     varyj ~ C/nj  to create the weights,
    #rather than use the varyj, to avoid
    # poor varyj estimation from generating bad weights.
    # This captures the essence of the weights,
    #  and the result is independant of C.
    #Thus no need to tell this function which "version"
    #  is being used (see wtspecq_statlog for more details)

    # define the effective varjj's used, depending on weight type
    if wtype == 0:
        # uniform weights
        wvarjj = np.ones(J)
        wstr = 'Uniform'
    elif wtype == 1:
        # Gaussian type weights
        wvarjj = 1. / njj
        wstr = 'Gaussian'
    elif  wtype == 2:
        # weights from data
        wvarjj = varyjj
        wstr = 'Estimated'
    else:
        #% all other cases
        print '** Weight option not recognised, using uniform weights\n'
        wvarjj = np.ones(1, J)
        wstr = 'Uniform'
    #% use weighted regression formula in all cases

    S0 = sum(1. / wvarjj)
    S1 = sum((jj + 1) / wvarjj)
    S2 = sum((jj + 1) ** 2. / wvarjj)
    wjj = (S0 * (jj + 1) - S1) / wvarjj / (S0 * S2 - S1 * S1)
    vjj = (S2 - S1 * (jj + 1)) / wvarjj / (S0 * S2 - S1 * S1)

    #  Estimate  zeta
    zetaest = yjj.dot(wjj)
    # zeta is just the slope, unbiased regardless of the weights
    # intercept  'a'
    aest = yjj.dot(vjj)

    #Calculation of the variance of zetahat
    Vzeta = np.sum(varyjj * wjj * wjj, axis=-1)
    #this is exact if the varyjj are

    #--- Goodness of fit, based on inputted variances
    #   If at least 3 points, Apply a Chi-2 test , no level chosen,
    #should be viewed as a function of j1
    #I don't use that code it produces memory errors
    #if J > 2:
        #J2 = (J - 2) / 2.
        #Y = yjj - np.outer(jj + 1, zetaest).T - np.outer(aest, np.ones(len(jj)))
        #X = Y.dot(Y.T) / varyjj
        #Q = 1 - gammainc(J2, X / 2.)
    #else:
        #print '\n***** Cannot calculate Q, need at least 3 points.\n'
        #Q = 0
    Q = 0

    return {'Zeta': zetaest, 'Vzeta': Vzeta, 'aest': aest, 'Q': Q, 'jj': jj}


def rlistcoefdaub(regu):
    """
    This function return the coefficient of Daubechie Wavelets of regularity
    'regu' after it can be convolve with the signal
    """
    if not (regu < 11 and regu > 0):
        mess = [' Ondelette non implantee']
        print mess
        return
    else:
        if regu == 1:
            h = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
            # g=[1/sqrt(2) -1/sqrt(2)];

        if regu == 2:
            h = np.array([0.482962913145, 0.836516303738,
                          0.224143868042, -0.129409522551])

        if regu == 3:
            h = np.array([0.332670552950, 0.806891509311, 0.459877502118,
                          -0.135011020010, -0.085441273882, 0.035226291882])

        if regu == 4:
            h = np.array([0.230377813309, 0.714846570553, 0.630880767930,
                          -0.027983769417, -0.187034811719,
                          0.030841381836, 0.032883011667, -0.010597401785])

        if regu == 5:
            h = np.array([0.160102397974, 0.603829269797, 0.724308528438,
                          0.138428145901, -0.242294887066, -0.032244869585,
                          0.077571493840, -0.006241490213, -0.012580751999,
                          0.003335725285])

        if regu == 6:
            h = np.array([0.111540743350, 0.494623890398, 0.751133908021,
                          0.315250351709, -0.226264693965, -0.129766867567,
                          0.097501605587, 0.027522865530, -0.031582039318,
                          0.000553842201, 0.004777257511, -0.001077301085])

        if regu == 7:
            h = np.array([0.077852054085, 0.396539319482, 0.729132090846,
                          0.469782287405, -0.143906003929, -0.224036184994,
                          0.071309219267, 0.080612609151, -0.038029936935,
                          -0.016574541631, 0.012550998556, 0.000429577973,
                          -0.001801640704, 0.000353713800])

        if regu == 8:
            h = np.array([0.054415842243, 0.312871590914, 0.675630736297,
                          0.585354683654, -0.015829105256, -0.284015542962,
                          0.000472484574, 0.128747426620, -0.017369301002,
                         -0.044088253931, 0.013981027917, 0.008746094047,
                         -0.004870352993, -0.000391740373, 0.000675449406,
                         -0.000117476784])

        if regu == 9:
            h = np.array([0.038077947364, 0.243834674613, 0.604823123690,
                          0.657288078051, 0.133197385825, -0.293273783279,
                          -0.096840783223, 0.148540749338, 0.030725681479,
                          -0.067632829061, 0.000250947115, 0.022361662124,
                          -0.004723204758, -0.004281503682, 0.001847646883,
                          0.000230385764, -0.000251963189, 0.000039347320])

        if regu == 10:
            h = np.array([0.026670057901, 0.188176800078, 0.52720118932,
                          0.688459039454, 0.281172343661, -0.249846424327,
                         -0.195946274377, 0.127369340336, 0.093057364604,
                         -0.071394147166, -0.029457536822, 0.033212674059,
                          0.003606553567, -0.010733175483, 0.001395351747,
                          0.001992405295, -0.000685856695, -0.000116466855,
                          0.000093588670, -0.000013264203])

    return h

#------------------------------------------------------------------------------
#            logstat.m
#
#     Lyon, 22 Septembre 1999
#     P. Abry and D.Veitch
#
# DV, Melbourne  2 June 2001
#------------------------------------------------------------------------------
#  This routine takes stationary data X(i) of length n, and studies log(E[X])
#  and the variance of
#  Y_n = log[  X^(n)  ]  , where  X^(m)  is the mth level aggregation of X
#  (ie the process which averages  X  over blocks of size m).
#  Since only one sample value of  Y_n  is available, making
#  the variance impossible
#  to estimate directly,  the data is split into blocks of
#  different sizes m<=n, and the
#  mean and the variance of the variables  Y_m = log[ X^(m) ]
#  are studied as a function of m.
#  Typically  X = |dj|^q , so that Y_nj would correpond to  log[ S_q(j) ] .
#  ( The fitting here has nothing to do with scaling, as it all takes place at
#  constant j, rather we deal with the linear error in
#  approximating E[log] by log[E] etc ).
#
#  log(E[X]):  By assuming and fitting a linear relationship with m of
#  the E[Y_m], we can interpolate
#  to obtain an estimate as m -> infinity, which corresponds to log(E[X]),
#  the ultimate goal.  This is better than even a perfect estimate of
#  E[Y_n] as this is not equal to log(E[X]). If there is insufficient
#  data to split into blocks
#  a point estimate of E[Y_n] is returned instead.
#
#  Var[Y_n]:  again, we assume a parametric form:  Var[Y_n] ~ C/n  , and use
#  values m<n to estimate C.  The difference is that here the estimation is
#  must trickier and we must check adaptively to see
#  if we are close to convergence or not.
#
#
# ***  Usage:   [LogE,Var,Elog,ndecoupe] = logstat(X,printout)
#
#--- Routines called Directly :
#
#   Input:     X: input data over which estimation is to be performed
#                 ( typically X = |dj|^q )
#       printout: 1:  output some diagnostics, else output nothing except
#                     warnings
#
#  Output:   LogE:  Estimated  log(E[X]),
#                   basic quantities satisfying linear regression
#	      Var:  Estimated  Var[Y_n],   variances for linear regression
#            Elog:  Estimated    E[Y_n],   for interest,
#                   could determine the bias term
#        ndecoupe:  # of different block sizes used  (apart from the full data)
#
#------------------------------------------------------------------------------


def  logstat(X, printout):
    n = len(X)
    conv_cutoff = 6
    #  don't cut into more than 2^6 blocks , estimate becomes bad.
    ndecoupe = min(conv_cutoff, int(np.log2(n / 16.)))
    nk = np.zeros(ndecoupe + 1)
    m = np.zeros(ndecoupe + 1)
    moy = np.zeros(ndecoupe + 1)
    var = np.zeros(ndecoupe + 1)
    #(Look at last diagnostic plot, this value seems to cut off well
    #away from convergence, so no need to go further
    sys.stdout.write(ndecoupe)
    #  Not enough data to split into blocks.
    if ndecoupe < 3:
        # Not enough data, use simple point estimate for E[Y_n]
        Elog = np.log2(np.mean(X))
        #  Can't do better in this case
        LogE = Elog
        #  Indicates to calling function that the variance is unestimated
        Var = 0
    else:
        # Collect basic stats for Y_m for a range of m
        # (sample mean and variance)
        for k in np.arange(0, ndecoupe + 1):
            # pick logarithmically spaced set of m values: m=n to n/2^ndecoupe
            blocksize = int(n / 2. ** (k - 1))
            # number of whole blocks (ie length of Y_m at this m)
            nk[k] = int(n / (blocksize * 1.))
            #   matrix:   blocksize * m
            XM = np.reshape(X[: nk[k] * blocksize], blocksize, nk[k])
            #    store aggregation level corresponding to k
            m[k] = blocksize
            #  perform aggregation, ie get average over each block  (column)
            means = np.mean(XM)
            #  check for nasty case, zeros over a full block!
            index = np.where(means == 0)[0][0]
            if len(index) > 0:
                means[index] = np.spacing(1)
                print 'In "logstat", oops! had to replace a large block of zeros in the series! \n'

            # log of means over a block, this is our series Y_m
            Y_m = np.log2(means)
            # Average of these, estimate of the    E[ Y_m ]
            moy[k] = np.mean(Y_m)
            # Var[ Y_m ]
            var[k] = np.var(Y_m)

        # Take care of  E[Y_m]
        #  make linear fit of  moy  versus 1/m
        p = np.polyfit(1. / m, moy, deg=1)
        #  return value from fit evaluated at actual data length, est-E[Y_n]
        Elog = np.polyval(p, 1. / n)
        # intercept gives prediction from the model,
        # corresponds to looking at est-E[Y_m]
        LogE = p[1]
        #  for larger and larger m,
        # until it converges to log(E[X]), the ultimate goal.

        # Now the variance    Recall:  k=(1,2,3) => m(k)=n*(1,1/2,1/4)
        #=> nk=n/m(k)=(1,2,4)
        # Test and use parametric hypothesis:   Var[Y_m] ~ C/m
        # these would all be C
        nvar = m * var
        # the first two have very few points, discard
        nvar = nvar[2:len(nvar)]
        nk = nk[2:]
        #  weighted average to fit C, weights according to the amount of data
        mm = sum(nvar * nk) / sum(nk)
        ss = np.std(nvar)
        # detect those which are too far from the weighted average
        index = np.where(abs(nvar - mm) < ss)[0]
        #try a simple mean instead -> is bad
        # mm1 = mean(nvar)
        # try a simple mean of the `safe' ones  (currently used)
        mm2 = np.mean(nvar[index])
        # apply the model to get  est-Var[Y_n],  using  C = mm2
        Var = mm2 / n

        # Diagnostics
        if (printout == 1 and ndecoupe == conv_cutoff):
            print m
            print nk
            print nvar
            print nvar[index]
            #    calculate resulting linear fit at original points
            fit = np.polyval(p, 1. / m)
            plt.figure(1)
            plt.clf
            plt.subplot(311)
            # plot actual, fit, and value returned
            plt.plot(1. / m, moy, 'rs-', 1. / m, fit, 'bo-', 1 / n, Elog, '*g')
            plt.subplot(312)
            # Look as a function of m to see
            # the convergence to  lim E[Y_m] = log(E[X])
            # m is logarithmical chosen, so rescale to look
            # (p(2) is the INtercept)
            plt.plot(np.log2(m), moy - p[2], 'o-')
            plt.subplot(313)
            # Check validity of  est-Var[Y_m] ~ C/m
            # hypothesis and estimation of C
            plt.plot(np.arange(3, ndecoupe + 1), nvar, 'bo-', np.arange(2,
                     ndecoupe + 1), mm, 'ro-', np.arange(2, ndecoupe
                     + 1), mm2, 'go-')
            #grid
    return {'LogE': LogE, 'Var': Var, 'Elog': Elog, 'ndecoupe': ndecoupe}
