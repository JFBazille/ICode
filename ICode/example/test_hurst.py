# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:11:42 2015

@author: mr243268
"""
import os
import numpy as np
from ICode.loader import load_dynacomp
from ICode.estimators.hurst_estimator import Hurst_Estimator
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, ttest_1samp
from mne.stats import permutation_t_test
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.base import Bunch
from matplotlib import rc as changefont

dataset = load_dynacomp(preprocessing_folder='pipeline_1',
                        prefix='wr')
lr = LogisticRegression()
groups = [ ['av', 'v'], ['av', 'avn'], ['v', 'avn'] ]


def classify_group(group, fc):
    """Classification for a pair of groups
    """
    ind = np.hstack((dataset.group_indices[group[0]],
                    dataset.group_indices[group[1]]))
    #X =  fc[ind, :]
    X = StandardScaler().fit_transform([fc[i] for i in ind])
    y = np.array([1]* len(dataset.group_indices[group[0]]) +
                 [-1]* len(dataset.group_indices[group[1]]))
    sss = StratifiedShuffleSplit(y, n_iter=50, test_size=.25, random_state=42)
    
    accuracy = []; coef = []
    for train, test in sss:
        lr.fit(X[train], y[train])
        accuracy.append(lr.score(X[test], y[test]))
        coef.append(lr.coef_)
    return Bunch(accuracy=np.array(accuracy),
                 coef=np.array(coef))


def ttest_group(group, threshold, fc):
    """T-test
    """

    #n_rois = 22
    #threshold /= n_rois*(n_rois - 1)/2.

    fc_group_1 = [fc[i] for i in dataset.group_indices[group[0]]]
    fc_group_2 = [fc[i] for i in dataset.group_indices[group[1]]]
    tv, pv = ttest_ind(fc_group_1, fc_group_2)
    pv = -np.log10(pv)
    thresh_log = -np.log10(threshold)    
    ind_threshold = np.where(pv < thresh_log)
    pv[ind_threshold] = 0
    return pv


def ttest_onesample(group, threshold, fc):
    fc_group = [fc[i] for i in dataset.group_indices[group]]
    tv, pv = ttest_1samp(fc_group, 0.5)
    tv_p = tv > 0
    pv[tv_p] = 0.5 * pv[tv_p]
    tv_n = np.invert(tv_p)
    pv[tv_n] = 1 - 0.5 * pv[tv_n]
    pv = -np.log10(pv)
    thresh_log = -np.log10(threshold)    
    ind_threshold = np.where(pv < thresh_log)
    pv[ind_threshold] = 0  
    return pv


def ttest_onesample_Hmean(group, threshold, fc):
    fc_group = [fc[i] for i in dataset.group_indices[group]]
    fc_group_mean = np.mean(np.array(fc_group), axis=0)
    tv, pv = ttest_1samp(fc_group, 0.5)
    tv_p = tv > 0
    pv[tv_p] = 0.5 * pv[tv_p]
    tv_n = np.invert(tv_p)
    pv[tv_n] = 1 - 0.5 * pv[tv_n]
    pv = -np.log10(pv)
    thresh_log = -np.log10(threshold)    
    ind_threshold = np.where(pv < thresh_log)
    fc_group_mean[ind_threshold] = 0
    return fc_group_mean


def mne_permutation_ttest(group, threshold, fc, tail=1):
    fc_group_1 = [fc[i] for i in dataset.group_indices[group]]
    tv, pv, H0 = permutation_t_test(np.array(fc_group_1), tail=tail, n_jobs=3)
    pv = -np.log10(pv)
    thresh_log = -np.log10(threshold)    
    ind_threshold = np.where(pv < thresh_log)
    pv[ind_threshold] = 0
    return pv



def ttest_onesample_coef(coefs, threshold, fc):
    """T-test
    """
    
    tv, pv = ttest_1samp(coefs, 0.)
    tv_p = tv > 0
    pv[tv_p] = 0.5 * pv[tv_p]
    tv_n = np.invert(tv_p)
    pv[tv_n] = 1 - 0.5 * pv[tv_n]
    pv = -np.log10(pv)              
    thresh_log = -np.log10(threshold)
    #Locate unsignificant tests
    ind_threshold = np.where(pv < thresh_log)
    #and then threshold
    pv[ind_threshold] = 0

    cc = np.mean(coefs, axis=0)
    ind_threshold = np.where(pv < threshold)
    print 'pvalues :', ind_threshold
    cc[ind_threshold] = 0

    return cc


from joblib import Parallel, delayed



def compute_hurst_and_stat(metric='dfa', regu='off', OUTPUT_PATH = '/volatile/hubert/beamer/test_hurst/', plot=False):
    conn = Hurst_Estimator(metric=metric, mask=dataset.mask, regu=regu, n_jobs=5)
    os.write(1,'fit')
    fc = conn.fit(dataset.func1)
    os.write(1,'save')

    conn.save(save_path=OUTPUT_PATH)
    if plot:
        a = Parallel(n_jobs=3, verbose=5)(delayed(classify_group)(group, fc)
                                        for group in groups)

        tst = Parallel(n_jobs=3, verbose=5)(delayed(ttest_group)(group, .05, fc)
                                        for group in groups)

        ost = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample)(group, 0.05, fc)
                                            for group in ['v', 'av', 'avn'])

        mht = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample_Hmean)(group, 0.05, fc)
                                            for group in ['v', 'av', 'avn'])

        mpt = Parallel(n_jobs=3, verbose=5)(delayed(mne_permutation_ttest)(group,0.05, fc, 1)
                                            for group in ['v', 'av', 'avn'])
        
        
        cot = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample_coef)(np.reshape(coef['coef'], (coef['coef'].shape[0], coef['coef'].shape[-1])),
                                            0.05, fc)
                                            for coef in a)

        gr = ['v', 'av', 'avn']
        if regu=='off':
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric)
        else:
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric, regu)

        for i in range(3):
            title = '_'.join(groups[i])
            output_file = os.path.join(OUTPUT_PATH, title)
            img = conn.masker.inverse_transform(tst[i])
            plot_stat_map(img, cut_coords=(3, -63, 36), title=title, output_file=output_file + '.pdf')
            img = conn.masker.inverse_transform(cot[i])
            plot_stat_map(img, title='coef_map ' + title, output_file=output_file + 'coef_map.pdf')

            title = gr[i]
            output_file = os.path.join(OUTPUT_PATH, title)
            img = conn.masker.inverse_transform(ost[i])
            plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + '.pdf')
            img = conn.masker.inverse_transform(mht[i])
            plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + 'meanH.pdf')
            img = conn.masker.inverse_transform(mpt[i])
            plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + 'mnepermutH.pdf')


        plt.figure()
        plt.boxplot(map(lambda x: x['accuracy'], a))
        plt.savefig(os.path.join(OUTPUT_PATH, 'boxplot.pdf'))

def stat_computed_hurst(metric='dfa', regu='off', INPUT_PATH = '/volatile/hubert/beamer/test_hurst/', OUTPUT_PATH=''):
    conn = Hurst_Estimator(metric=metric, mask=dataset.mask, regu=regu, n_jobs=5)
    os.write(1,'load\n')
    conn.load_map(INPUT_PATH)
    fc = conn.hurst
    os.write(1,'stat\n')


    a = Parallel(n_jobs=3, verbose=5)(delayed(classify_group)(group, fc)
                                        for group in groups)
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    changefont('font', **font)

    plt.figure()
    plt.boxplot(map(lambda x: x['accuracy'], a))
    plt.xlabel(groups)
    plt.savefig(os.path.join(OUTPUT_PATH, metric+regu+'boxplot.pdf'))

def diff_computed_hurst(metric='wavelet', regu='off', INPUT_PATH = '/volatile/hubert/beamer/test_hurst/', OUTPUT_PATH=''):
    conn = Hurst_Estimator(metric=metric, mask=dataset.mask, regu=regu, n_jobs=5)
    os.write(1,'load\n')
    conn.load_map(INPUT_PATH)
    fc = conn.hurst
    os.write(1,'stat\n')

    tst = ttest_group(['av', 'v'], .05, fc)
    vmean_avmean = np.mean([fc[i] for i in dataset.group_indices['v']], axis=0) - np.mean([fc[i] for i in dataset.group_indices['av']], axis=0)
    vmean_avmean[tst == 0] = 0
    
    img = conn.masker.inverse_transform(vmean_avmean)
    plot_stat_map(img)
    plt.show()


