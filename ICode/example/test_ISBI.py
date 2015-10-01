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
import pickle


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

def stat_ttest_function(conn, prefix='', OUTPUT_PATH=None):
    fc = conn.hurst


    ost = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample)(group, 0.05, fc)
                                        for group in ['v', 'av', 'avn'])

    mht = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample_Hmean)(group, 0.05, fc)
                                        for group in ['v', 'av', 'avn'])
    
    
    gr = ['v', 'av', 'avn']
    if OUTPUT_PATH is None:
        for i in range(3):

            title = prefix + gr[i]
            try:
                img = conn.masker.inverse_transform(ost[i])
                plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title)

            except ValueError:
                print "problem with ost " + title
            try:
                img = conn.masker.inverse_transform(mht[i])
                plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title)

            except ValueError:
                print "problem with mht " + title
            
    else:
        for i in range(3):

            title = prefix + gr[i]
            output_file = os.path.join(OUTPUT_PATH, title)
            try:
                img = conn.masker.inverse_transform(ost[i])
                #plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + '.pdf')
                plot_stat_map(img, output_file= output_file + '.pdf')

            except ValueError:
                print "problem with ost " + title
            try:
                img = conn.masker.inverse_transform(mht[i])
                #plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + 'meanH.pdf')
                plot_stat_map(img, output_file= output_file + 'meanH.pdf')

            except ValueError:
                print "problem with mht " + title

def stat_function_tst(conn, prefix='', OUTPUT_PATH=None, threshold=0.05):
    fc = conn.hurst

    tst = Parallel(n_jobs=3, verbose=5)(delayed(ttest_group)(group, threshold, fc)
                                    for group in groups)
    
    if OUTPUT_PATH is None:
        font = {'family' : 'normal',
            'size'   : 20}
        changefont('font', **font)
        gr = ['v', 'av', 'avn']
        for i in range(3):
            title = prefix + '_'.join(groups[i])
            try:
                img = conn.masker.inverse_transform(tst[i])
                print title
                plot_stat_map(img, cut_coords=(3, -63, 36))
                plt.show()

            except ValueError:
                print "problem with tst " + title
        changefont.func_defaults
            
    else:
        for i in range(3):
            title = prefix + '_'.join(groups[i])
            output_file = os.path.join(OUTPUT_PATH, title)
            try:
                img = conn.masker.inverse_transform(tst[i])
                plot_stat_map(img, cut_coords=(3, -63, 36), output_file=output_file + '.pdf')
            except ValueError:
                print "problem with tst " + title


def save_stat(a, save_path='/volatile/hubert/beamer/HEES0/stat',
             save_file='stat'):
        save_file = os.path.join(save_path, save_file)
        with open(save_file,'wb') as myfile:
            monpickler = pickle.Pickler(myfile)
            monpickler.dump(a)

def stat_function(conn, prefix='', OUTPUT_PATH=None):
    fc = conn.hurst

    a = Parallel(n_jobs=3, verbose=5)(delayed(classify_group)(group, fc)
                                        for group in groups)
    #tst = Parallel(n_jobs=3, verbose=5)(delayed(ttest_group)(group, .05, fc)
                                    #for group in groups)

    #save_stat({'a':a, 'tst': tst}, save_file=prefix)
    ##ost = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample)(group, 0.05, fc)
                                        ##for group in ['v', 'av', 'avn'])

    ##mht = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample_Hmean)(group, 0.05, fc)
                                        ##for group in ['v', 'av', 'avn'])

    ##mpt = Parallel(n_jobs=3, verbose=5)(delayed(mne_permutation_ttest)(group,0.05, fc, 1)
                                        ##for group in ['v', 'av', 'avn'])
    
    
    ##cot = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample_coef)(np.reshape(coef['coef'], (coef['coef'].shape[0], coef['coef'].shape[-1])),
                                        ##0.05, fc)
                                        ##for coef in a)
    #gr = ['v', 'av', 'avn']
    if OUTPUT_PATH is None:
        for i in range(3):
            title = prefix + '_'.join(groups[i])
            #try:
                #img = conn.masker.inverse_transform(tst[i])
                #plot_stat_map(img, cut_coords=(3, -63, 36), title=title)

            #except ValueError:
                #print "problem with tst " + title
            ##try:
                ##img = conn.masker.inverse_transform(cot[i])
                ##plot_stat_map(img, title='coef_map ' + title)

            ##except ValueError:
                ##print "problem with cot " + title

            ##title = prefix + gr[i]
            ##try:
                ##img = conn.masker.inverse_transform(ost[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title)

            ##except ValueError:
                ##print "problem with ost " + title
            ##try:
                ##img = conn.masker.inverse_transform(mht[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title)

            ##except ValueError:
                ##print "problem with mht " + title
            ##try:
                ##img = conn.masker.inverse_transform(mpt[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title)

            ##except ValueError:
                ##print "problem with mpt " + title
    else:
        #for i in range(3):
            #title = prefix + '_'.join(groups[i])
            #output_file = os.path.join(OUTPUT_PATH, title)
            #try:
                #img = conn.masker.inverse_transform(tst[i])
                #plot_stat_map(img, cut_coords=(3, -63, 36), title=title, output_file=output_file + '.pdf')

            #except ValueError:
                #print "problem with tst " + title
            ##try:
                ##img = conn.masker.inverse_transform(cot[i])
                ##plot_stat_map(img, title='coef_map ' + title, output_file=output_file + 'coef_map.pdf')

            ##except ValueError:
                ##print "problem with cot " + title

            ##title = prefix + gr[i]
            ##output_file = os.path.join(OUTPUT_PATH, title)
            ##try:
                ##img = conn.masker.inverse_transform(ost[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + '.pdf')

            ##except ValueError:
                ##print "problem with ost " + title
            ##try:
                ##img = conn.masker.inverse_transform(mht[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + 'meanH.pdf')

            ##except ValueError:
                ##print "problem with mht " + title
            ##try:
                ##img = conn.masker.inverse_transform(mpt[i])
                ##plot_stat_map(img, title='t-test H0 : H = 0.5 pvalue in -log10 scale groupe : ' + title, output_file= output_file + 'mnepermutH.pdf')

            ##except ValueError:
                ##print "problem with mpt " + title
        font = {'family' : 'normal',
            'size'   : 12}
        changefont('font', **font)
        plt.figure()
        box = plt.boxplot(map(lambda x: x['accuracy'], a))
        #for i,line in enumerate(box['medians']):# get position data for median line
            #x, y = line.get_xydata()[1] # top of median line
            ## overlay median value
            #plt.text(x + 0.1, y - 0.02, '%.2f\n%.2e' % (np.mean(a[i]['accuracy']),
                                                #np.var(a[i]['accuracy'])),
            #horizontalalignment='center') # draw above, centered

        plt.ylim(0.1,1)
        plt.xticks([1,2,3], ['AV-V\n $\mu =$ %.2f\n $\sigma^2 = $%.2e' %(np.mean(a[0]['accuracy'],np.var(a[0]['accuracy'])))
                            , 'AV-AVn\n$\mu =$ %.2f\n $\sigma^2 = $%.2e' %(np.mean(a[1]['accuracy'],np.var(a[1]['accuracy'])))
                            , 'V-AVn\n$\mu =$ %.2f\n $\sigma^2 = $%.2e' %(np.mean(a[2]['accuracy'],np.var(a[2]['accuracy'])))])
        plt.savefig(os.path.join(OUTPUT_PATH, prefix+'boxplot.pdf'))
        changefont.func_defaults


def compute_hurst_and_stat(metric='wavelet', regu='off', lbda = 1, OUTPUT_PATH = '/volatile/hubert/beamer/HEES0/', plot=False):
    conn = Hurst_Estimator(metric=metric, lbda = lbda, mask=dataset.mask,smoothing_fwhm=0, regu=regu, n_jobs=5)
    os.write(1,'fit\n')
    fc = conn.fit(dataset.func1)
    #conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_l2' +str(lbda))
    fc = conn.hurst
    os.write(1,'save\n')
    #stat_function_tst(conn, metric+' '+regu+' ', OUTPUT_PATH)
    conn.save(save_path=OUTPUT_PATH)
    if plot:
        os.write(1,'plot\n')
        if regu=='off':
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric)
        else:
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric, regu)
        stat_function(conn, prefix = 'lbda' + str(lbda), OUTPUT_PATH=OUTPUT_PATH)
    return conn

def stat(metric='wavelet', regu='off', lbda = 1, OUTPUT_PATH = '/volatile/hubert/beamer/HEES0/'):
    conn = Hurst_Estimator(metric=metric, lbda = lbda, mask=dataset.mask,smoothing_fwhm=0, regu=regu, n_jobs=5)
    os.write(1,'load\n')
    conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_tv' +str(lbda))

    os.write(1,'stat\n')
    if regu=='off':
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric)
    else:
            OUTPUT_PATH = os.path.join(OUTPUT_PATH, metric, regu)
    stat_function(conn, prefix = 'lbda' + str(lbda), OUTPUT_PATH=OUTPUT_PATH)
    return conn

def comparison(lbdatv = [0.5,1,2,3,4,5,10,15], lbdal2=[5,10,15,20,25,40], OUTPUT_PATH='/volatile/hubert/beamer/HEES0/'):
    conn = Hurst_Estimator(mask=dataset.mask)
    for lbda in lbdatv:
        os.write(1,'tv' +str(lbda)+'\n')

        conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_tv' +str(lbda))
        stat_function(conn, prefix = 'lbda' + str(lbda), OUTPUT_PATH=os.path.join(OUTPUT_PATH, 'wavelet', 'tv'))
        #fc = conn.hurst[0]
        #fc[fc==0.4] = 0
        #img = conn.masker.inverse_transform(fc)
        #lbda = '_'.join(str(lbda).split('.'))
        #plot_stat_map(img, output_file=os.path.join(OUTPUT_PATH, 'wavelet', 'tv'+lbda))

    for lbda in lbdal2:
        os.write(1,'l2' +str(lbda)+'\n')
        conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_l2' +str(lbda))
        stat_function(conn, prefix = 'lbda' + str(lbda), OUTPUT_PATH=os.path.join(OUTPUT_PATH, 'wavelet', 'l2'))
        #fc = conn.hurst[0]
        #fc[fc==0.4] = 0
        #img = conn.masker.inverse_transform(fc)
        #plot_stat_map(img, output_file=os.path.join(OUTPUT_PATH, 'wavelet', 'l2'+str(lbda)))

    conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_off')
    stat_function(conn, prefix = 'regu_off', OUTPUT_PATH=os.path.join(OUTPUT_PATH, 'wavelet'))
    #fc = conn.hurst[0]
    #fc[fc==0.4] = 0
    #img = conn.masker.inverse_transform(fc)
    #plot_stat_map(img, output_file=os.path.join(OUTPUT_PATH, 'wavelet', 'regu_off'))
    
    conn.load_map(INPUT_PATH = OUTPUT_PATH, save_file= 'hurstmap_metric_wavelet_regu_off_presmooth6')
    stat_function(conn, prefix = 'presmooth', OUTPUT_PATH=os.path.join(OUTPUT_PATH, 'wavelet'))
    #fc = conn.hurst[0]
    #img = conn.masker.inverse_transform(fc)
    #plot_stat_map(img, output_file=os.path.join(OUTPUT_PATH, 'wavelet', 'presmooth'))
    
    plt.show()


