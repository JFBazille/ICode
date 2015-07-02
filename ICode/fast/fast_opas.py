#! /usr/bin/python3.4
#fast has been implemented to test or regularisation
#	
#	on real datas
import matplotlib.pyplot as plt
from nilearn.region import signals_to_img_maps
from nilearn.plotting import plot_img, plot_stat_map
import os, time
import numpy as np
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiMasker
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import fmin_l_bfgs_b

from ICode.estimators.wavelet.hdwpab import *
import ICode.opas
from ICode.estimators import penalyzed
from ICode.loader import load_dynacomp
from ICode.extractsignals.extract import extract_one_vpv_signal
##Chargement des donnees prend du temps a voir si besoin
dataset = load_dynacomp(preprocessing_folder='pipeline_1', prefix='swr')

print 'chargement des donnees'
#on recupere notre signal et le maps_img qui va bien
[x, masker] = extract_one_vpv_signal(dataset)
x = x.T
j1 = 3
j2 = 6
wtype = 1
l = x.shape[1]
N = x.shape[0]
estimate = np.zeros(N)
aest = np.zeros(N)
dico = wtspecq_statlog32(x, 2, 1, np.array(2), int(np.log2(l)), 0, 0)
Elog = dico['Elogmuqj'][:, 0]
Varlog = dico['Varlogmuqj'][:, 0]
nj = dico['nj']

for j in np.arange(0, N):
    sortie = regrespond_det2(Elog[j], Varlog[j], 2, nj, j1, j2, wtype)
    estimate[j] = sortie['Zeta'] / 2.
    aest[j] = sortie['aest']

#on cree l'image Nifti avec notrnifti image to numpy.arraye masker
img = masker.inverse_transform(estimate)
#On recupere le tableau de donnees
datas = img.get_data()
shape = x.shape[0]
mask = masker.mask_img_.get_data() > 0
lam = 5
f = lambda x: penalyzed.jh2bis(x, shape, Elog, Varlog, nj, j1, j2, l=lam)
g = lambda x: penalyzed.gradjh2bis(x, shape, Elog, Varlog, nj, j1, j2, l=lam)
H = np.concatenate((estimate, aest))

fg = lambda x, **kwargs: (f(x), g(x))

monmin = fmin_l_bfgs_b(fg, H)
minimiseur = monmin[0][:monmin[0].shape[0] / 2]

#l'image obtenu en minimisant notre fonction JH2
img2 = masker.inverse_transform(minimiseur)
#l'image obtenu par convolution gaussienne
img3 = masker.inverse_transform(gaussian_filter(img2.get_data(), 1)[mask])

p = plot_stat_map(img)
p.title('computed Hurst exponent')

p = plot_stat_map(img2)
p.title('JH2 minimisation lambda = ' + str(lam))

plt.show()
