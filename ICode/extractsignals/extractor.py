import os, time
import numpy as np
from nilearn.input_data import NiftiMasker

class signal_extractor():

    def __init__(self, dataset = None):
        self.dataset = dataset
        self.masker = NiftiMasker(mask_img = self.dataset.mask,
                                low_pass = .1,
                                high_pass = .01,
                                smoothing_fwhm =6.,
                                t_r = 1.05,
                                detrend = True,
                                standardize = False,
                                memory_level = 0,
                                verbose=5)


    def extract(self):
        for idx, func in enumerate([self.dataset.func1, self.dataset.func2]):
            #add mask, smoothing, filter and detrending


            for i in range(len(self.dataset.subjects)):
                tic = time.clock()
                #extract signal to x
                x = self.masker.fit_transform(func[i])
                print "loading time : "+ str(time.clock() - tic)
                yield x, self.masker