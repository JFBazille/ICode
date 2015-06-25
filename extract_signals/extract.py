import time
from nilearn.input_data import NiftiMasker

def extract_rois_signals(preprocessing_folder ='pipeline_2', prefix= 'resampled_wr'):
    dataset = load_dynacomp(preprocessing_folder = preprocessing_folder,prefix = prefix)
    for idx, func in enumerate([dataset.func1, dataset.func2]):
      for i in range(len(dataset.subjects)):
	tic = time.clock()
	print func[i]
	output_path, _ = os.path.split(func[i])
	print dataset.subjects[i]
	maps_img = dict_to_list(dataset.rois[i])
	#add mask, smoothing, filter and detrending
	print 'Nifti'
	masker = NiftiMapsMasker(maps_img=maps_img,
				mask_img = dataset.mask,
				low_pass = .1,
				high_pass = .01,
				smoothing_fwhm =6.,
				t_r = 1.05,
				detrend = True,
				standardize = False,
				resampling_target ='data',
				memory_level = 0,
				verbose=5)
	
	#extract signal to x
	print 'masker'
	x = masker.fit_transform(func[i])
	print x
	np.save(os.path.join('/volatile/hubert/datas/rois_filter','output' + str(i+1) +'_rois_filter'),x)
	
      print time.clock() - tic
      
      return x
    
def extract_one_signal(dataset):
    for idx, func in enumerate([dataset.func1, dataset.func2]):
      for i in range(len(dataset.subjects)):
	tic = time.clock()



	#maps_img = dict_to_list(func)
	#add mask, smoothing, filter and detrending
	maps_img = dict_to_list(dataset.rois[i])
	masker = NiftiMapsMasker(maps_img=maps_img,
				mask_img = dataset.mask,
				low_pass = .1,
				high_pass = .01,
				smoothing_fwhm =6.,
				t_r = 1.05,
				detrend = True,
				standardize = False,
				resampling_target ='data',
				memory_level = 0,
				verbose=5)
	
	#extract signal to x
	x = masker.fit_transform(func[i])
	
	print "loading time : "+ str(time.clock() - tic)
	return x,maps_img
      
def extract_one_vpv_signal(dataset):
    for idx, func in enumerate([dataset.func1, dataset.func2]):
      for i in range(len(dataset.subjects)):
	tic = time.clock()



	#maps_img = dict_to_list(func)
	#add mask, smoothing, filter and detrending
	masker = NiftiMasker(mask_img = dataset.mask,
				low_pass = .1,
				high_pass = .01,
				smoothing_fwhm =6.,
				t_r = 1.05,
				detrend = True,
				standardize = False,
				memory_level = 0,
				verbose=5)
	
	#extract signal to x
	x = masker.fit_transform(func[i])
	
	print "loading time : "+ str(time.clock() - tic)
	return x,masker
          
##This following function encapsulate the piece of code contained
#in the inner loop of extract_rois_one_signal

def extract_from_masker(dataset_like,dataset_mask, funci, timer =True):
  tic = time.clock()



  maps_img = dict_to_list(dataset_like)
  #add mask, smoothing, filter and detrending
  print 'Nifti'
  masker = NiftiMapsMasker(maps_img=maps_img,
			  mask_img = dataset_mask,
			  low_pass = .1,
			  high_pass = .01,
			  smoothing_fwhm =6.,
			  t_r = 1.05,
			  detrend = True,
			  standardize = False,
			  resampling_target ='data',
			  memory_level = 0,
			  verbose=5)

  #extract signal to x
  print 'masker'
  x = masker.fit_transform(funci)
  if timer:
    print time.clock() - tic
  return x
