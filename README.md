# ICode
This Code is made to post-procees fMRI data using Hurst exponent
analysis. You can also use it to compute Hurst Exponent on any
Signal. To do so you should just have a look to the file Estimator.

test file will give you a lot of example about the way to use 
Estimators function.

The file Optimize contains some usefull function to compute
gradient and laplacian and to derivate a norm l2 of a gradient
(hflap) it also works with masked data (make sure that your mask
is a boolean array)

If you want to use your code on real data (typically DYNACOMP)
you should first add the PATH to your data in the file
ICode/loader/paths.pref