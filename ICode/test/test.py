"""This function compute the DFA exponent of a signal,
it gives the same result as the matlab function
HDFAEstim of Biyu_code in schubert project
"""
#first we should use numpy cumsum to sum up the data array
if CumSum == 1:
    CSdata = np.cumsum(data, axis=-1)
else:
    CSdata = data
lendata = CSdata.shape[-1]
b2 = j2
if j2 > np.log2(lendata / 2):
    b2 = int(np.log2(lendata / 4))
scales = np.arange(j1, b2 + 1)
scales = 2 ** scales
n = lendata / scales
if data.ndim>1:
    F = np.zeros((len(scales), data.shape[0]))
else:
    F = np.zeros(len(scales))
for i in np.arange(0, len(scales)):
    breakpoints = np.arange(1, n[i]) * scales[i]
    data0 = signal.detrend(CSdata, bp=breakpoints)
    data0 = np.array_split(data0, breakpoints, axis=-1)
    data0 = np.array(data0)
    F[i] = np.mean(np.std(data0, axis=-1), axis=0)
    F[i] = F[i] / n[i]
tmp = np.polyfit(np.log2(scales), np.log2(F), deg=1)
alpha = tmp[0]-1