import numpy as np
from ICode.optimize.objective_functions import gradient, _unmask


def tv(img):
    """This function compute the total variation of an image
    """
    spatial_grad = gradient(img)
    return np.sum(np.sqrt(np.sum(spatial_grad[:-1] * spatial_grad[:-1],
                                    axis=0)))
    
def tv_for_masked_data(data, mask):
    """This function compute the total variation of an image
    """
    grad_data = gradient(_unmask(data))
    grad_mask = gradient(mask) != 0

    for i in range(len(grad_data)):
        grad_data[i][grad_mask[i]] = 0

    return np.sum(np.sqrt(np.dot(grad_data[:-1], grad_data[:-1])))

def grad_for_masked_data(data,mask):
    grad_data = gradient(_unmask(data, mask))
    grad_mask = gradient(mask) != 0
    
    for i in range(len(grad_data)):
        grad_data[i][grad_mask[i]] = 0
    return grad_data
