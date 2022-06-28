from sklearn.utils import resample
from collections import Counter
import numpy as np

def undersampler(X, y):
    """ Undersamples the dominant class to the size of the second largest class
    Arguments:
        X:
        y: 1D sparse classification labels
    Returns
        X_downsampled:
        Y_downsampled:
    """
    X, y = np.array(X), np.array(y)
    
    class_counts = Counter(y)
    class_counts_List = sorted(list(class_counts.items()), key=lambda x:x[1])
    
    largest = class_counts_List[-1]
    secondLargest = class_counts_List[-2]
    
    class_instances = {k:[] for k in np.unique(y)}
    for i,r in enumerate(y):
        class_instances[r].append(i)
    
    resampledLargest = resample(class_instances[largest[0]], n_samples = secondLargest[1], random_state=101)
    
    X_downsampled = []
    Y_downsampled = []
    for class_ in class_instances.keys():
        if class_ != largest[0]:
            X_downsampled.extend(X[class_instances[class_]])
            Y_downsampled.extend([class_ for _ in class_instances[class_]])
        else:
            X_downsampled.extend(X[resampledLargest])
            Y_downsampled.extend([class_ for _ in range(secondLargest[1])])
    
    return np.array(X_downsampled), np.array(Y_downsampled)
