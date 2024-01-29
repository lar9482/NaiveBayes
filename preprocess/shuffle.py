import numpy as np

def shuffleDataset(X, Y):
    if (X.shape[0] != Y.shape[0]):
        raise RuntimeError('Number of samples need to be the same')
    
    numRows = X.shape[0]
    indices = np.arange(numRows)
    np.random.shuffle(indices)

    return (X.copy()[indices, :], Y.copy()[indices, :])