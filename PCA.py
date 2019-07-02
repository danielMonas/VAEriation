
import numpy as np
from sklearn.decomposition import PCA
import h5py

def pca(enc):
    """
    Uses the PCA algorithm to sort the latent vectors by importance of each feature.
    Input: filename - The encoded dataset file name
    """
    dataset = h5py.File(enc, 'r')
    data = dataset['data'][:]
    dataset.close()

    normalized = data - np.mean(data, axis=0)
    pca = PCA(n_components=normalized.shape[1])
    pca.fit(normalized)
    values = np.sqrt(pca.explained_variance_)
    vectors = pca.components_
    np.save('eigenvalues.npy', values)
    np.save('eigenvectors.npy', vectors)
