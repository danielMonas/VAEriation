'''
Preparing any necessary files/data.
'''
from glob import glob
from os.path import isfile

from vae import CVAE
from PCA import *
from data_util import *

CHECKMARK_UNI = u'\u2713'

print('[*] Checking prerequisites...')

if not isfile(WEIGHTS):
    print('[!] Error: Unable to continue - missing file ' + WEIGHTS)
    print(
        '[*] To generate the missing file, please train the model by running vae.py')
    exit()

# Encoding the entire dataset, if not already encoded
if not isfile(ENCODED_DATASET):
    print('[!] Warning: Missing file {}. Generating...'.format(ENCODED_DATASET))
    dataset = glob(PATH)
    if not dataset:
        print('[!] Error: Unable to encode dataset - missing files at '+ PATH)
        exit()
    else:
        net = CVAE()
        net.vae.load_weights(WEIGHTS)
        encode(ENCODED_DATASET, dataset, net)

# Generating the Eigen values and vectors, if they do not already exist.
if not isfile(EIGENVALUES) or not isfile(EIGENVECTORS):
    print('[!] Warning: Missing files {},{}. Generating...'.format(
        EIGENVALUES, EIGENVECTORS))
    pca(ENCODED_DATASET)

print('[{}] All prerequisites found'.format(CHECKMARK_UNI))

