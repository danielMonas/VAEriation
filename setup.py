'''
Preparing any necessary files/data.
'''
from glob import glob
from os.path import isfile

from vae import CVAE
from PCA import *
from data_util import *

CHECKMARK_UNI = u'\u2713'

# Required files:
ENCODED_DATASET = 'data/encoded_dataset.hdf5'
FITTED_DATASET = 'data/dataset.npy'
EIGENVALUES = 'data/eigenvalues.npy'
EIGENVECTORS = 'data/eigenvectors.npy'
WEIGHTS = 'data/vae_weights.h5'
# Optional files needed for the live demo function
FACE_CASCADE  = 'haar/haarcascade_frontalface_default.xml'
MOUTH_CASCADE = 'haar/Mouth.xml'
EYE_CASCADE   = 'haar/haarcascade_eye.xml' 

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

# Searching for the optional cascade files
cascades = [EYE_CASCADE,MOUTH_CASCADE,FACE_CASCADE]
DEMO_FLAG = True
if not all([isfile(x) for x in cascades]):
    print('[!] Warning: Missing the following files:')
    print(*cascades, sep= ', ')
    print('[!] Disabling the live demo function...')
    DEMO_FLAG = False

print('[{}] All prerequisites found'.format(CHECKMARK_UNI))

