"""
A collection of data preprocessing and utility functions,
regarding the dataset images.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import imageio
import h5py

# Allow GPU memory growth to prevent crashes
from tensorflow.config.gpu import set_per_process_memory_growth
set_per_process_memory_growth(True)

PATH = 'img_align_celeba/*.jpg'

OUT_SIZE = 128
CHANNELS = 3

def make_gif():
    """
    Creating a gif from a collection of test images
    """
    png_dir = "./tests/"

    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('cvae.gif', images)


def get_image(path):
    """
    Getting an image from a given path
    """
    img = np.array(imageio.imread(path), dtype="float32")
    h, w = img.shape[:2]
    j = int(round((h - OUT_SIZE)/2.))
    i = int(round((w - OUT_SIZE)/2.))
    crop = img[j:j + OUT_SIZE, i:i + OUT_SIZE, :]
    img = np.resize(crop, (OUT_SIZE, OUT_SIZE, CHANNELS))
    img /= 255.
    return img


def merge(images, size):
    """
    Merge an array of images into one collage.
    Inputs: images - Array of images to merge
            size - A tuple of collage's width and height in images
    Output: The collage
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def imsave(images, size, path):
    """
    Saving an image.
    Inputs: images - Collection of images to save under one file as a collage
            size - A tuple of collage's width and height in images
            path - Location to save the file to
    """
    images *= 255
    imageio.imwrite(path, merge(images, size).astype("uint8"))


def plotScores(scores, test_scores, fname, on_top=True):
    """
    Creating a graph of the change in the loss value during training.
    Inputs: scores - A list of loss scores
            test_scores - A list of test scores
            fname - Name of file to save as
            on_top - Placement of legend
    """
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    plt.plot(scores)
    # plt.plot(test_scores)
    plt.xlabel('Epoch')
    plt.ylim([1000.0, 10000.0])
    loc = ('upper right' if on_top else 'lower right')
    plt.legend(['Train', 'Test'], loc=loc)
    plt.draw()
    plt.savefig(fname)


def progressbar(it, prefix="", file=sys.stdout):
    """
    Create a custom progress bar.
    Inputs: it - An iterable object
            prefix - An optional string at the beginning of the bar
            file - Output to save the bar to. Set by default to stdout, the console.
    """
    count = len(it)
    size = 60

    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" %
                   (prefix, "#" * x, "." * (size-x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield i
        if (i+1) % 100 == 0:
            show(i + 1)
    show(count)
    file.write("\n")
    file.flush()


def encode(enc_file, dataset, net):
    """
    Encoding the entire dataset to a single file. The encoded file contains the original
    latent vectors of every image in the dataset.
    Inputs: enc_file - The encoded file's name
            dataset - A collection of images to encode in the file
            net - The main model, creating the latent vectors.
    """
    print("[+] Dataset contains {} images".format(len(dataset)))

    with h5py.File(enc_file, 'a') as f:
        dset = f.create_dataset("data", (len(dataset), 256), 'float32')
        for i in progressbar(dataset, "[+] "):
            _, _, encoded = net.encoder.predict(
                np.array([get_image(dataset[i]), ]))
            dset[i] = encoded[0]

    print("[+] Encoding complete!")


def read(fname):
    with h5py.File(fname, 'r') as f:
        latent = f["data"][1]
        image = np.array(net.decoder.predict(np.array([latent, ])))[0]
