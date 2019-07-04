![Alt text](Images/VaeriationLogoTransparent.png?raw=true "Title")
#### Created by Daniel Monastirski & Tomer Ben David

## Description
This project is an implementation of a Variational Autoencoder (VAE), a generative model which utilizes data compression.
The model was trained on the CelebA dataset, containing over 200,000 images of over 10,000 celebrities from all over the world.

## Introduction - What is a VAE?
First, let's start with an Autoencoder (AE), the basis of a VAE model. An Autoencoder is a neural network that learns how to efficiently break down an image to it's essential features, and then to reconstruct it. The process is similiar to several image compression methods, as the original image is encoded into a much smaller representation. However, there is one major problem with an Autoencoder - It can only encode and decode the same images, making it quite a boring process.  
This is where the Variational Autoencoder comes into play - It is mostly identical to the regular autoencoder, but it has a twist. By introducing just a bit of randomness at the transition between encoding an image and reconstructing it, a VAE can generate entirely new images, based on a collection it received as input originally.  
In this case, the model was trained on 200,000 images of celebrities. Each image was originally 128x128 pixels, and the model translated each to an array of 256 numbers. This array is called the latent vector. The user can edit those numbers to "edit" the generated image.

## Project description
The interface was designed to showcase the model's capabilites. On the left hand side are labeled scrollbars, each representing a value from the latent vector. Changing these values affects the generated image, displayed in the center. On the right hand side are several unique functions:
+ Randomize: Assign random values to the latent vector, thus creating a random image. You can change the values' range, but note that larger values may cause the image to look unnatural or distorted.
+ Dataset indexer: You can load specific images from the original dataset, either by entering their index number or going through the dataset one image at a time. These images were reconstructed by the model, meaning they are not exactly the same as the actual images from the dataset.
+ Demo: A live demo, enabling the user to take a picture of their face via the webcam, and let the model reconstruct it and display the most similiar representation of the user in the dataset.
+ Invert: Changes the sign of every value of the latent vector. Doing so displays the mirror opposite of the original image, in terms of features. For an image of a blonde, smiling woman, the opposite will be a dark haired, serious man.
+ Reset: Changes all latent vector values to zero. The displayed image is the average photo of the dataset.
+ Interpolation: Stopping the transition between two images anywhere in between.

## Installation and dependencies
### Dependencies
+ Python 3.6 or later
+ `Numpy`
+ `Tensorflow 2.0`
+ `imageio`
+ `h5py`

To install the project and its dependencies, download the project files and run
`pip install -r requirements.txt`

To run the project: `python gui.py`

### Training
In case you wish to train the model by yourself or in case the file `data/vae_weights.h5` does not exist, you must run the following command: `python vae.py`. This will begin a new training session.  
**Note:** You must have the entire dataset files you wish to train the model on. The model was orignally trained on the CelebA dataset, however it is possible to train it on a different dataset, given a few tweaks to the code.

## Sources
- Altosaar, J. (n.d.). Tutorial - What is a variational autoencoder? – Jaan Altosaar. Retrieved from https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- Erdogan, G. (2017, August 8). Variational Autoencoder - University of Rochester. Retrieved from http://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheet/VariationalAutoEncoder.pdf
- Frans, K. (2017, May 5). Variational Autoencoders Explained. Retrieved from http://kvfrans.com/variational-autoencoders-explained/
- Jordan, J. (2018, July 16). Variational autoencoders. Retrieved from http://www.jeremyjordan.me/variational-autoencoders
- Ng, A. (n.d.). Machine Learning. Retrieved from http://www.coursera.org/learn/machine-learning
- Shafkat, I. (2018, February 4) Intuitively Understanding Variational Autoencoders Retrieved from https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
- Ziwei, L., Ping, L., Xiaogang, W., & Xiaoou, T. (n.d.). Retrieved from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
