from tensorflow.config.gpu import set_per_process_memory_growth
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import h5py
import cv2

from setup import *
from vae import CVAE, N_LATENT
from face_aligner import align, get_face

# Window and frame size specifications
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 650
IMG_SIZE = 512

# Smaller # of scales improves performance
N_SCALES = N_LATENT // 4

# Set scale range
SCALE_RANGE = 3

# Allow GPU memory growth to prevent crashes
set_per_process_memory_growth(True)

# PrettyPrint np arrays
np.set_printoptions(suppress=True, precision=3)

net = CVAE()
net.vae.load_weights(WEIGHTS)

# Setup of model & data
dataset = h5py.File(ENCODED_DATASET, 'r')
data = dataset['data'][:]
dataset.close()
mean_data = data.mean(axis=0)

eigen_values = np.load(EIGENVALUES)
eigen_vectors = np.load(EIGENVECTORS)
eigenvector_inverses = np.linalg.pinv(eigen_vectors)


def calc_sliders(i):
    '''
    Calculate the importance a slider represents
    '''
    traits = data[i] - mean_data
    return np.matmul(traits, eigenvector_inverses) / eigen_values


if not isfile(FITTED_DATASET):
    print('[!] Performing initial calculations...')
    # Calculating the importance all sliders represent using the results of the PCA calculation
    fitted_data = np.asarray([calc_sliders(i)
                              for i in range(data.shape[0])]).astype(np.float32)
    np.save(FITTED_DATASET, fitted_data)
    print('[{}] Initial calculations complete on {}'.format(
        CHECKMARK_UNI, FITTED_DATASET))
else:
    fitted_data = np.load(FITTED_DATASET)
    print('[{}] Data loaded'.format(CHECKMARK_UNI))


def make_image(z):
    '''
    Passes a given latent vector through the decoder
    to produce a reconstructed image
    '''
    copy_vector = mean_data.copy()
    for i in range(N_LATENT):
        copy_vector += z[i] * eigen_values[i] * eigen_vectors[i]
    copy_vector = copy_vector.reshape((1, N_LATENT))

    img = net.decoder.predict(copy_vector)[0]
    img = Image.fromarray((img*255).astype('uint8'))
    return ImageTk.PhotoImage(image=img.resize((IMG_SIZE, IMG_SIZE)))


class GUI:
    def __init__(self, root):
        self.root = root
        # Start with a random celeb displayed
        self.dataset_index = np.random.randint(0, len(fitted_data))
        self.latent_vector = fitted_data[self.dataset_index]
        self.image = make_image(self.latent_vector)

        # Set up three main frames - Scrollbars, display, and buttons
        self.set_buttons()
        self.set_display()
        self.set_scales()
        self.update_scales()

    def set_scales(self):
        '''
        Generate the GUI's left panel, containing the scale controls for
        all values of a latent vector
        '''
        # Creating the canvas and overlapping frame structures
        canvas = tk.Canvas(self.root, width=300, height=WINDOW_HEIGHT)

        scrollbar = tk.Scrollbar(self.root, command=canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda event: canvas.configure(
            scrollregion=canvas.bbox('all')))

        frame = tk.Frame(canvas)

        canvas.create_window((0, 0), window=frame, anchor='nw')
        canvas.pack(side=tk.LEFT)

        # Generating all scale controls
        cols = 5
        rows = N_SCALES // cols
        if not N_SCALES % cols:
            rows += 1

        self.scales = []
        for i in range(N_SCALES):
            r = i // cols * 2
            c = i % cols

            tk.Label(frame, text='#{}'.format(i+1)).grid(row=r, column=c)
            curr = tk.Scale(frame, from_=SCALE_RANGE, to=-SCALE_RANGE, orient=tk.VERTICAL, resolution=0.01,
                            command=lambda value, id=i: self.update_vector(value, id))
            curr.grid(row=r+1, column=c)

            self.scales.append(curr)

    def set_display(self):
        '''
        Create the center panel, containing the display image
        '''
        frame = tk.Frame(self.root, width=IMG_SIZE, height=WINDOW_HEIGHT)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_panel = tk.Label(frame, image=self.image)
        self.image_panel.pack(side='bottom', fill='both', expand='yes')

    def set_buttons(self):
        '''
        Create right-hand panel, containing the reset buttons
        '''
        frame = tk.Frame(self.root, bg='yellow',
                         width=175, height=WINDOW_HEIGHT)
        frame.pack(side=tk.RIGHT)

        # Randomize button
        tk.Button(frame, text='RANDOMIZE', command=self.randomize).place(
            x=0, y=0, relwidth=1, relheight=0.15)

        # Random control slider
        self.random_slider = tk.Scale(
            frame, from_=0, to=SCALE_RANGE, resolution=0.01, orient=tk.HORIZONTAL, label="Randomness range")
        self.random_slider.place(relx=0, rely=0.15, relwidth=1, relheight=0.1)
        self.random_slider.set(SCALE_RANGE/2)

        # Dataset iteration
        tk.Button(frame, text='PREV', command=lambda: self.iterate_dataset(False)).place(
            relx=0, rely=0.3, relwidth=0.5, relheight=0.1)

        tk.Button(frame, text='NEXT', command=lambda: self.iterate_dataset(True)).place(
            relx=0.5, rely=0.3, relwidth=0.5, relheight=0.1)

        # Index display
        self.index_display = tk.Entry(frame, font=('Courier', 20))
        self.index_display.place(relx=0, rely=0.25, relwidth=0.8)
        self.index_display.insert(0, self.dataset_index)

        tk.Button(frame, text='Enter', command=self.change_index).place(
            relx=0.8, rely=0.25, relwidth=0.2, relheight=0.05)

        tk.Button(frame, text="DEMO", command=self.demo).place(
            relx=0, rely=0.4, relwidth=1, relheight=0.2)

        # Invert button
        tk.Button(frame, text='INVERT', command=self.invert).place(
            relx=0, rely=0.6, relwidth=1, relheight=0.2)

        # Reset button
        tk.Button(frame, text='RESET', command=lambda: self.set_values(
            0)).place(relx=0, rely=0.8, relwidth=1, relheight=0.2)

    def update_vector(self, val, i):
        '''
        Update the latent vector according to the user's changes.
        Inputs: val - new value of one of the scales
                i - the changed scale's id
        '''
        self.latent_vector[i] = val
        self.update_image()

    def update_image(self):
        '''
        Updates the currently displayed image 
        based on the saved latent vector
        '''
        self.image = make_image(self.latent_vector)
        self.image_panel.configure(image=self.image)
        self.image_panel.image = self.image

    def update_scales(self):
        '''
        Update the value of all scales
        to match the latent vector shown
        '''
        for i, scale in enumerate(self.scales):
            scale.set(self.latent_vector[i])

    def randomize(self):
        '''
        Randomize all scale values 
        to generate a random image.
        '''
        random_range = self.random_slider.get()
        for i in range(len(self.latent_vector)):
            self.latent_vector[i] = np.random.uniform(
                -random_range, random_range, 1)[0]
        self.update_scales()

    def invert(self):
        '''
        Invert all scale values
        '''
        self.latent_vector *= -1
        self.update_scales()

    def set_values(self, value):
        '''
        Set the same value to all scales
        '''
        for i in range(len(self.latent_vector)):
            self.latent_vector[i] = value
        self.update_scales()

    def iterate_dataset(self, oper):
        '''
        Moves forwards or backwards in the data and updates GUI & variables accordingly
        Input: oper - A flag indicating wether to move forward or backwards
        '''
        self.dataset_index += 1 if oper else -1
        self.dataset_index %= len(fitted_data)

        self.index_display.delete(0, tk.END)
        self.index_display.insert(0, self.dataset_index)
        self.latent_vector = fitted_data[self.dataset_index].copy()

        self.update_scales()

    def change_index(self):
        '''
        Get the entered dataset index & make it the current one
        '''
        new_index = self.dataset_index

        try:
            new_index = int(self.index_display.get())
        except ValueError:
            self.index_display.delete(0, tk.END)
            self.index_display.insert(0, self.dataset_index)
            print('[!] Invalid input entered')

        new_index = new_index % len(fitted_data)
        self.dataset_index = new_index
        self.latent_vector = fitted_data[self.dataset_index].copy()
        self.update_scales()

    def demo(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Press Any Key - ESC to exit")
        demo_image = 0
        while True:
            ret, frame = cam.read()
            cv2.imshow("Press Any Key - ESC to exit", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                break
            elif k != -1:
                #cv2.namedWindow("Captured - ESC to confirm")
                demo_image = get_face(frame)
                print(demo_image)
                if demo_image is not None:
                    cv2.imshow("Captured - ESC to confirm", demo_image)
        cam.release()
        cv2.destroyAllWindows()
        if demo_image is 0:
            return
        inp = np.resize(np.array([demo_image,]), 
            (1, OUT_SIZE, OUT_SIZE, CHANNELS))
        _,_,out = net.encoder.predict(inp)
        self.latent_vector = np.resize(out,self.latent_vector.shape)

        try:
            self.update_scales()
        except:
            print("[!] Unable to apply encoding")


if __name__ == '__main__':
    print('[!] Running GUI...')
    root = tk.Tk()
    root.title('Vaeriation')
    root.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))

    program = GUI(root)
    root.mainloop()

    print('[{}] Thank you for using Vaeriation!'.format(CHECKMARK_UNI))
