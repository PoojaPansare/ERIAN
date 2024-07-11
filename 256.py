
import scipy
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
#from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot as plt
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dropout, Dense
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def define_discriminator(input_shape=(256, 256, 3)):
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    # Flatten layer
    model.add(Flatten())
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

test_discriminator = define_discriminator()
print(test_discriminator.summary())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.keras.initializers import RandomNormal

def define_generator(latent_dim):
    model = Sequential()
    # We will reshape input latent vector into 16x16 image as a starting point.
    n_nodes = 256 * 16 * 16  # 65536 nodes
    model.add(Dense(n_nodes, input_dim=latent_dim))  # Dense layer to work with 1D latent vector
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 16, 256)))  # Reshape to 16x16x256
    
    # Upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))  # 32x32x128
    model.add(LeakyReLU(alpha=0.2))
    
    # Upsample to 64x64
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))  # 64x64x64
    model.add(LeakyReLU(alpha=0.2))
    
    # # Upsample to 128x128
    # model.add(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))  # 128x128x32
    # model.add(LeakyReLU(alpha=0.2))
    
    # Upsample to 256x256
    model.add(Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))  # 256x256x16
    model.add(LeakyReLU(alpha=0.2))
    
    # # Upsample to 512x512
    # model.add(Conv2DTranspose(4, (4, 4), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))  # 512x512x8
    # model.add(LeakyReLU(alpha=0.2))

    # Upsample to 1024x1024
    model.add(Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02), activation='tanh'))  # 1024x1024x3
    
    return model



test_gen = define_generator(100)
print(test_gen.summary())

# define the combined generator and discriminator model, for updating the generator
#Discriminator is trained separately so here only generator will be trained by keeping
#the discriminator constant. 
def define_gan(generator, discriminator):
	discriminator.trainable = False  #Discriminator is trained separately. So set to not trainable.
	# connect generator and discriminator
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


# Load x_train from the .pkl file
def load_real_samples():
    with open('X_train_256.pkl', 'rb') as f:
        X = pickle.load(f)

    X =(X- 127.5) / 127.5 
    return X


#def generate_real_samples(train_generator, n_samples):
    # X_real = next(train_generator)
    # return X_real



def generate_real_samples(train_generator, n_samples):
	# choose random images
	ix = randint(0, train_generator.shape[0], n_samples)
	# select the random images and assign it to X
	X = train_generator[ix]
	# generate class labels and assign to y
	y = ones((n_samples, 1)) ##Label=1 indicating they are real
	return X, y

# generate n_samples number of latent vectors as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# use the generator to generate n fake examples, with class labels
#Supply the generator, latent_dim and number of samples as input.
#Use the above latent point generator to generate latent points. 
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict using generator to generate fake samples. 
	X = generator.predict(x_input)
	# Class labels will be 0 as these samples are fake. 
	y = zeros((n_samples, 1))  #Label=0 indicating they are fake
	return X, y

def save_imgs(epoch):
    # load model
    #model = load_model(r"C:\Users\upratham\GAN\cifar_generator_2epochs.h5") #Model trained for 100 epochs
    # generate images
    latent_points = generate_latent_points(100, 4)  #Latent dim and n_samples
    # generate images
    X = generator.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0

    import numpy as np
    X = (X*255).astype(np.uint8)

    # plot the result
    #show_plot(X, 5)
    #X.shape
    img=X[1]

    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(img)

    # Save the image as a PNG file

    # Save the image as a PNG file
    img.save("/media/titan/My Book/Pratham/generated_256/GCP_%d.png" % epoch)
    
    return


import numpy as np

def train(g_model, d_model, gan_model, train_generator, latent_dim, n_epochs=100, n_batch=32,save_interval=250):
    bat_per_epo = int(2500 / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            print(j)
            # Train the discriminator on real and fake images, separately (half batch each)
            X_real, y_real = generate_real_samples(train_generator, half_batch)
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real)
            
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)
            
            # Train the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
            
        if i % save_interval == 0:
            g_model.save("/media/titan/My Book/Pratham/generated_256/model.h5")
            save_imgs(epoch=i)
        
    g_model.save('/media/titan/My Book/Pratham/generated_256/new_last_epoch.h5')
    print("Model saved")



###################################################################
#Train the GAN

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = define_discriminator()
# create the generator
# generator = define_generator(latent_dim)gg

generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
train_generator = load_real_samples()
# train model
train(generator, discriminator, gan_model, train_generator, latent_dim,save_interval=500, n_epochs=25000, n_batch=256,)

################################################################################