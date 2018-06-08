import csv	
import os
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.switch_backend('agg') # need this to run this script over terminal


def read_img (source_path):
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = Image.open(current_path)
	image_array = np.asarray(image)
	return image_array

def add_img_meas(img, steering, imgs, steerings):
    # add the original image and measurement
	imgs.append(img)
	steerings.append(steering)
	# flipping image and steer measurement, and use them as in training and validation
	imgs.append(np.fliplr(img))
	steerings.append(-1 * steering)
	return imgs, steerings

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:

                # read the command
	            center_steering = float(batch_sample[3])
	            correction = 0.25
	            left_steering   = center_steering + correction
	            right_steering  = center_steering - correction

	            # read the images
	            center_img = read_img(batch_sample[0])
	            left_img   = read_img(batch_sample[1])
	            right_img  = read_img(batch_sample[2])
	            # add these images and steerings
	            images, measurements = add_img_meas(center_img, center_steering, images, measurements)
	            images, measurements = add_img_meas(left_img, left_steering, images, measurements)
	            images, measurements = add_img_meas(right_img, right_steering, images, measurements)


            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(x_train, y_train)

print('start to read file')
lines = []
# read the driving_log file
with open ('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('start to define neural network')

# define the neural network
from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda
from keras.layers import Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.utils.visualize_util import plot 

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
#model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3, activation = 'relu'))
#model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = \
            len(train_samples) * 6, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples) * 6, nb_epoch=5)


model.save('model.h5')


print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.grid()
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig ('training.png')
plt.show()

# plot the CNN architecture
plot(model, to_file='model.png')
