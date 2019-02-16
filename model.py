import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.image as mpimg
# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sys

class DataProcessor:
    def __init__(self, data_root):
        self.data_root = data_root
        self.batch_size = 32
        self.driving_log = pd.read_csv(data_root + 'driving_log.csv')
        self.train_samples, self.validation_samples = self.init_samples()
        self.train_generator = self.generator(self.train_samples, self.batch_size)
        self.validation_generator = self.generator(self.validation_samples, self.batch_size)

    def init_samples(self):
        correction = 0.2
        self.driving_log['steering_left'] = self.driving_log['steering'] + correction
        self.driving_log['steering_right'] = self.driving_log['steering'] - correction

        self.samples = np.concatenate((
            self.driving_log[['center', 'steering']].values,
            self.driving_log[['left', 'steering_left']].values,
            self.driving_log[['right', 'steering_right']].values
        ))

        return train_test_split(self.samples, test_size=0.2)

    def generator(self, samples, batch_size=32):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffled_samples = sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = shuffled_samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    # print("Batch=", batch_sample)
                    name = self.data_root + batch_sample[0].strip()
                    center_image = mpimg.imread(name)
                    center_angle = float(batch_sample[1])
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def get_train_generator(self):
        return self.train_generator

    def get_validation_generator(self):
        return self.validation_generator

    def get_input_shape(self):
        return (160, 320, 3)

    def get_batch_size(self):
        return self.batch_size

    def get_training_steps_per_epoch(self):
        return int(len(self.train_samples) / self.batch_size)

    def get_validation_steps(self):
        return int(len(self.validation_samples) / self.batch_size)



class SelfDrivingModel:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = Sequential()
        self.build()

    def build(self):
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=self.data_processor.get_input_shape()))
        self.model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid'))
        # self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='same'))
        self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        # self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))

    def get_model(self):
        return self.model

    def train(self):
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit_generator(self.data_processor.get_train_generator(),
                                 steps_per_epoch=self.data_processor.get_training_steps_per_epoch(),
                                 validation_data=self.data_processor.get_validation_generator(),
                                 validation_steps=self.data_processor.get_validation_steps(),
                                 epochs=5, verbose=1)

    def save_model(self):
        self.model.save('./model.h5')

if __name__ == '__main__':
    data_processor = DataProcessor(sys.argv[1] + '/')
    self_driving_model = SelfDrivingModel(data_processor)
    print(self_driving_model.get_model().summary())
    self_driving_model.train()
    self_driving_model.save()
