import argparse
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
from keras import optimizers
from keras import backend as K
from keras.models import load_model

class DataProcessor:
    def __init__(self, data_root, batch_size=128):
        self.data_root = data_root
        self.batch_size = batch_size
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

    def generator(self, samples, batch_size):
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
    def __init__(self, data_processor, output_model_path='./model.h5', learning_rate=0.001, epochs=5, input_model_path=None):
        self.data_processor = data_processor
        self.output_model_path = output_model_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_model_path = input_model_path
        if input_model_path is None:
            self.model = Sequential()
            self.build()
        else:
            self.model = self.restore_model()
            self.freeze_layers()

    def build(self):
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=self.data_processor.get_input_shape()))
        self.model.add(Cropping2D(cropping=((50, 20), (0, 0))))
        self.model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200))))
        self.model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        # self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid'))
        # self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
        self.model.add(MaxPooling2D())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))

    def get_model(self):
        return self.model

    def train(self):
        optimizer = optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.model.fit_generator(self.data_processor.get_train_generator(),
                                 steps_per_epoch=self.data_processor.get_training_steps_per_epoch(),
                                 validation_data=self.data_processor.get_validation_generator(),
                                 validation_steps=self.data_processor.get_validation_steps(),
                                 epochs=self.epochs, verbose=1)

    def save_model(self):
        self.model.save(self.output_model_path)

    def restore_model(self):
        return load_model(self.input_model_path)

    def freeze_layers(self):
        print("Freezing layers for fine tuning")
        for layer in self.model.layers[:-9]:
            layer.trainable = False
        for layer in self.model.layers[-9:]:
            layer.trainable = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to data.'
    )
    parser.add_argument(
        'transfer_learning_param',
        type=str,
        nargs='?',
        default='',
        help='Path of input model and output model for transfer learning'
    )
    args = parser.parse_args()

    batch_size = 128

    data_processor = DataProcessor(args.data_root + '/', batch_size=batch_size)

    self_driving_model = None
    if args.transfer_learning_param == '':
        print("Performing training from scratch")
        self_driving_model = SelfDrivingModel(data_processor)
    elif ',' not in args.transfer_learning_param:
        print("Performing training from scratch and saving model to given path")
        self_driving_model = SelfDrivingModel(data_processor, output_model_path=args.transfer_learning_param)
    else:
        print("Using pre-trained model")
        epochs = 5
        learning_rate = 0.001
        input_model_path, output_model_path= args.transfer_learning_param.split(',')
        self_driving_model = SelfDrivingModel(data_processor, learning_rate=learning_rate, epochs=epochs,
                                              input_model_path=input_model_path,
                                              output_model_path=output_model_path)


    print(self_driving_model.get_model().summary())
    self_driving_model.train()
    self_driving_model.save_model()