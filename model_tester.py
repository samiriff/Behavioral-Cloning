import argparse
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
# Setup Keras
from keras.models import load_model


class DataProcessor:
    def __init__(self, data_root, batch_size):
        self.data_root = data_root
        self.batch_size = batch_size
        self.driving_log = pd.read_csv(data_root + '/' + 'driving_log.csv')
        self.samples = self.init_samples()
        self.test_generator = self.generator(self.samples, self.data_root, self.batch_size)

    def init_samples(self):
        correction = 0.2
        self.driving_log['steering_left'] = self.driving_log['steering'] + correction
        self.driving_log['steering_right'] = self.driving_log['steering'] - correction

        return np.concatenate((
            self.driving_log[['center', 'steering']].values,
            self.driving_log[['left', 'steering_left']].values,
            self.driving_log[['right', 'steering_right']].values
        ))

    def generator(self, samples, data_root, batch_size=32):
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # print("Batch=", batch_sample)
                name = data_root + '/' + batch_sample[0].strip()
                center_image = mpimg.imread(name)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

    def get_test_generator(self):
        return self.test_generator

    def get_step_count(self):
        return len(self.samples) // self.batch_size


class ModelTester:
    def __init__(self, data_processor, model_path):
        self.data_processor = data_processor
        self.model = load_model(model_path)

    def evaluate(self):
        print("Evaluating Test set")
        self.metrics = self.model.evaluate_generator(self.data_processor.get_test_generator(),
                                                steps=self.data_processor.get_step_count())

    def print_metrics(self):
        print("Printing Metrics")
        for metric_i in range(len(self.model.metrics_names)):
            metric_name = self.model.metrics_names[metric_i]
            metric_value = self.metrics[metric_i]
            print('{}: {}'.format(metric_name, metric_value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to data.'
    )
    parser.add_argument(
        'model_path',
        type=str,
        nargs='?',
        default='',
        help='Path to trained model'
    )
    args = parser.parse_args()

    data_processor = DataProcessor(args.data_root, 128)
    model_tester = ModelTester(data_processor, args.model_path)
    model_tester.evaluate()
    model_tester.print_metrics()
