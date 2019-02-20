import pandas as pd
import numpy as np
import shutil
import os
import glob

class DataCleaner:
    """
    Cleans up the data recorded by the Simulator, and presents the data as a zip file that can be downloaded
    by the Udacity workspace while creating a model
    """

    def __init__(self, data_root):
        '''
        Constructor that initializes the data root folder path, renames the driving log before modifications are made
        and renames the headers
        :param data_root: Root folder containing the driving log and a folder of recorded images from simulator
        '''
        self.data_root = data_root
        self.rename_driving_log()
        self.column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        self.driving_log = pd.read_csv(data_root + '/orig_driving_log.csv', header=None,
                                      names=self.column_names)

    def rename_driving_log(self):
        '''
        Renames the driving log file before performing modifications in case a backup is required
        '''

        print("Renaming")
        is_rename_required = True
        for file in glob.glob(self.data_root + '/*.csv'):
            print(file)
            if 'orig_driving_log' in file:
                print("Rename not required")
                is_rename_required = False
        if is_rename_required:
            os.rename(self.data_root + '/driving_log.csv', self.data_root + '/orig_driving_log.csv')

    def process_driving_log(self):
        '''
        Converts the image paths in the driving log to relative paths, making it agnostic to the host machine
        '''

        print("Cleaning up paths")
        self.driving_log['center'] = self.driving_log['center'].apply(lambda path: '/'.join(path.split('\\')[-2:]))
        self.driving_log['left'] = self.driving_log['left'].apply(lambda path: '/'.join(path.split('\\')[-2:]))
        self.driving_log['right'] = self.driving_log['right'].apply(lambda path: '/'.join(path.split('\\')[-2:]))

    def prune(self, num_to_remove):
        '''
        Prunes away the given number of rows from the driving log, where each removed row has a steering value of 0
        :param num_to_remove: Number of rows with steering angle equal to 0 to be removed
        '''

        print("Pruning")
        pruned_df, dropped_indices = self.get_pruned_df(num_to_remove)
        print("Dropped Indices Len = ", len(dropped_indices))
        print("Original Data Frame Len = ", len(self.driving_log))
        print("Pruned Data Frame Len = ", len(pruned_df))
        self.delete_dropped_images(self.driving_log.iloc[dropped_indices])
        self.driving_log = pruned_df

    def get_pruned_df(self, num_to_remove):
        '''
        Drops the given number of rows picked at random, with steering angle = 0
        :param num_to_remove: Number of rows to remove
        :return: new data frame with the rows removed, indices of the dropped rows
        '''

        drop_indices = np.random.choice(self.driving_log.index[self.driving_log['steering'] == 0], num_to_remove, replace=False)
        return self.driving_log.drop(drop_indices), drop_indices

    def delete_dropped_images(self, dropped_rows):
        '''
        Deletes the actual image files corresponding to the dropped rows
        :param dropped_rows: data frame containing the path of images to be deleted
        '''

        print("Deleting images after pruning")
        dropped_rows['center'].apply(
            lambda img: self.delete_image(img))
        dropped_rows['left'].apply(
            lambda img: self.delete_image(img))
        dropped_rows['right'].apply(
            lambda img: self.delete_image(img))

    def delete_image(self, filename):
        '''
        Deletes an image with the given name, if found
        :param filename: path of image to be deleted
        :return: boolean indicating whether image was found or not
        '''

        path = self.data_root + '/' + filename
        if os.path.exists(path):
            print("Removing: ", path)
            os.remove(path)
            return True
        print('Not found: ', path)
        return False

    def write_driving_log(self):
        '''
        Writes the modified and cleaned driving log to data root as "driving_log.csv"
        '''

        print("Writing CSV")
        self.driving_log.to_csv(self.data_root + '/driving_log.csv', index=False)

    def create_zip(self):
        '''
        Creates a new zipped file - "data.zip", out of the data root folder, and saves it in the data root folder
        '''
        print("Creating ZIP")
        shutil.make_archive(self.data_root + '/' + 'data', 'zip', self.data_root)

def startPythonServer():
    '''
    Starts a python server in a new shell, to serve files from the project folder
    '''
    os.system('start cmd /c python -m http.server 5005')

def startNgrok():
    '''
    Exposes the Python server with a Public domain name that can be accessed in the Udacity workspace
    '''
    os.system('start cmd /c ngrok http 5005')

if __name__ == '__main__':
    data_root = '../new_data'
    prune_num_rows = 3800
    if os.path.exists(data_root + ' - Copy'):
        data_cleaner = DataCleaner(data_root)
        data_cleaner.process_driving_log()
        data_cleaner.prune(prune_num_rows)
        data_cleaner.write_driving_log()
        data_cleaner.create_zip()
        startNgrok()
        startPythonServer()
    else:
        print("IMPORTANT: Please make a backup of this folder before running this program")



