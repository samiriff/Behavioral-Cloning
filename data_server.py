import pandas as pd
import numpy as np
import shutil
import os
import glob

class DataProvider:
    def __init__(self, data_root):
        self.data_root = data_root
        self.rename_driving_log()
        self.column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        self.driving_log = pd.read_csv(data_root + '/orig_driving_log.csv', header=None,
                                      names=self.column_names)

    def rename_driving_log(self):
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
        print("Cleaning up paths")
        self.driving_log['center'] = self.driving_log['center'].apply(lambda path: '/'.join(path.split('\\')[-2:]))
        self.driving_log['left'] = self.driving_log['left'].apply(lambda path: '/'.join(path.split('\\')[-2:]))
        self.driving_log['right'] = self.driving_log['right'].apply(lambda path: '/'.join(path.split('\\')[-2:]))

    def prune(self, num_to_remove):
        print("Pruning")
        pruned_df, dropped_indices = self.get_pruned_df(num_to_remove)
        print("Dropped Indices Len = ", len(dropped_indices))
        print("Original Data Frame Len = ", len(self.driving_log))
        print("Pruned Data Frame Len = ", len(pruned_df))
        self.delete_dropped_images(self.driving_log.iloc[dropped_indices])
        self.driving_log = pruned_df

    def get_pruned_df(self, num_to_remove):
        drop_indices = np.random.choice(self.driving_log.index[self.driving_log['steering'] == 0], num_to_remove, replace=False)
        return self.driving_log.drop(drop_indices), drop_indices

    def delete_dropped_images(self, dropped_rows):
        print("Deleting images after pruning")
        dropped_rows['center'].apply(
            lambda img: self.delete_image(img))
        dropped_rows['left'].apply(
            lambda img: self.delete_image(img))
        dropped_rows['right'].apply(
            lambda img: self.delete_image(img))

    def delete_image(self, filename):
        path = self.data_root + '/' + filename
        if os.path.exists(path):
            print("Removing: ", path)
            os.remove(path)
            return True
        print('Not found: ', path)
        return False

    def write_driving_log(self):
        print("Writing CSV")
        self.driving_log.to_csv(self.data_root + '/driving_log.csv', index=False)

    def create_zip(self):
        print("Creating ZIP")
        shutil.make_archive(self.data_root + '/' + 'data', 'zip', self.data_root)

def startPythonServer():
    os.system('start cmd /c python -m http.server 5005')

def startNgrok():
    os.system('start cmd /c ngrok http 5005')

if __name__ == '__main__':
    data_root = '../new_data'
    prune_num_rows = 3800
    if os.path.exists(data_root + ' - Copy'):
        data_provider = DataProvider(data_root)
        data_provider.process_driving_log()
        data_provider.prune(prune_num_rows)
        data_provider.write_driving_log()
        data_provider.create_zip()
        startNgrok()
        startPythonServer()
    else:
        print("IMPORTANT: Please make a backup of this folder before running this program")



