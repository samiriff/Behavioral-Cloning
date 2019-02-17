import pandas as pd
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

    def write_driving_log(self):
        print("Writing CSV")
        self.driving_log.to_csv(self.data_root + '/driving_log.csv', index=False)

    def create_zip(self):
        print("Creating ZIP")
        shutil.make_archive(self.data_root + '/' + 'data', 'zip', self.data_root)



def startNgrok():
    os.system('start cmd /c ngrok http 5005')


if __name__ == '__main__':
    data_provider = DataProvider('../new_data')
    data_provider.process_driving_log()
    data_provider.write_driving_log()
    data_provider.create_zip()
    #startNgrok()



