import zipfile
import os
from tqdm import tqdm
import sys

class DataExtractor:
    '''
    Used by the Udacity workspace to extract the contents of the zipped file provided by the DataCleaner
    '''

    def __init__(self, src, dst):
        '''
        Constructor that initializes the source zipped file that has to be extracted into the given destination
        :param src: Path of zipped file
        :param dst: Path where the contents of the zipped file will be extracted
        '''

        self.src = src
        self.dst = dst

    def unzip_data(self):
        '''
        Extracts the zipped file into the destination path, and displays progress using tqdm
        :return:
        '''

        zip = zipfile.ZipFile(self.src, 'r')
        nameList = zip.namelist()

        count = 0

        for item in tqdm(nameList, desc='Unzipping'):
            dir, file = os.path.split(item)
            if dir.find("__MACOSX") == -1:
                # print(count, item, "Dir=", dir, "File=", file)
                count += 1
                zip.extract(item, self.dst)
        zip.close()

if __name__ == '__main__':
    data_extractor = DataExtractor(sys.argv[1], sys.argv[2])
    data_extractor.unzip_data()