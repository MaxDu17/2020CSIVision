import numpy as np
from os import listdir
masterdirectory = "../datasets"
class DataParser:

    def __init__(self):
        self.datasetList = list()
        files = listdir(masterdirectory)
        for file in files:
            if file.find(".") < 0:
                self.datasetList.append(file)

        print(self.datasetList)

k = DataParser()
