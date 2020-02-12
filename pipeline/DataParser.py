import numpy as np
from os import listdir
from pipeline.ProjectUtility import Utility
masterdirectory = "../datasets"
tool = Utility()
class DataParser:

    def __init__(self):
        self.datasetList = list()
        files = listdir(masterdirectory)
        for file in files:
            if file.find(".") < 0:
                self.datasetList.append(file)

    def get_meta(self):
        return self.datasetList

    def getAmpArr(self, dataset):
        return tool.csv_file_to_amp(masterdirectory + "/" + dataset + "/" + dataset + "_amplitude.csv")


    def remove_gaps(self, data):
        newData = list()
        for frame in data:
            carrier = frame[6:60]
            carrier.extend(frame[65:124])
            carrier.extend(frame[133:])
            newData.append(carrier)

        return newData



    def first_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[6:60]
            newData.append(carrier)

        return newData

    def second_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[65:124]
            newData.append(carrier)

        return newData

    def third_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[133:]
            newData.append(carrier)

        return newData


    def normalize(self, data): #only run this after you have removed chunky bits
        for i in range(len(data)):
            min_ = min(data[i])
            max_ = max(data[i])
            for j in range(len(data[i])):
                data[i][j] = (data[i][j]- min_)/(max_-min_)




k = DataParser()
data = k.getAmpArr("BedroomWork")
print(data[1])
print(k.remove_gaps(data)[1])
