import numpy as np
from os import listdir
from pipeline.ProjectUtility import Utility
masterdirectory = "../datasets"
tool = Utility()
class DataParser:

    def __init__(self):
        self.datasetList = list()
        self.amparr = list()

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
        return data

    def load_data(self, datafile): #must call at first
        self.amparr = self.getAmpArr(datafile)

    def get_data(self, start, end, chunkIdentifier):
        if len(self.amparr) == 0:
            raise Exception("You did not load any data") #constructive error feedback
        if start > len(self.amparr) or end > len(self.amparr):
            raise Exception("You overshot on your array access")

        carrier = self.amparr[start:end]
        if chunkIdentifier == 1:
            return self.first_chunk(carrier)
        elif chunkIdentifier == 2:
            return self.second_chunk(carrier)
        elif chunkIdentifier == 3:
            return self.third_chunk(carrier)
        elif chunkIdentifier == 4:
            return self.remove_gaps(carrier)
        elif chunkIdentifier == 0:
            return carrier
        else:
            raise Exception("Invalid chunkIdentifier")

    def get_square_data(self, start, chunkIdentifier):
        size = 0
        if chunkIdentifier == 1:
            size = 54
        elif chunkIdentifier == 2:
            size = 59
        elif chunkIdentifier == 3:
            size = 59
        elif chunkIdentifier == 4:
            size = 172
        elif chunkIdentifier == 0:
            size = 192

        return self.get_data(start, start + size, chunkIdentifier)


k = DataParser()
k.load_data("BedroomWork")
print(np.shape(k.get_square_data(0, 3)))

