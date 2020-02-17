import numpy as np
from os import listdir
from pipeline.ProjectUtility import Utility
masterdirectory = "../datasets"
class DataParser(Utility):

    def __init__(self):
        self.datasetList = list()
        self.amparr = list()
        self.masteramparr = {}

        files = listdir(masterdirectory)
        for file in files:
            if file.find(".") < 0:
                try:
                    self.datasetList.append(file)
                    self.masteramparr[file] =  self.getAmpArr(file)
                except:
                    print(file + " was empty. I skipped it")

    def get_meta(self):
        return self.datasetList

    def getAmpArr(self, dataset):
        return self.csv_file_to_amp(masterdirectory + "/" + dataset + "/" + dataset + "_amplitude.csv")

    def getMasterAmpArr(self):
        return self.masteramparr

    def get_size(self):
        if len(self.amparr) == 0:
            raise Exception("You did not load any data")  # constructive error feedback
        return len(self.amparr)

    def remove_gaps(self, data):
        newData = list()
        for frame in data:
            carrier = frame[6:32]
            carrier.extend(frame[33:59])
            carrier = self.normalize_single_column(carrier)

            carrier.extend(self.normalize_single_column(frame[66:123]))

            carrier.extend(self.normalize_single_column(frame[134:191]))

            newData.append(carrier)

        return newData


    def first_chunk(self, data):
        newData = list()
        for frame in data:
            #carrier = frame[6:60] #30
            carrier = frame[6:32]
            carrier.extend(frame[33:59])
            newData.append(carrier)

        return newData

    def second_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[65:124]
            carrier = frame[66:123]
            newData.append(carrier)

        return newData

    def third_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[133:]
            carrier = frame[134:191]
            newData.append(carrier)

        return newData

    def normalize_single_column(self, data):
        min_ = min(data)
        max_ = max(data)
        for j in range(len(data)):
            data[j] = (data[j] - min_) / (max_ - min_)

        return data

    def normalize(self, data): #only run this after you have removed chunky bits
        for i in range(len(data)):
            min_ = min(data[i])
            max_ = max(data[i])
            for j in range(len(data[i])):
                data[i][j] = (data[i][j]- min_)/(max_-min_)
        return data

    def load_data(self, datafile): #must call at first
        assert datafile in self.datasetList, "your file does not exist"
        try:
            self.amparr = self.masteramparr[datafile]
        except:
            raise Exception("Unable to load the filename specified")

    def get_data(self, start, end, chunkIdentifier):
        assert len(self.amparr) > 0, "you did not load any data"

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
            size = 52
        elif chunkIdentifier == 2:
            size = 59
            size = 57
        elif chunkIdentifier == 3:
            size = 59
            size = 57
        elif chunkIdentifier == 4:
            size = 166
        elif chunkIdentifier == 0:
            size = 192

        return self.get_data(start, start + size, chunkIdentifier)

    def get_square_data_norm(self, start, chunkIdentifier):
        return self.normalize(self.get_square_data(start, chunkIdentifier))

    def get_data_norm(self, start, end, chunkIdentifier):
        return self.normalize(self.get_data(start, end, chunkIdentifier))\



    def return_size(self, chunkIdentifier):
        size = 0
        if chunkIdentifier == 1:
            size = 52
        elif chunkIdentifier == 2:
            size = 57
        elif chunkIdentifier == 3:
            size = 57
        elif chunkIdentifier == 4:
            size = 166
        elif chunkIdentifier == 0:
            size = 192

        return size

    def return_chunkIdent_name(self, chunkIdentifier):
        if chunkIdentifier == "first":
            return 1
        elif chunkIdentifier == "second":
            return 2
        elif chunkIdentifier == "third":
            return 3
        elif chunkIdentifier == "all":
            return 4
        elif chunkIdentifier == "raw":
            return 0

    def return_size_name(self, chunkIdentifier):
        size = 0
        if chunkIdentifier == "first":
            size = 52
        elif chunkIdentifier == "second":
            size = 57
        elif chunkIdentifier == "third":
            size = 57
        elif chunkIdentifier == "all":
            size = 166
        elif chunkIdentifier == "raw":
            size = 192

        return size

    def return_each_dataset_size(self):
        carrier = {}
        for key, value in self.masteramparr.items():
            carrier[key] = len(value)
        return carrier

    def test(self):
        self.load_data("BedroomWork")
        self.plot(self.get_square_data_norm(0, 4))
        self.plot(self.get_square_data_norm(0, 3))
        self.plot(self.get_square_data_norm(0, 2))
        self.plot(self.get_square_data_norm(0, 1))
        self.plot(self.get_square_data_norm(0, 0))


k = DataParser()
k.load_data("BedroomWork")

k.plot(k.get_square_data_norm(0, 4))
k.plot(k.get_square_data_norm(0, 3))
k.plot(k.get_square_data_norm(0, 2))
k.plot(k.get_square_data_norm(0, 1))
k.plot(k.get_square_data_norm(0, 0))



