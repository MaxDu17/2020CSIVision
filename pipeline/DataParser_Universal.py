import numpy as np
from os import listdir
from pipeline.ProjectUtility import Utility

masterdirectory = "../datasets"

#directoryList = ["../datasets", "../datasets_bigbedroom", "../datasets_downstairs"]


class DataParser_Universal(Utility):
    def __init__(self, directoryList):
        self.datasetList = list()
        self.amparr = list()
        self.superList = list()  # this will contain dictionasries for each directory

        for largeDirectory in directoryList:
            masteramparr = {}
            files = sorted(listdir(largeDirectory))

            for file in files:
                if file.find(".") < 0:
                    try:
                        masteramparr[file] = self.getAmpArr(file)
                        print(largeDirectory + "/" + file)
                        self.datasetList.append(file)
                    except:
                        print(file + " was empty. I skipped it")
            self.superList.append(masteramparr)

        self.datasetList = list(dict.fromkeys(self.datasetList))  # removes duplicates

    def get_meta(self):
        return self.datasetList

    def getAmpArr(self, dataset):
        return self.csv_file_to_amp(masterdirectory + "/" + dataset + "/" + dataset + "_amplitude.csv")

    def getMasterAmpArr(self, fileDirectory):
        return self.superList[fileDirectory]

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

    def arbi_chunk(self, data, start, size):
        newData = list()
        for frame in data:
            carrier = frame[start: start + size]
            newData.append(carrier)
        return newData

    def normalize_single_column(self, data):
        min_ = min(data)
        max_ = max(data)
        for j in range(len(data)):
            data[j] = (data[j] - min_) / (max_ - min_)

        return data

    def normalize(self, data):  # only run this after you have removed chunky bits
        newData = np.zeros(shape=[len(data), len(data[0])])
        for i in range(len(data)):
            min_ = min(data[i])
            max_ = max(data[i])
            for j in range(len(data[i])):
                newData[i][j] = (data[i][j] - min_) / (max_ - min_)
        return newData


    def load_data_multiple_file(self, dataName, count):  # must call at first
        self.current_name = dataName
        assert dataName in self.superList[count], "your file does not exist"
        try:
            self.amparr = self.superList[count][dataName]
        except:
            raise Exception("Unable to load the filename specified")

    def get_data_arbi(self, start_time, end, start_vert, size, removegap):
        assert len(self.amparr) > 0, "you did not load any data"

        if start_time > len(self.amparr) or end > len(self.amparr):
            print(str(start_time) + "\t" + str(end))
            print(len(self.amparr))
            raise Exception("You overshot on your array access")

        if(removegap):
            carrier = self.remove_gaps(self.amparr[start_time:end])
        else:
            carrier =self.amparr[start_time:end]
        # carrier = self.amparr[start_time:end]

        return self.arbi_chunk(carrier, start_vert, size)


    def get_square_data_arbi(self, start, size, start_vert, removegaps):
        # print("\t" + str(start) + "------" + str(size))
        return self.get_data_arbi(start, start + size, start_vert, size, removegaps)

    def get_square_data_arbi_norm(self, start, size, start_vert, removegaps):
        return self.normalize(self.get_square_data_arbi(start, size, start_vert, removegaps))




