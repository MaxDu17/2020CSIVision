import numpy as np
from os import listdir
from pipeline.ProjectUtility import Utility

class LiveParser(Utility):

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

    def normalize_single_column(self, data):
        min_ = min(data)
        max_ = max(data)
        for j in range(len(data)):
            data[j] = (data[j] - min_) / (max_ - min_)

        return data

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
            carrier = frame[66:123]
            newData.append(carrier)

        return newData

    def third_chunk(self, data):
        newData = list()
        for frame in data:
            carrier = frame[134:191]
            newData.append(carrier)

        return newData

    def normalize(self, data): #only run this after you have removed chunky bits
        for i in range(len(data)):
            min_ = min(data[i])
            max_ = max(data[i])
            for j in range(len(data[i])):
                data[i][j] = (data[i][j]- min_)/(max_-min_)
        return data


    def process_from_raw(self, data, chunkIdentifier):
        chunkNum = self.return_chunkIdent_name(chunkIdentifier)
        assert len(data) == self.return_size_name(chunkIdentifier), "problem with array generation"
        carrier = data
        if chunkNum == 1:
            return self.normalize(self.first_chunk(carrier))
        elif chunkNum == 2:
            return self.normalize(self.second_chunk(carrier))
        elif chunkNum == 3:
            return self.normalize(self.third_chunk(carrier))
        elif chunkNum == 4:
            return self.normalize(self.remove_gaps(carrier))
        elif chunkNum == 0:
            return self.normalize(carrier)
        else:
            raise Exception("Invalid chunkIdentifier")

    def process_from_raw_arbi(self, data, start): #makes the largest square matrix possible
        carrier = list()
        for value in data:
            carrier.append(value[start: start + len(data)])
        return carrier



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

    def result_interpret(self, oneHot):
        dictWhole = {0: "Ambient", 1: "Fall", 2: "Sleep", 3: "Walk", 4: "Work" }
        return dictWhole[np.argmax(oneHot)]

