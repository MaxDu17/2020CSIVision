import numpy as np
from pipeline.CSIFrame import CSIFrame
import csv
import pickle
from matplotlib import pyplot as plt
class Utility:

    def csv_file_to_pickle(self, amplitude, phase, picklename): #this takes csvs and turns them to pickle file
        k = open(amplitude, "r")
        w = open(phase, "r")

        amp_ = self.cast_to_float(list(csv.reader(k)))
        phase_ = self.cast_to_float(list(csv.reader(w)))
        if(len(amp_) != len(phase_)): #must be same length to work
            print(len(amp_))
            print(len(phase_))
            raise Exception("These two files are not the same length!")

        carrier = list()

        for i in range(len(amp_)):
            singleobject = CSIFrame(MAC = "bogus", amplitude = amp_[i], phase = phase_[i])
            carrier.append(singleobject)

        pickle.dump(carrier, open(picklename, "wb"))


    def load_from_pickle(self, picklename):
        frames = pickle.load(open(picklename, "rb"))
        return frames


    def csv_file_to_CSIObject(self, amplitude, phase):
        k = open(amplitude, "r")
        w = open(phase, "r")

        amp_ = self.cast_to_float(list(csv.reader(k)))
        phase_ = self.cast_to_float(list(csv.reader(w)))
        if (len(amp_) != len(phase_)):  # must be same length to work
            print(len(amp_))
            print(len(phase_))
            raise Exception("These two files are not the same length!")

        carrier = list()

        for i in range(len(amp_)):
            singleobject = CSIFrame(MAC="bogus", amplitude=amp_[i], phase=phase_[i])
            carrier.append(singleobject)

        return carrier

    def csv_file_to_amp(self, amplitude):
        k = open(amplitude, "r")
        carrier = list(csv.reader(k))
        amp_ = self.cast_to_float(carrier)
        return amp_


    def csv_file_to_phase(self, phase):
        k = open(phase, "r")
        carrier = list(csv.reader(k))
        print(carrier)
        phase_ = self.cast_to_float(carrier)
        return phase_



    def cast_to_float(self, array):
        for i in range(len(array)):
            for j in range(len(array[0])):
                array[i][j] = float(array[i][j])
        return array

    def frame_normalize_minmax(self, array):
        for i in range(len(array)):
            min_ = min(array[i])
            max_ = max(array[i])
            for j in range(len(array[i])):
                array[i][j] = (array[i][j] - min_)/ (max_ - min_)
        return array


    def plot(self, array): #plots csi frame collections
        if max(array) > 1:
            array = self.frame_normalize_minmax(array)
        matrix = np.transpose(np.asarray(array))
        self.display_image(matrix)
        pass

    def display_image(self, matrix):  # this prints out a 3d image
        images_plot = matrix.astype('uint8')
        plt.imshow(images_plot)
        plt.show()



