import numpy as np
from pipeline.CSIFrame import CSIFrame
import csv
import pickle
from matplotlib import pyplot as plt
from collections import Counter
from PIL import Image

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

    def cast_csv_to_float(self, file_object): #this takes a file object csv and returns a matrix
        logger = csv.reader(file_object)
        matrix = list(logger)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = float(matrix[i][j])
        return matrix

    def cast_csv_to_int(self, file_object):
        logger = csv.reader(file_object)
        matrix = list(logger)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = int(matrix[i][j])
        return matrix


    def frame_normalize_minmax(self, array):
        for i in range(len(array)):
            min_ = min(array[i])
            max_ = max(array[i])
            for j in range(len(array[i])):
                array[i][j] = (array[i][j] - min_)/ (max_ - min_)
        return array

    def frame_normalize_minmax_image(self, array):
        for i in range(len(array)):
            min_ = min(array[i])
            max_ = max(array[i])
            for j in range(len(array[i])):
                array[i][j] = 255*(array[i][j] - min_)/ (max_ - min_)
        return array


    def plot(self, array): #plots csi frame collections
        array = self.frame_normalize_minmax_image(array)
        matrix = np.transpose(np.asarray(array))
        self.display_image(matrix)
        pass

    def display_image(self, matrix):  # this prints out a 3d image
        images_plot = matrix.astype('uint8')
        plt.imshow(images_plot)
        plt.show()

    def flip_lr(self, matrix):
         return np.fliplr(matrix)

    def flip_ud(self, matrix):
         return np.flipud(matrix)

    def rot_ck(self, matrix):
        return np.rot90(matrix, k = 3)

    def rot_cck(self, matrix):
        return np.rot90(matrix, k = 1)

    def add_noise_RGB(self, matrix):
        shape = np.shape(matrix)
        noise = np.random.randint(10, size=shape, dtype='uint8')
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                for k in range(len(matrix[0][0])):
                    if (matrix[i][j][k] != 245):
                        matrix[i][j][k] += noise[i][j][k]

        return matrix

    def add_noise_L(self, matrix): #this is for greyscale
        shape = np.shape(matrix)
        carrier = matrix.copy()
        noise = np.random.randint(10, size=shape, dtype='uint8')
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if (matrix[i][j] < 245):
                    carrier[i][j] += noise[i][j]
        return matrix


    def trans_vert(self, matrix, amount, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        a = 1
        b = 0
        c = 0  # left/right (i.e. 5/-5)
        d = 0
        e = 1
        f = amount  # up/down (i.e. 5/-5)
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        return np.asarray(img)

    def trans_hor(self, matrix, amount, type):
        img = Image.fromarray(matrix.astype(np.uint8), type)
        a = 1
        b = 0
        c = amount  # left/right (i.e. 5/-5)
        d = 0
        e = 1
        f = 0  # up/down (i.e. 5/-5)
        img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
        return np.asarray(img)

    def save_image(self, matrix, path, type):
        matrix = np.asarray(matrix)
        img = Image.fromarray(matrix.astype(np.uint8), type)
        img.save(path)

