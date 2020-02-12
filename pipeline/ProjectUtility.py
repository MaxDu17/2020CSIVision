import numpy as np
from pipeline.CSIFrame import CSIFrame
import csv
class Utility:

    def csv_file_to_pickle(self, amplitude, phase): #this takes csvs and turns them to pickle file
        k = open(amplitude, "r")
        w = open(phase, "r")

        amp_ = list(csv.reader(k))
        phase_ = list(csv.reader(w))
        if(len(amp_) != len(phase_)): #must be same length to work
            print(len(amp_))
            print(len(phase_))
            raise Exception("These two files are not the same length!")



test = Utility()

test.csv_file_to_pickle("../datasets/TwoPersonAmbient_amplitude.csv", "../datasets/TwoPersonAmbient_phase.csv")

