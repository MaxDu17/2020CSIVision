
import serial
import pickle
import csv
import pyaudio
import time
from datacollector.ParserTool import Parser
from serial.tools import list_ports
import numpy as np
tool = Parser(False, "")

try:
    ser = serial.Serial(port='COM3', baudrate=115200)
except:
    print("sorry, this port is busy or not correct. double check programs!")
    ports = list(list_ports.comports())
    print("here are the available ports: " + str([k.device for k in ports]))
    quit()


#k = open("test.txt", "w")
framearr = []
count = 0
amp_ = []
phase_ = []
idle_counter = 0

MASTERNAME = "TwoPersonAmbient"
try:
    k = open(MASTERNAME + "_amplitude.csv", "r")
    w = open(MASTERNAME + "_phase.csv", "r")

    amp_ = list(csv.reader(k))
    print("I recovered " + str(len(amp_)) + " from amplitude")
    phase_ = list(csv.reader(w))
    print("I recovered " + str(len(phase_)) + " from phase")


    picklefile = open(MASTERNAME + ".pkl", "rb")
    frames = pickle.load(picklefile)
    print("I recovered " + str(len(frames)) + " frames from the pickle")
    picklefile.close()

    pickle.dump(frames, open(MASTERNAME + ".pkl", "wb"))

except:
    print("No existing file to recover. Press enter to continue.")
    input()

k = open(MASTERNAME + "_amplitude.csv", "w")
w = open(MASTERNAME + "_phase.csv", "w")

amp = csv.writer(k, lineterminator = "\n")
phase = csv.writer(w, lineterminator = "\n")

print("Recovering Past Data Session")
amp.writerows(amp_)
phase.writerows(phase_) #recovers past files

while count < 1000:
    print(count)
    s = str(ser.readline())
    frame = tool.extract_and_store(s)
    idle_counter += 1

    print(s)
    if(idle_counter > 8):
        ser.close()
        ser.open()
        print("******************I just reset the serial connection*******************")


    if(frame):
        print("\t Frame Recorded")
        amp.writerow(frame.amplitude)
        phase.writerow(frame.phase)

        if(count % 50 == 0):
            k.flush()
            w.flush()
            print("\t Flush Successful")

        #print("New Frame! This was from:{}, \n\tAmplitude:{} \n\tPhase:{}".format(frame.MAC, frame.amplitude, frame.phase))
        #print("\n")
        count +=1
        idle_counter = 0 #this will reset the serial reset counter
        framearr.append(frame) #this will be dumped into a pickle

pickle.dump(framearr, open(MASTERNAME + ".pkl", "wb"))