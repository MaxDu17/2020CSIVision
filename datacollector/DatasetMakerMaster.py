
import serial
import pickle
import csv
import pyaudio
import time
from datacollector.ParserTool import Parser
from serial.tools import list_ports
import numpy as np
p = pyaudio.PyAudio()
tool = Parser(False, "")


try:
    ser = serial.Serial(port='COM3', baudrate=115200)
except:
    print("sorry, this port is busy or not correct. double check programs!")
    ports = list(list_ports.comports())
    print("here are the available ports: " + str([k.device for k in ports]))
    quit()


MASTERNAME = "../datasets_downstairs/BedroomFall/BedroomFall"
#available
#BedroomAmbient
#BedroomFall
#BedroomSleep
#BedroomWalk
#BedroomWork

framearr = []
amp_ = []
phase_ = []
fs = 44100  # sampling rate, Hz, must be integer
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

DATALENGTH = 1000  # roughly one hour


def playSound():
    volume = 1  # range [0.0, 1.0]

    duration = 0.5  # in seconds, may be float
    f = 880.0  # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    # for paFloat32 sample values must be in range [-1.0, 1.0]


    stream.write(volume * samples)
    #stream.stop_stream()
    #stream.close()
    #p.terminate()

def recoverOperations():
    try:
        k = open(MASTERNAME + "_amplitude.csv", "r")
        w = open(MASTERNAME + "_phase.csv", "r")
        semantic = input("do you want to recover files? (y,n)")

        if semantic == "y":
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
        input("No existing file to recover. Press enter to continue.")


def main():
    recoverOperations()

    k = open(MASTERNAME + "_amplitude.csv", "w")
    w = open(MASTERNAME + "_phase.csv", "w")

    amp = csv.writer(k, lineterminator = "\n")
    phase = csv.writer(w, lineterminator = "\n")

    print("Recovering Past Data Session")
    amp.writerows(amp_)
    phase.writerows(phase_) #recovers past files

    semantic = int(input("How many seconds do you want to wait for?"))
    time.sleep(semantic)

    idle_counter = 0
    count = 0

    ser.reset_input_buffer()

    while count < DATALENGTH:
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
            playSound()
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

if __name__ == "__main__":
    main()
