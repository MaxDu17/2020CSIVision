
import serial
import pickle

import time
from datacollector.ParserTool import Parser
from serial.tools import list_ports
from pipeline.Hyperparameters import Hyperparameters
from pipeline.LiveTestParser import LiveParser
import numpy as np
from pipeline.MyCNNLibrary import * #this is my own "keras" extension onto tensorflow

HYP = Hyperparameters()
LiveHelp = LiveParser()


version = "BasicCNN_" + HYP.MODE_OF_LEARNING

weight_bias_list = list() #this is the weights and biases matrix

base_directory = "../Graphs_and_Results/Vanilla" + "/" + version + "/"

tool = Parser(False, "")


try:
    ser = serial.Serial(port='COM3', baudrate=115200)
except:
    print("sorry, this port is busy or not correct. double check programs!")
    ports = list(list_ports.comports())
    print("here are the available ports: " + str([k.device for k in ports]))
    quit()


def main(model):
    holdinglist = list()
    semantic = int(input("How many seconds do you want to wait for?"))
    time.sleep(semantic)

    idle_counter = 0

    ser.reset_input_buffer()

    buffer_size = LiveHelp.return_size_name(HYP.MODE_OF_LEARNING)


    while True:
        #print(count)
        s = str(ser.readline())
        frame = tool.extract_and_store(s)
        idle_counter += 1

        #print(s)
        if(idle_counter > 8):
            ser.close()
            ser.open()
            print("******************I just reset the serial connection*******************")


        if(frame):
            idle_counter = 0
            print("\t Frame Recorded" + str(len(holdinglist)))
            holdinglist.append(frame.amplitude)

            if len(holdinglist) == buffer_size:
                print("This is happening")
                feed_in = LiveHelp.process_from_raw(holdinglist, HYP.MODE_OF_LEARNING)
                feed_in = np.reshape(feed_in, [LiveHelp.return_size_name(HYP.MODE_OF_LEARNING),
                                      LiveHelp.return_size_name(HYP.MODE_OF_LEARNING),1,1])
                runNet(feed_in, model)
                holdinglist = list()



class Model():
    def __init__(self):
        pool_size = (int(LiveHelp.return_size_name(HYP.MODE_OF_LEARNING) / 4) + 1) ** 2 * 8
        self.cnn_1 = Convolve(weight_bias_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.cnn_2 = Convolve(weight_bias_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.pool_1 = Pool()

        self.cnn_3 = Convolve(weight_bias_list, [3, 3, 4, 8], "Layer_2_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, pool_size], "Fully_Connected")
        self.fc_1 = FC(weight_bias_list, [pool_size, 5], "Layer_1_FC") #5 categories; change if necessary
        self.softmax = Softmax()

    def build_model_from_pickle(self, file_dir):
        big_list = unpickle(file_dir)
        # weights and biases are arranged alternating and in order of build
        self.cnn_1.build(from_file=True, weights=big_list[0:2])
        self.cnn_2.build(from_file=True, weights=big_list[2:4])
        self.cnn_3.build(from_file=True, weights=big_list[4:6])
        self.fc_1.build(from_file=True, weights=big_list[6:8])

    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.cnn_3.build()
        self.fc_1.build()

    @tf.function
    def call(self, input):
        print("I am in calling {}".format(np.shape(input)))
        x = self.cnn_1.call(input)
        x = self.cnn_2.call(x)
        x = self.pool_1.call(x)
        x = self.cnn_3.call(x)
        x = self.pool_2.call(x)

        x = self.flat.call(x)
        x = self.fc_1.call(x)
        output = self.softmax.call(x)
        return output

def makeNet():
    print("Making model")
    model = Model()
    model.build_model_from_pickle(base_directory + "SAVED_WEIGHTS.pkl")
    return model

def runNet(inVal, model):
    prediction = model.call(inVal)
    print("This is what the predicted value is: " + LiveHelp.result_interpret(prediction))





if __name__ == "__main__":
    model = makeNet()
    main(model)
