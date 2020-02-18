from pipeline.DataParser import DataParser
import tensorflow as tf
import numpy as np
import csv
import os
from pipeline.ProjectUtility import Utility
import shutil
import pickle
util = Utility()
from pipeline.DataPipeline import DataPipeline
from pipeline.MyCNNLibrary import * #this is my own "keras" extension onto tensorflow
from pipeline.Hyperparameters import Hyperparameters


big_list = list() #this is the weights and biases matrix


class Model():
    def __init__(self):
        self.cnn_1 = Convolve(big_list, [3, 3, 1, 4], "Layer_1_CNN") #3, 3, 1 in, 4 out
        self.cnn_2 = Convolve(big_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.pool_1 = Pool()

        self.cnn_3 = Convolve(big_list, [3, 3, 4, 8], "Layer_2_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, 25*25*8], "Fully_Connected")
        self.fc_1 = FC(big_list, [25*25*8, output_size], "Layer_1_FC")
        self.softmax = Softmax()

    #at this point, the objects are made, but nothing is done yet
    def build_model_from_pickle(self, file_dir):
        big_list = unpickle(file_dir)
        #weights and biases are arranged alternating and in order of build
        self.cnn_1.build(from_file = True, weights = big_list[0:2])
        self.cnn_2.build(from_file = True, weights = big_list[2:4])
        self.cnn_3.build(from_file=True, weights=big_list[4:6])
        self.fc_1.build(from_file = True, weights = big_list[6:8])

    def build_model(self):
        self.cnn_1.build()
        self.cnn_2.build()
        self.cnn_3.build()
        self.fc_1.build()

    @tf.function
    def call(self, input):
        print("I am in calling {}".format(np.shape(input)))
        x= self.cnn_1.call(input)
        l2 = self.cnn_1.l2loss()
        x = self.cnn_2.call(x)
        l2 += self.cnn_2.l2loss()
        x = self.pool_1.call(x)

        x = self.cnn_3.call(x)
        l2 += self.cnn_3.l2loss()
        x = self.pool_2.call(x)

        x = self.flat.call(x)
        x = self.fc_1.call(x)
        output = self.softmax.call(x)
        return output, l2

def accuracy(pred, labels):
    assert len(pred) == len(labels), "lengths of prediction and labels are not the same"
    counter = 0
    for i in range(len(pred)):
        k = np.argmax(pred[i])
        l = np.argmax(labels[i])
        if k == l:
            counter += 1
    return float(counter)/len(pred)

def record_error(data, labels, pred):
    assert len(data[0]) == len(pred), "your data and prediction don't match"
    assert len(pred) == len(labels), "your prediction and labels don't match"

    wrong = list()
    right = list()
    wrong_index = list()
    for i in range(len(data)):
        if np.argmax(pred[i]) != np.argmax(labels[i]):
            wrong.append(data[i])
            wrong_index.append(np.argmax(labels[i]))
        else:
            right.append(data[i])
    return right, wrong, wrong_index

def Big_Train():
    try:
        os.mkdir("Graphs_and_Results/basic/" + version + "/")
    except:
        pass

    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    print("*****************Training*****************")

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, [version])

    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_INIT)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    print("loading dataset")
    datafeeder.load_train_to_RAM()  # loads the training data to RAM
    summary_writer = tf.summary.create_file_writer(logdir="Graphs_and_Results/basic/" + version + "/")
    print("starting training")

    print("Making model")
    model = Model()
    model.build_model()
    tf.summary.trace_on(graph=True, profiler=True)


    train_logger = csv.writer(open("Graphs_and_Results/basic/" + version + "/xentropyloss.csv", "w"),
                              lineterminator="\n")
    acc_logger = csv.writer(open("Graphs_and_Results/basic/" + version + "/accuracy.csv", "w"),
                              lineterminator="\n")
    l2_logger = csv.writer(open("Graphs_and_Results/basic/" + version + "/l2.csv", "w"),
                              lineterminator="\n")
    valid_logger = csv.writer(open("Graphs_and_Results/basic/" + version + "/valid.csv", "w"),
                              lineterminator="\n")

    for epoch in range(1001):
        data, label = datafeeder.nextBatchTrain_dom(150)
        data = data[0]
        with tf.GradientTape() as tape:
            predictions, l2_loss = model.call(data) #this is the big call

            pred_loss = loss_function(label, predictions) #this is the loss function
            pred_loss = pred_loss + L2WEIGHT * l2_loss #this implements lasso regularization

            train_logger.writerow([np.asarray(pred_loss)])
            acc_logger.writerow([accuracy(predictions, label)])
            l2_logger.writerow([np.asarray(l2_loss)])

            if epoch == 0: #creates graph
                with summary_writer.as_default():
                    tf.summary.trace_export(name="Graph", step=0, profiler_outdir="Graphs_and_Results/basic/" + version + "/")

            if epoch % 20 == 0 and epoch > 1:
                print("***********************")
                print("Finished epoch", epoch)
                print("Accuracy: {}".format(accuracy(predictions, label)))
                print("Loss: {}".format(np.asarray(pred_loss)))
                print("***********************")
                with summary_writer.as_default():
                    tf.summary.scalar(name = "Loss", data = pred_loss, step = epoch)
                    tf.summary.scalar(name = "Accuracy", data = accuracy(predictions, label), step = epoch)
                    for var in big_list:
                        name = str(var.name)
                        tf.summary.histogram(name = "Variable_" + name, data = var, step = epoch)
                    tf.summary.flush()

            if epoch % 50 == 0:
                valid_accuracy = Validation(model, datafeeder)
                with summary_writer.as_default():
                    tf.summary.scalar(name = "Validation_accuracy", data = valid_accuracy, step = epoch)
                valid_logger.writerow([valid_accuracy])

            if epoch % 100 == 0 and epoch > 1:
                print("\n##############SAVING MODE##############\n")
                try:
                    os.remove("Graphs_and_Results/basic/" + version + "/" + "SAVED_WEIGHTS.pkl")
                except:
                    print("the saved weights were not removed, because they were not there!")
                dbfile = open("Graphs_and_Results/basic/" + version + "/" + "SAVED_WEIGHTS.pkl", "ab")
                pickle.dump(big_list, dbfile)

        gradients = tape.gradient(pred_loss, big_list)

        optimizer.apply_gradients(zip(gradients, big_list))
    Test_live(model, datafeeder)

def Validation(model, datafeeder):
    print("\n##############VALIDATION##############\n")

    data, label = datafeeder.GetValid_dom()
    data = data[0]
    predictions, l2loss = model.call(data)
    assert len(label) == len(predictions)
    valid_accuracy = accuracy(predictions, label)
    print("This is the validation set accuracy: {}".format(valid_accuracy))
    return valid_accuracy


def Test_live(model, datafeeder):
    print("\n##############TESTING##############\n")

    data, label = datafeeder.GetTest_dom()
    predictions, l2loss = model.call(data)
    data = data[0]
    assert len(label) == len(predictions)
    conf = np.zeros(shape=[len(label[0]), len(predictions[0])])
    for i in range(len(predictions)):
        k = np.argmax(predictions[i])
        l = np.argmax(label[i])
        conf[k][l] += 1
    test = open("Graphs_and_Results/basic/" + version + "/confusion.csv", "w")
    logger = csv.writer(test, lineterminator="\n")

    test_ = open("Graphs_and_Results/basic/" + version + "/results.csv", "w")
    logger_ = csv.writer(test_, lineterminator="\n")
    logger_.writerow([accuracy(predictions, label)])

    for iterate in conf:
        logger.writerow(iterate)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    right, wrong, wrong_index = record_error(data, label, predictions)
    return right, wrong, wrong_index

def Test():
    print("Making model")
    model = Model()
    model.build_model_from_pickle("Graphs_and_Results/basic/" + version + "/" + "SAVED_WEIGHTS.pkl")

    datafeeder = Prep(TEST_AMOUNT, VALID_AMOUNT, [version])
    datafeeder.load_train_to_RAM()
    data, label = datafeeder.GetTest_dom()
    data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)

    assert len(label) == len(predictions), "something is wrong with the loaded model or labels"
    right, wrong, wrong_list = record_error(data, label, predictions)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    try:
        os.mkdir("Graphs_and_Results/basic/" + version + "/wrong/")
        os.mkdir("Graphs_and_Results/basic/" + version + "/right/")
        os.mkdir("Graphs_and_Results/wrongs/")
    except:
        shutil.rmtree("Graphs_and_Results/basic/" + version + "/wrong/")
        shutil.rmtree("Graphs_and_Results/basic/" + version + "/right/")
        os.mkdir("Graphs_and_Results/basic/" + version + "/wrong/")
        os.mkdir("Graphs_and_Results/basic/" + version + "/right/")

    wrong_logger = csv.writer(open("Graphs_and_Results/wrongs/" + version + "SimpleCNN.csv", "w"),
                              lineterminator="\n")
    for element in wrong_list:
        wrong_logger.writerow([element])
    for i in range(len(wrong)):
        print("Saving wrong image {}".format(i))
        carrier = np.reshape(wrong[i], [100, 100])
        util.save_image(255 * carrier, "Graphs_and_Results/basic/" + version + "/wrong/" + str(i) + ".jpg", "L")

    for i in range(len(right)):
        print("Saving right image {}".format(i))
        carrier = np.reshape(right[i], [100, 100])
        util.save_image(255 * carrier, "Graphs_and_Results/basic/" + version + "/right/" + str(i) + ".jpg", "L")


def main():
    print("Starting the program!")
    query = input("What mode do you want? Train (t) or Test from model (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Test()


if __name__ == '__main__':
    main()



