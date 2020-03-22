import tensorflow as tf
import numpy as np
from pipeline.ProjectUtility import Utility
import pickle
util = Utility()
import csv
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        objects = pickle.load(fo, encoding='bytes')
    assert len(objects) > 0, "there are no weights saved"
    return objects

def accuracy(pred, labels):
    assert len(pred) == len(labels), "lengths of prediction and labels are not the same"
    counter = 0
    for i in range(len(pred)):
        k = np.argmax(pred[i])
        l = np.argmax(labels[i])
        if k == l:
            counter += 1
    return float(counter) / len(pred)

def record_error( data, labels, pred): #this function will record the right and wrong entries
    assert len(data[0]) == len(pred), "your data and prediction don't match"
    assert len(pred) == len(labels), "your prediction and labels don't match"

    wrong = list()
    right = list()
    for i in range(len(data[0])):
        if np.argmax(pred[i]) != np.argmax(labels[i]):
            wrong.append(data[0][i])
        else:
            right.append(data[0][i])
    return right, wrong

def record_error_with_labels(data, labels, pred):
    assert len(data) == len(pred), "your data and prediction don't match"
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

class Logging(): #this class reduces the number of file directory changes per model
    #this is the full package deal
    '''
    it logs your training loss, validation, accuracy, and l2 regularization (if you need more, there can be more!)
    '''
    def __init__(self, base_dir, printout_step, summary_step, save_step): #slash on the directory please!
        assert base_dir[-1] == "/", "you forgot the slash in your base directory"
        self.base_dir = base_dir
        self.train_logger = csv.writer(open(self.base_dir + "xentropyloss.csv", "w"), lineterminator="\n")
        self.acc_logger = csv.writer(open(self.base_dir + "accuracy.csv", "w"), lineterminator="\n")
        self.l2_logger = csv.writer(open(self.base_dir + "l2.csv", "w"), lineterminator="\n")
        self.valid_logger = csv.writer(open(self.base_dir + "valid.csv", "w"), lineterminator="\n")
        self.printout_step = printout_step
        self.summary_step = summary_step
        self.save_step = save_step

    def log_train(self, step, prediction, label, loss, l2loss, big_list): #run me under "with summary_writer as default()
        self.train_logger.writerow([np.asarray(loss)])
        self.acc_logger.writerow([accuracy(prediction, label)])
        self.l2_logger.writerow([np.asarray(l2loss)])

        if step % self.printout_step == 0:
            print("***********************")
            print("Finished epoch", step)
            print("Accuracy: {}".format(accuracy(prediction, label)))
            print("Loss: {}".format(np.asarray(loss)))
            print("***********************")

        if step % self.summary_step == 0:
            tf.summary.scalar(name="Loss", data=loss, step=step)
            tf.summary.scalar(name="Accuracy", data=accuracy(prediction, label), step=step)
            for var in big_list:
                name = str(var.name)
                tf.summary.histogram(name="Variable_" + name, data=var, step=step)
            tf.summary.flush()

        if step % self.save_step == 0:
            print("\n##############SAVING MODE##############\n")
            try:
                os.remove(self.base_dir + "SAVED_WEIGHTS.pkl")
            except:
                print("the saved weights were not removed, because they were not there!")
            dbfile = open(self.base_dir +  "SAVED_WEIGHTS.pkl", "ab")
            pickle.dump(big_list, dbfile)

    @staticmethod
    def make_confusion_matrix(prediction, label):
        assert len(label) == len(prediction), "you seem to have messed up your dimensions!"
        conf = np.zeros(shape=[len(label[0]), len(prediction[0])])
        for i in range(len(prediction)):
            k = np.argmax(prediction[i])
            l = np.argmax(label[i])
            conf[k][l] += 1
        return conf

    @staticmethod
    def test_log(base_dir, prediction, label, confusionmodifier):
        test = open(base_dir+ "confusion" + confusionmodifier + ".csv", "w")
        logger = csv.writer(test, lineterminator="\n")

        conf = Logging.make_confusion_matrix(prediction, label)

        for iterate in conf:
            logger.writerow(iterate)

        test_ = open(base_dir + "results.csv", "w")
        logger_ = csv.writer(test_, lineterminator="\n")
        logger_.writerow([accuracy(prediction, label)])

    def log_valid(self, valid_accuracy, step):
        tf.summary.scalar(name="Validation_accuracy", data=valid_accuracy, step=step)
        self.valid_logger.writerow([valid_accuracy])



class Convolve():
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self, from_file = False, weights = None): #input shape is NOT the parameter you feed into convolve's constructor
        if not(from_file):
            self.w_conv = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev =  0.1),
                                        name = self.name + "_weight", trainable = True, dtype = tf.dtypes.float32)
            self.current_list.append(self.w_conv)

            self.b_conv = tf.Variable(initial_value = tf.zeros(self.shape[3]),
                                        name=self.name + "_bias", trainable = True, dtype = tf.dtypes.float32)
            self.current_list.append(self.b_conv)
        else:
            print("Loading filters from saved weights")
            assert np.shape(weights[0]) == self.shape, "Shape mis-match in class Convolve"

            self.w_conv = tf.Variable(initial_value=weights[0],
                                        name=self.name + "_weight", trainable=True, dtype = tf.dtypes.float32)
            self.current_list.append(self.w_conv)

            self.b_conv = tf.Variable(initial_value=weights[1],
                                        name=self.name + "_bias", trainable=True, dtype = tf.dtypes.float32)
            self.current_list.append(self.b_conv)


    def call(self, input):
        input = tf.cast(input, dtype = np.float32)
        with tf.name_scope("Convolve"):
            conv = tf.nn.relu(tf.nn.conv2d(input, self.w_conv, strides=[1, 1, 1, 1], padding='SAME', name="conv"))
            conv = conv + self.b_conv
            return conv

    def l2loss(self):
        with tf.name_scope("L2Reg"):
            l2 = tf.reduce_sum(tf.abs(self.w_conv))
            return l2

class Pool():
    def call(self, input):
        with tf.name_scope("Pool"):
            pooled = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool")
            return pooled

class Flatten():
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

    def call(self, input):
        with tf.name_scope("Flatten"):
            flattened = tf.reshape(input, self.shape, name=self.name + "_flatten")
            return flattened


class Dropout():
    def __init__(self, name, hold_prob):
        self.name = name
        self.hold_prob = hold_prob

    def call(self, input):
        with tf.name_scope("Dropout"):
            output = tf.nn.dropout(input, rate=1 - self.hold_prob, name = self.name)
            return output

class Softmax():
    def call(self, input):
        with tf.name_scope("Softmax"):
            prediction = tf.nn.softmax(input)
            return prediction


class Combine_add():
    def call(self, input1, input2):
        with tf.name_scope("Combine_and_add"):
            output = input1 + input2
            return output

class FC():
    def __init__(self, current_list, shape, name):
        self.current_list = current_list
        self.shape = shape
        self.name = name

    def build(self, from_file = False, weights = None):
        if not(from_file):
            self.w_fc = tf.Variable(initial_value = tf.random.truncated_normal(self.shape, stddev=0.1),
                                      name=self.name + "_weight", trainable = True)
            self.current_list.append(self.w_fc)
            self.b_fc = tf.Variable(initial_value = tf.zeros(self.shape[1]),
                                      name=self.name + "_bias", trainable = True)
            self.current_list.append(self.b_fc)
        else:
            print("Loading neurons from saved weights")
            assert np.shape(weights[0]) == self.shape, "shape mis-match in FC"
            self.w_fc = tf.Variable(initial_value=weights[0],
                                      name=self.name + "_weight", trainable=True)
            self.current_list.append(self.w_fc)
            self.b_fc = tf.Variable(initial_value=weights[1],
                                      name=self.name + "_bias", trainable=True)
            self.current_list.append(self.b_fc)


    def call(self, input):
        with tf.name_scope("Fully_Connected_Layer"):
            fc = tf.matmul(input, self.w_fc) + self.b_fc
            return fc



class ResNetChunk(): #this is a "super" model class, and it builds a resnet chunk
    def __init__(self, deep, weight_shape, current_list, name):
        assert deep % 2 == 0, "depth must be an even number"
        self.depth = deep
        self.shape = weight_shape
        self.current_list = current_list
        self.name = name
        self.layer_list = list() #this contains propagation list

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == self.depth * 2, "there seems to be a dimension problem with the pickle list"
        for i in range(self.depth): #this is the init pass
            layer_obj = Convolve(self.current_list, shape=self.shape, name = self.name + "_Resnet_layer_" + str(i))
            layer_obj.build(from_file = True, weights = exclusive_list[2*i:2*i+2])
            self.layer_list.append(layer_obj)

    def build(self):
        for i in range(self.depth): #this is the init pass
            layer_obj = Convolve(self.current_list, shape=self.shape, name = self.name + "_Resnet_layer_" + str(i))
            layer_obj.build()
            self.layer_list.append(layer_obj)

    def call(self, input): #disregard l2 norm for now, but add later
        current_input = input
        for i in range(0, self.depth, 2):
            with tf.name_scope(name = "Resnet_layer_group_" + str(i)):
                residual = current_input
                output_1 = self.layer_list[i].call(input)
                output_2 = self.layer_list[i+1].call(output_1)
                current_input = output_2 + residual #this propagates the network
        return current_input

    def l2loss(self): #wrapper function
        l2loss = 0
        with tf.name_scope("ResNetL2Loss"):
            for layer in self.layer_list:
                l2loss += layer.l2loss()

        return l2loss
    
    
class Inceptionv1Chunk_naive(): #for programming simplicity, this does NOT have pooling yet
    def __init__(self, current_list, name, data_depth):
        self.current_list = current_list
        self.name = name
        self.data_depth = data_depth

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == 3*2, "there seems to be a dimension problem with the pickle list"
        self.one_one = Convolve(self.current_list, shape = [1, 1, self.data_depth, self.data_depth], name = self.name +
                           "_Inception_one")
        self.one_one.build(from_file = True, weights = exclusive_list[0:2])

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_three")
        self.three_three.build(from_file=True, weights=exclusive_list[2:4])

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_five")
        self.five_five.build(from_file=True, weights=exclusive_list[4:6])


    def build(self):
        self.one_one = Convolve(self.current_list, shape=[1, 1, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_one")
        self.one_one.build()

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_three")
        self.three_three.build()

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                     "_Inception_five")
        self.five_five.build()


    def call(self, input):
        with tf.name_scope("Inception_net_prop"):
            one_one_out = self.one_one.call(input)

            three_three_out = self.three_three.call(input)

            five_five_out = self.five_five.call(input)

            output = tf.concat(values = [one_one_out, three_three_out, five_five_out], axis = 3)


        return output

    def l2loss(self):
        with tf.name_scope("Inception_L2"):
            l2 = self.one_one.l2loss()
            l2 += self.three_three.l2loss()
            l2 += self.five_five.l2loss()
            return l2

class Inceptionv1Chunk(): #for programming simplicity, this does NOT have pooling yet
    def __init__(self, current_list, name, data_depth):
        self.current_list = current_list
        self.name = name
        self.data_depth = data_depth

    def build_model_from_pickle(self, exclusive_list):
        assert len(exclusive_list) == 3*2, "there seems to be a dimension problem with the pickle list"

        self.one_one = Convolve(self.current_list, shape = [1, 1, self.data_depth, self.data_depth], name = self.name +
                           "_Inception_one")
        self.one_one.build(from_file = True, weights = exclusive_list[0:2])

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_three")
        self.three_three.build(from_file=True, weights=exclusive_list[2:4])

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_five")
        self.five_five.build(from_file=True, weights=exclusive_list[4:6])


    def build(self):
        self.one_one = Convolve(self.current_list, shape=[1, 1, self.data_depth, self.data_depth], name=self.name +
                                                                                                   "_Inception_one")
        self.one_one.build()

        self.three_three = Convolve(self.current_list, shape=[3, 3, self.data_depth, self.data_depth], name=self.name +
                                                                                                       "_Inception_three")
        self.three_three.build()

        self.five_five = Convolve(self.current_list, shape=[5, 5, self.data_depth, self.data_depth], name=self.name +
                                                                                                     "_Inception_five")
        self.five_five.build()


    def call(self, input):
        with tf.name_scope("Inception_net_prop"):
            one_one_out = self.one_one.call(input)

            three_three_out = self.three_three.call(input)

            five_five_out = self.five_five.call(input)

            output = tf.concat(values = [one_one_out, three_three_out, five_five_out], axis = 3)


        return output

    def l2loss(self):
        with tf.name_scope("Inception_L2"):
            l2 = self.one_one.l2loss()
            l2 += self.three_three.l2loss()
            l2 += self.five_five.l2loss()

            return l2




