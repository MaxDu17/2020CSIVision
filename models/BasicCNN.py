from pipeline.MyCNNLibrary import * #this is my own "keras" extension onto tensorflow
from pipeline.Hyperparameters import Hyperparameters
from pipeline.DatasetMaker_Universal import DatasetMaker_Universal
from pipeline.DataParser_Universal import DataParser_Universal
from housekeeping.csv_to_mat import ConfusionMatrixVisualizer

DP = DataParser_Universal()

name = "Vanilla"

version = "AllDataCNN" + Hyperparameters.MODE_OF_LEARNING

weight_bias_list = list() #this is the weights and biases matrix

base_directory = "../Graphs_and_Results/" + name + "/" + version + "/"
try:
    os.mkdir(base_directory)
    print("made directory {}".format(base_directory)) #this can only go one layer deep
except:
    print("directory exists!")
    pass


pool_size = (int(Hyperparameters.sizedict[Hyperparameters.MODE_OF_LEARNING]/4.0 + 0.99))**2 * 8
class Model():
    def __init__(self, DM):
        self.cnn_1 = Convolve(weight_bias_list, [3, 3, 1, 4], "Layer_1_CNN")
        self.cnn_2 = Convolve(weight_bias_list, [3, 3, 4, 4], "Layer_2_CNN")
        self.pool_1 = Pool()

        self.cnn_3 = Convolve(weight_bias_list, [3, 3, 4, 8], "Layer_2_CNN")
        self.pool_2 = Pool()

        self.flat = Flatten([-1, pool_size], "Fully_Connected")
        self.fc_1 = FC(weight_bias_list, [pool_size, DM.num_labels()], "Layer_1_FC")
        self.softmax = Softmax()

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

def Big_Train():
    logger = Logging(base_directory, 10, 20, 100)  # makes logging object
    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    print("*****************Training*****************")

    print("loading dataset")
    DM = DatasetMaker_Universal(DP, Hyperparameters.MODE_OF_LEARNING)

    optimizer = tf.keras.optimizers.Adam(learning_rate = Hyperparameters.LEARNING_RATE) #can use a changing learning rate
    loss_function = tf.keras.losses.CategoricalCrossentropy()


    summary_writer = tf.summary.create_file_writer(logdir=base_directory)
    print("starting training")

    print("Making model")
    model = Model(DM)
    try:
        semantic = input("restore model? (y,n)")
        if semantic == "y":
            model.build_model_from_pickle(base_directory + "SAVED_WEIGHTS.pkl")
        else:
            model.build_model()
    except:
        model.build_model()


    tf.summary.trace_on(graph=True, profiler=False)


    for epoch in range(1001):
        data, label = DM.next_epoch_batch()


        with tf.GradientTape() as tape:
            predictions, l2_loss = model.call(data) #this is the big call

            pred_loss = loss_function(label, predictions) #this is the loss function
            pred_loss = pred_loss + Hyperparameters.L2WEIGHT * l2_loss #this implements lasso regularization

            if epoch == 0: #creates graph
                with summary_writer.as_default():
                    tf.summary.trace_export(name="Graph", step=0, profiler_outdir=base_directory)

            if epoch % 50 == 0: #takes care of validation accuracy
                valid_accuracy = Validation(model, DM)
                with summary_writer.as_default():
                    logger.log_valid(valid_accuracy, epoch)

            with summary_writer.as_default(): #this is the big player logger and printout
                logger.log_train(epoch, predictions, label, pred_loss, l2_loss, weight_bias_list)

        gradients = tape.gradient(pred_loss, weight_bias_list)
        optimizer.apply_gradients(zip(gradients, weight_bias_list))

    Test_live(model, DM)

def Validation(model, datafeeder):
    print("\n##############VALIDATION##############\n")

    data, label = datafeeder.valid_batch()

    predictions, l2loss = model.call(data)
    assert len(label) == len(predictions)
    valid_accuracy = accuracy(predictions, label)
    print("This is the validation set accuracy: {}".format(valid_accuracy))
    return valid_accuracy


def Test_live(model, datafeeder):
    print("\n##############TESTING##############\n")

    data, label = datafeeder.test_batch()

    predictions, l2loss = model.call(data)
    Logging.test_log(base_directory, predictions, label, "")

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    right, wrong, wrong_index = record_error_with_labels(data, label, predictions)
    ConfusionMatrixVisualizer(name=name, version=version, testTag = "")
    return right, wrong, wrong_index

def Test():
    print("Making model")
    testTag = "_bigbedroom_only"
    DM = DatasetMaker_Universal(DP, Hyperparameters.MODE_OF_LEARNING)
    model = Model(DM)
    model.build_model_from_pickle(base_directory + "SAVED_WEIGHTS.pkl")

    data, label = DM.test_batch()

    #data = data[0]  # this is because we now have multiple images in the pickle
    predictions, l2loss = model.call(data)
    Logging.test_log(base_directory, predictions, label, testTag)

    print("This is the test set accuracy: {}".format(accuracy(predictions, label)))
    ConfusionMatrixVisualizer(name = name, version = version, testTag = testTag)


def main():
    print("Starting the program!")
    query = input("What mode do you want? Train (t) or Test from model (m)?\n")
    if query == "t":
        Big_Train()
    if query == "m":
        Test()


if __name__ == '__main__':
    main()
