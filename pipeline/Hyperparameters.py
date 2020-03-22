class Hyperparameters:
    VALIDATION_NUMBER = 5
    TEST_NUMBER = 20 #these are all PER SCENARIO

    LEARNING_RATE = 0.001

    MODE_OF_LEARNING = "third" #choices: first, second, third, all, raw
    EPOCH_SIZE = 25
    SIZE = 20
    START = 50
    HOLD_PROB = 0.8
    L2WEIGHT = 0.01

    third_start = 134
    third_end = 191

    second_start = 66
    second_end = 123

    first_start1 = 6
    first_end1 = 32
    first_size1 = first_end1 - first_start1
    first_start2 = 33
    first_end2 = 59
    first_size2 = first_end2 - first_start2

    firstSize = 52
    secondsize =  57
    thirdsize = 57

    allsize = 166
    rawsize = 192

    sizedict = {"first" : 52, "second": 57, "third": 57, "all": 166, "raw": 192}
    #data_to_include = ["../datasets_bigbedroom"]
    data_to_include = ["../datasets", "../datasets_bigbedroom", "../datasets_downstairs"]
