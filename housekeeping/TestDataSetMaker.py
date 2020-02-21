from pipeline.DatasetMaker import DatasetMaker
from pipeline.ProjectUtility import Utility
from pipeline.DataPipeline import DataPipeline
import numpy as np

DM = DatasetMaker()
tool = Utility()

big = DM.next_epoch_batch()
print("\n\n")
print("The validation set start at:" + str(DM.validation_start) + "and constains " +
      str(len(DM.valid_set)) + " elements")
print("The test set starts at: " + str(DM.test_start) + "and contains " +
      str(len(DM.test_set)) + " elements")
print("The training set starts at (universal): " + str(DM.train_start))
print("This epoch's has " + str(len(DM.train_matrix)) + " elements")
print("This is the number of frames for training: " + str(len(DM.train_set)))
print(np.shape(big))
print(len(big[0]))

'''
trainObject = DM.next_train()
print(trainObject.label)
print(trainObject.oneHot)
print(trainObject.data)
print(trainObject.startIndex)
tool.plot(trainObject.data)
'''
