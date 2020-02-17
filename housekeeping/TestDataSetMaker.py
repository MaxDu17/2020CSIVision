from pipeline.DatasetMaker import DatasetMaker
from pipeline.ProjectUtility import Utility
from pipeline.DataPipeline import DataPipeline

DM = DatasetMaker()
tool = Utility()

DM.next_epoch()
print("\n\n")
print("The validation set start at:" + str(DM.validation_start) + "and constains " +
      str(len(DM.valid_set)) + " elements")
print("The test set starts at: " + str(DM.test_start) + "and contains " +
      str(len(DM.test_set)) + " elements")
print("The training set starts at (universal): " + str(DM.train_start))
print("This epoch's has " + str(len(DM.train_matrix)) + " elements")
print("This is the number of frames for training: " + str(len(DM.train_set)))
trainObject = DM.next_train()
print(trainObject.label)
print(trainObject.oneHot)
print(trainObject.data)
print(trainObject.startIndex)
tool.plot(trainObject.data)

