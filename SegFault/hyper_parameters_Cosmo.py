Input = {
        "BATCH_SIZE" : 4,              #mini-batch size for training and validation
        "NUM_THREADS" : 64,              #number of threads to read data
        "CAPACITY" : 0,
        "MIN_AFTER_DEQUEUE" : 200       #the minimum number in the queue after dequeue (Min_after_dequeue and capacity together determines the shuffling of input data)
        }

Input["CAPACITY"] = Input["BATCH_SIZE"]*4 + Input["MIN_AFTER_DEQUEUE"]

Input_Test = {
	"BATCH_SIZE" : 4,              #mini-batch size for test data
	"NUM_THREADS" : 64,              #number of threads to read data
	"CAPACITY" : 0,
	"MIN_AFTER_DEQUEUE" : 32
	}

Input_Test["CAPACITY"] = Input_Test["BATCH_SIZE"]*4 + Input_Test["MIN_AFTER_DEQUEUE"]

Model = {
        "REG_RATE": 0.,                 #regularization of weights: currently set to 0 since batch_normalization has the same effect of regularization
        "LEAK_PARAMETER": 0.01,         #leaky parameter for leaky relu
        "LEARNING_RATE" : 0.0001,       #adam_optimizer to do the update. 
        "DROP_OUT": 0.5                 #apply drop out in fully connected layer. this value gives the probabilty of keep the node. 
}

RUNPARAM={
	"num_epoch": 1,              #each epoch means a fully pass over the data. The program might stop before running num_epoch (see next line).        
        "require_improvement": 50,      #if with require_improvement, there is no improvement in validation error, then stop running. 
	"num_train":4,                #total number of simulations for training
	"num_val":1,                   #total number of simulations for validation
        "num_test":1,                  #total number of simulations for testing
	"batch_per_epoch":0,             
	"batch_per_epoch_val":0,
        "iter_test":0                 
}

RUNPARAM["batch_per_epoch"] = RUNPARAM['num_train']*64/Input['BATCH_SIZE']
RUNPARAM["batch_per_epoch_val"] = RUNPARAM['num_val']*64/Input['BATCH_SIZE']
RUNPARAM['iter_test'] = RUNPARAM['num_test']*64/Input_Test['BATCH_SIZE']

Path={
        "Model_path" : './result/',                 #Path to save the best model where the validation error is the smallest. And then we use this model for test
        "train_data" : './data/',            #path where the  train data is
	"train_result" : './result/',        #path to store the train result
	"val_data" : './data/',              #path where the  validation data is
	"val_result" : './result',          #path to store the validation result
	"test_data" : './data/',              #path where the  test data is
	"test_result" : './result/'           #path to store the test result

}
