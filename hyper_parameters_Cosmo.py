magic_number = 64

DATAPARAM={
        "output_dim" : 2,
	#Ns
        #"zsAVG" : [0.3, 0.8628, 0.95],
        #"zsSTD" : [0.02853, 0.04887, 0.028]
	#H0
        #"zsAVG" : [0.3, 0.8628, 0.701],
        #"zsSTD" : [0.02853, 0.04887, 0.05691]

        "zsAVG": [2.995679839999998983e-01,8.610806619999996636e-01],
        "zsSTD": [2.905168635566176411e-02,4.023372385668218254e-02]
        }
Input = {
        "BATCH_SIZE" : 1,              #mini-batch size for training and validation
        "NUM_THREADS" : 2,              #number of threads to read data
        "CAPACITY" : 0,
        "MIN_AFTER_DEQUEUE" : 200       #the minimum number in the queue after dequeue (Min_after_dequeue and capacity together determines the shuffling of input data)
        }

Input["CAPACITY"] = Input["BATCH_SIZE"]*4 + Input["MIN_AFTER_DEQUEUE"]

Input_Test = {
	"BATCH_SIZE" : 64,              #mini-batch size for test data
	"NUM_THREADS" : 2,              #number of threads to read data
	"CAPACITY" : 0,
	"MIN_AFTER_DEQUEUE" : 64
	}

Input_Test["CAPACITY"] = Input_Test["BATCH_SIZE"]*4 + Input_Test["MIN_AFTER_DEQUEUE"]

Model = {
        "REG_RATE": 0.,                 #regularization of weights: currently set to 0 since batch_normalization has the same effect of regularization
        "LEAK_PARAMETER": 0.01,         #leaky parameter for leaky relu
        "LEARNING_RATE" : 0.0001,       #adam_optimizer to do the update. 
        "DROP_OUT": 0.5                 #apply drop out in fully connected layer. this value gives the probabilty of keep the node. 
}

RUNPARAM={
	"num_epoch": 70,              #each epoch means a fully pass over the data. The program might stop before running num_epoch (see next line).        
        "require_improvement": 20,      #if with require_improvement, there is no improvement in validation error, then stop running. 
	"num_train":400,                #total number of simulations for training
	"num_val":50,                   #total number of simulations for validation
        "num_test":49,                  #total number of simulations for testing
	"batch_per_epoch":0,             
	"batch_per_epoch_val":0,
        "iter_test":0                 
}

RUNPARAM["batch_per_epoch"] = RUNPARAM['num_train']*magic_number/Input['BATCH_SIZE']
RUNPARAM["batch_per_epoch_val"] = RUNPARAM['num_val']*magic_number/Input['BATCH_SIZE']
RUNPARAM['iter_test'] = RUNPARAM['num_test']*magic_number/Input_Test['BATCH_SIZE']

#target_dir = "new_data_2"
#main_dir = '/data1/jamesarnemann/cosmoNet/'
#target_dir = "new_data_3_param_1"

##### CHANGE THIS TO LOCAL DIRECTORY
#main_dir = "/data0/jamesarnemann/cosmoNet/"
#target_dir = "orig_paper"
main_dir = '/lus/scratch/p02472/cosmoflow/'
target_dir = 'new_data_3_param_2'

#######
#main_dir = '/data0/jamesarnemann/cosmoNet/'
#target_dir = "new_data_3_param_n0"


#/data0/jamesarnemann/cosmoNet/new_data_3_param_n0/
#target_path = 
result_dir = '/result/'



Path={

	"init_data" :  '.',                 #Path where the init data is
        "Model_path" : main_dir + target_dir + result_dir,                 #Path to save the best model where the validation error is the smallest. And then we use this model for test
        "train_data" : main_dir + target_dir + '/data/train/',            #path where the  train data is
	"train_result" : main_dir + target_dir + result_dir,        #path to store the train result
	"val_data" : main_dir + target_dir + '/data/val/',              #path where the  validation data is
	"val_result" : main_dir + target_dir + result_dir,          #path to st/data0/jamesarnemann/cosmoNet/' + target_dir + '/result/'ore the validation result
	"test_data" : main_dir + target_dir + '/data/test/',              #path where the  test data is
	"test_result" : main_dir + target_dir + result_dir,           #path to store the test result

}  
