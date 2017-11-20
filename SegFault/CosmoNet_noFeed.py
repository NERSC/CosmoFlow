## for this one, change the order between relu and batch

import tensorflow as tf
import numpy as np
from io_Cosmo import *
import hyper_parameters_Cosmo as hp
import time
from numpy import linalg as LA


def weight_variable(shape,name):
	W = tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return W

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def lrelu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CosmoNet:
    def __init__(self,train_data,train_label, val_data = None, val_label = None, test_data = None, test_label = None, is_train = None):
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
	self.test_data = test_data
	self.test_label = test_label
	self.is_train = is_train
        
        self.num_parameters = 1

        #initialize weight and bias
        self.W = {}
        self.b = {}
        self.bn_param = {}
        
        self.W['W_conv1'] = weight_variable([3, 3, 3, 1, 2],'w1')
	self.b['b_conv1'] = bias_variable([2])
	self.W['W_conv2'] = weight_variable([4, 4, 4, 2, 12],'w2')
	self.b['b_conv2'] = bias_variable([12])

	self.W['W_conv3'] = weight_variable([4,4,4,12,64],'w3')
	self.b['b_conv3'] = bias_variable([64])
	self.W['W_conv4'] = weight_variable([3,3,3,64,64],'w4')
	self.b['b_conv4'] = bias_variable([64])
        self.W['W_conv5'] = weight_variable([2,2,2,64,128],'w5')
        self.b['b_conv5'] = bias_variable([128])
	self.W['W_conv6'] = weight_variable([2,2,2,128,128],'w6')
	self.b['b_conv6'] = bias_variable([128])
	self.W['W_fc1'] = weight_variable([1024,1024],'w7')
        self.b['b_fc1'] = bias_variable([1024])
	self.W['W_fc2'] = weight_variable([1024,256],'w8')
        self.b['b_fc2'] = bias_variable([256])
	self.W['W_fc3'] = weight_variable([256,2],'w9')
        self.b['b_fc3'] = bias_variable([2])
        

    #Define some fuctions that might be used   
    
    def BatchNorm(self,inputT, IS_TRAINING, scope,reuse=None):
	with tf.variable_scope(scope,'model',reuse = reuse):
	    return tf.contrib.layers.batch_norm(inputT, is_training=IS_TRAINING,center = True, scale = True,epsilon=0.0001,decay=0.99,scope=scope)

    def deepNet(self,inputBatch,IS_TRAINING,keep_prob,scope,reuse):
        
        # First convolutional layer
        with tf.name_scope('conv1'):
            h_conv1 = lrelu(self.BatchNorm(tf.nn.conv3d(inputBatch,self.W['W_conv1'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv1'],IS_TRAINING = IS_TRAINING, scope = scope+str(1), reuse = reuse),hp.Model['LEAK_PARAMETER'])
        
	with tf.name_scope('pool1'):
            h_pool1 = tf.nn.avg_pool3d(h_conv1, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
           
        #Second convoluational layer
        with tf.name_scope('conv2'):
            h_conv2 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool1, self.W['W_conv2'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv2'],IS_TRAINING=IS_TRAINING,scope = scope+str(2),reuse = reuse),hp.Model['LEAK_PARAMETER'])
            
        with tf.name_scope('pool2'):
            h_pool2 = tf.nn.avg_pool3d(h_conv2, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
        
        #Third convoluational layer
        with tf.name_scope('conv3'):
            h_conv3 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool2, self.W['W_conv3'],strides = [1,2,2,2,1],padding = 'VALID') + self.b['b_conv3'],IS_TRAINING=IS_TRAINING, scope = scope+str(3),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        #Fourth convoluational layer
        with tf.name_scope('conv4'):
            h_conv4 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv3, self.W['W_conv4'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv4'],IS_TRAINING=IS_TRAINING,scope = scope+str(4),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        #Fifth convolutional layer
        with tf.name_scope('conv5'):
            h_conv5 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv4, self.W['W_conv5'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv5'],IS_TRAINING=IS_TRAINING,scope = scope+str(5),reuse=reuse),hp.Model['LEAK_PARAMETER'])
            
        
        #Sixth convolutional layer
        with tf.name_scope('conv6'):
            h_conv6 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv5, self.W['W_conv6'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv6'],IS_TRAINING=IS_TRAINING,scope = scope+str(6),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        with tf.name_scope('fc1'):
            h_conv6_flat = tf.reshape(h_conv6,[-1,1024])
            h_fc1 = lrelu(tf.matmul(tf.nn.dropout(h_conv6_flat,keep_prob), self.W['W_fc1']) + self.b['b_fc1'],hp.Model['LEAK_PARAMETER'])
        
        with tf.name_scope('fc2'):
            h_fc2 = lrelu(tf.matmul(tf.nn.dropout(h_fc1,keep_prob), self.W['W_fc2']) + self.b['b_fc2'],hp.Model['LEAK_PARAMETER'])
            
        with tf.name_scope('fc3'):
            h_fc3 = tf.matmul(tf.nn.dropout(h_fc2,keep_prob), self.W['W_fc3']) + self.b['b_fc3']
            return h_fc3
        
            
    def loss(self):
        with tf.name_scope('loss'):
            predictions = self.deepNet(inputBatch = self.train_data,IS_TRAINING = True,keep_prob = hp.Model['DROP_OUT'],scope='conv_bn',reuse = None)
            lossL1 = tf.reduce_mean(tf.abs(self.train_label-predictions))
            for w in self.W:
                lossL1 += hp.Model["REG_RATE"]*tf.nn.l2_loss(self.W[w])/self.num_parameters
            return lossL1
    
    def validation_loss(self):
        val_predict = self.deepNet(inputBatch = self.val_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        val_true = self.val_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Val = tf.reduce_mean(tf.abs(val_true-val_predict)/val_true)
        return lossL1Val,val_true,val_predict

    def train_loss(self):
	train_predict = self.deepNet(inputBatch = self.train_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        train_true = self.train_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Train = tf.reduce_mean(tf.abs(train_true-train_predict)/train_true)
	return lossL1Train,train_true,train_predict

    def test_loss(self):
        test_predict = self.deepNet(inputBatch = self.test_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        test_true = self.test_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Test = tf.reduce_mean(tf.abs(test_true-test_predict)/test_true)
        return lossL1Test,test_true,test_predict

    def optimize(self):
        loss = self.loss()
        with tf.name_scope('adam_optimizer'):
	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	    with tf.control_dependencies(update_ops):
	        train_step = tf.train.AdamOptimizer(hp.Model['LEARNING_RATE']).minimize(loss)

	lossL1Train,train_true,train_predict = self.train_loss()    
        return train_step, loss,lossL1Train,train_true,train_predict
    
    def train(self):
        train_step, loss, lossL1Train,train_true,train_predict = self.optimize()
        lossL1Val,val_true,val_predict = self.validation_loss()
        lossL1Test,test_true,test_predict = self.test_loss()
        
	config = tf.ConfigProto()


        #used to save the model
        global best_validation_accuracy
        global last_improvement
        global total_iterations
	best_validation_accuracy = 1.0         #Best validation accuracy seen so far
	last_improvement = 0                   #Iteration-number for last improvement to validation accuracy.
	require_improvement = hp.RUNPARAM['require_improvement']               #Stop optimization if no improvement found in this many iterations.
        total_iterations = 0                   #Counter for total number of iterations performed so far.        
	
	if(self.is_train):
            print "training"
            
            with tf.Session() as sess:
        	losses_train = []  
        	losses_val = []
        	losses = []
		val_accuracys = []       
		data_accuracys = []   
        	sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
                coord = tf.train.Coordinator()
        	threads = tf.train.start_queue_runners(coord=coord)
                
		for epoch in range(hp.RUNPARAM['num_epoch']):
                        print epoch
			save_path = os.path.join(hp.Path['Model_path'], 'best_validation')
			total_iterations += 1
			start_time = time.time()
        	        loss_per_epoch_val = 0
        	        loss_per_epoch_train = 0
        	        for i in range(hp.RUNPARAM['batch_per_epoch']): 
				_,lossTrain,lossL1Train_,train_true_,train_predict_ = sess.run([train_step,loss,lossL1Train,train_true,train_predict])                                                       
        	                loss_per_epoch_train +=lossL1Train_


        	        losses.append(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
			losses_train.append(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
			
                        
			for i in range(hp.RUNPARAM['batch_per_epoch_val']):
				loss_,val_true_,val_predict_ = sess.run([lossL1Val,val_true,val_predict])
        	                loss_per_epoch_val += loss_
			losses_val.append(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])

                       
        	        if(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'] < best_validation_accuracy):
				best_validation_accuracy  = loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'] 
				last_improvement = total_iterations
				saver.save(sess=sess, save_path=save_path)
			
			print("Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))
                        print "  training loss: %.3f" %(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
                        print "  validation loss: %.3f" %(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])
                        print "  best loss: %.3f"%best_validation_accuracy	
			np.savetxt(os.path.join(hp.Path['train_result'],'loss_train.txt'),losses_train)
        	        np.savetxt(os.path.join(hp.Path['val_result'],'loss_val.txt'),losses_val)
        	        np.savetxt(os.path.join(hp.Path['train_result'],'losses.txt'),losses)
			if(total_iterations - last_improvement > require_improvement):
				print ("No improvement found in a while, stopping optimization.")
				break
                        
		coord.request_stop();
                coord.join(threads);


                        

	    
	    	

if __name__ == "__main__":
    NbodySimuDataBatch64, NbodySimuLabelBatch64 = readDataSet(filenames = [hp.Path['train_data']+str(i)+'.tfrecord' for i in range(0,4)])
    NbodySimuDataBatch32, NbodySimuLabelBatch32 = tf.cast(NbodySimuDataBatch64,tf.float32),tf.cast(NbodySimuLabelBatch64,tf.float32)
    valDataBatch64, valLabelbatch64 = readDataSet(filenames=[hp.Path['val_data']+str(i)+".tfrecord" for i in range(4,5)]);
    valDataBatch32, valLabelbatch32 = tf.cast(valDataBatch64,tf.float32),tf.cast(valLabelbatch64,tf.float32)
    testDataBatch64, testLabelbatch64 = readTestSet(filenames=[hp.Path['test_data']+str(i)+".tfrecord" for i in range(4,5)]);
    testDataBatch32, testLabelbatch32 = tf.cast(testDataBatch64,tf.float32),tf.cast(testLabelbatch64,tf.float32)


    trainCosmo = CosmoNet(train_data=NbodySimuDataBatch32,train_label=NbodySimuLabelBatch32,val_data=valDataBatch32,val_label=valLabelbatch32,test_data=testDataBatch32,test_label=testLabelbatch32,is_train=True)


    trainCosmo.train()

    
    
            
