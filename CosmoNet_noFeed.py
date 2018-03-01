## for this one, change the order between relu and batch

import tensorflow as tf
import numpy as np
from io_Cosmo import *
import hyper_parameters_Cosmo as hp
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import time
from numpy import linalg as LA

#import the Cray PE ML Plugin
import ml_comm as mc

#def weight_variable(shape):
#        initial = tf.truncated_normal(shape, stddev=0.1)
#        return tf.Variable(initial)


def weight_variable(shape,name):
	W = tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return W

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def lrelu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CosmoNet:
    def __init__(self,train_data = None,train_label = None, val_data = None, val_label = None, test_data = None, test_label = None, is_train = None, is_test = None):
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
	self.test_data = test_data
	self.test_label = test_label
	self.is_train = is_train
	self.is_test = is_test
        
        #self.num_parameters = 3*3*3*1*2+4*4*4*2*12+4*4*4*12*64+3*3*3*64*64+2*2*2*64*128+2*2*2*128*12+1024*1024+1024*256+256*2
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
   	    #tf.layers.batch_normalization(inputT,training=training,epsilon=0.0001,axis=-1,name=scope)

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
        val_predict = val_predict*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        val_true = self.val_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Val = tf.reduce_mean(tf.abs(val_true-val_predict)/val_true)
        return lossL1Val,val_true,val_predict

    def train_loss(self):
	train_predict = self.deepNet(inputBatch = self.train_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        train_predict = train_predict*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        train_true = self.train_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Train = tf.reduce_mean(tf.abs(train_true-train_predict)/train_true)
	return lossL1Train,train_true,train_predict

    def test_loss(self):
        test_predict = self.deepNet(inputBatch = self.test_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        test_predict = test_predict*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        test_true = self.test_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Test = tf.reduce_mean(tf.abs(test_true-test_predict)/test_true)
        return lossL1Test,test_true,test_predict

    def optimize(self):
        loss = self.loss()
        with tf.name_scope('adam_optimizer'):
	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	    with tf.control_dependencies(update_ops):

		#use the CPE ML Plugin to average gradients across processes
		optimizer      = tf.train.AdamOptimizer(hp.Model['LEARNING_RATE'])
		grads_and_vars = optimizer.compute_gradients(loss)
		grads          = mc.gradients([gv[0] for gv in grads_and_vars], 0)
		gs_and_vs      = [(g,v) for (_,v), g in zip(grads_and_vars, grads)]
		train_step     = optimizer.apply_gradients(gs_and_vs)


	lossL1Train,train_true,train_predict = self.train_loss()    
        return train_step, loss,lossL1Train,train_true,train_predict
    
    def train(self):
        train_step, loss, lossL1Train,train_true,train_predict = self.optimize()
        lossL1Val,val_true,val_predict = self.validation_loss()
        lossL1Test,test_true,test_predict = self.test_loss()
        
	config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
 
        ### taking config from the MKL benchmarks. 
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = 1 ## default
        config.inter_op_parallelism_threads = 2 ## Default



        #used to save the model
	saver = tf.train.Saver()
        global best_validation_accuracy
        global last_improvement
        global total_iterations
	best_validation_accuracy = 1.0         #Best validation accuracy seen so far
	last_improvement = 0                   #Iteration-number for last improvement to validation accuracy.
	require_improvement = hp.RUNPARAM['require_improvement']               #Stop optimization if no improvement found in this many iterations.
        total_iterations = 0                   #Counter for total number of iterations performed so far.        

	if(self.is_train):

            #initialize the CPE ML Plugin with one team (single thread for now) and the model size
            totsize = sum([reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()])
            mc.init(1, 1, totsize, "tensorflow")
            hp.RUNPARAM['batch_per_epoch'] = hp.RUNPARAM['batch_per_epoch'] / mc.get_nranks()
            totsteps = hp.RUNPARAM['num_epoch'] * hp.RUNPARAM['batch_per_epoch']
            mc.config_team(0, 0, totsteps, totsteps, 2, 50)

            if (mc.get_rank() == 0):
                print("+------------------------------+")
                print("| CosmoFlow                    |")
                print("| # Ranks = {:5d}              |".format(mc.get_nranks()))
                print("| Global Batch = {:6d}        |".format(mc.get_nranks() * hp.Input['BATCH_SIZE']))
		print("| # Parameters = {:9d}     |".format(totsize))
                print("+------------------------------+") 
 
            #use the CPE ML Plugin to broadcast initial model parameter values
            new_vars = mc.broadcast(tf.trainable_variables(),0)
            bcast    = tf.group(*[tf.assign(v,new_vars[k]) for k,v in enumerate(tf.trainable_variables())])

            with tf.Session(config=config) as sess:
        	losses_train = []  
        	losses_val = []
        	losses = []
		val_accuracys = []       
		data_accuracys = []   

                #do all parameter initializations
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
                sess.run(bcast)
		
        	coord = tf.train.Coordinator()
        	threads = tf.train.start_queue_runners(coord=coord)

                elapsed_time = 0.
		for epoch in range(hp.RUNPARAM['num_epoch']):
			save_path = os.path.join(hp.Path['Model_path'], 'best_validation')
			total_iterations += 1
			start_time = time.time()
        	        loss_per_epoch_val = 0
        	        loss_per_epoch_train = 0
        	        for i in range(hp.RUNPARAM['batch_per_epoch']): 
				step_start_time = time.time()
				_,lossTrain,lossL1Train_,train_true_,train_predict_ = sess.run([train_step,loss,lossL1Train,train_true,train_predict])
                                step_finish_time = time.time()
				
                                elapsed_time += (step_finish_time-step_start_time)
                                samps_per_sec = mc.get_nranks() * (epoch * hp.RUNPARAM['batch_per_epoch'] * hp.Input['BATCH_SIZE'] + (i+1) * hp.Input['BATCH_SIZE']) / elapsed_time
                                if (mc.get_rank() == 0):
                                  print("Train Step: " + str(i) + ", Samples/Sec = " + str(samps_per_sec) + ", Loss = " + str(lossTrain))
                               
        	                loss_per_epoch_train +=lossL1Train_
        	        losses.append(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
			losses_train.append(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
			
                        
			for i in range(hp.RUNPARAM['batch_per_epoch_val']):
                                if (mc.get_rank() == 0):
                                  print("Val Step = " + str(i))
				loss_,val_true_,val_predict_ = sess.run([lossL1Val,val_true,val_predict])
        	                loss_per_epoch_val += loss_
			losses_val.append(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])

                       
        	        if(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'] < best_validation_accuracy):
				best_validation_accuracy  = loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'] 
				last_improvement = total_iterations
				if (mc.get_rank() == 0):
					saver.save(sess=sess, save_path=save_path)

			if (mc.get_rank() == 0):
				print("Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))
				print "  training loss: %.3f" %(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
				print "  validation loss: %.3f" %(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])
				print "  best loss: %.3f"%best_validation_accuracy	
				np.savetxt(os.path.join(hp.Path['train_result'],'loss_train.txt'),losses_train)
				np.savetxt(os.path.join(hp.Path['val_result'],'loss_val.txt'),losses_val)
				np.savetxt(os.path.join(hp.Path['train_result'],'losses.txt'),losses)
		                #np.savetxt(os.path.join(hp.Path['train_result'],'train_pred'+str(epoch)+'.txt'),np.c_[train_true_,train_predict_])
        	                #np.savetxt(os.path.join(hp.Path['val_result'],'val_pred'+str(epoch)+'.txt'),np.c_[val_true_,val_predict_])
			if(total_iterations - last_improvement > require_improvement):
				if (mc.get_rank() == 0):
					print ("No improvement found in a while, stopping optimization.")
				break		                        

		coord.request_stop();
                coord.join(threads);

            #cleanup the CPE ML Plugin
            mc.finalize()

	if(self.is_test):
                
                with tf.Session() as sess:
	    		saver.restore(sess=sess,save_path=save_path)
			coord = tf.train.Coordinator()
                	threads = tf.train.start_queue_runners(coord=coord)
            		loss_test = []
            		for i in range(0,hp.RUNPARAM['iter_test']):
				start_time = time.time()
		    		lossL1Test_,test_true_,test_predict_ = sess.run([lossL1Test,test_true,test_predict])
		    		loss_test.append(lossL1Test_)	
				print("Box {} took {:.3f}s".format(i, time.time() - start_time))
				print "  test loss: %.3f"%lossL1Test_
	    		        np.savetxt(os.path.join(hp.Path['test_result'],'test_batch_'+str(i)+'.txt'),np.c_[test_true_,test_predict_])
	    		np.savetxt(os.path.join(hp.Path['test_result'],'loss_test.txt'),loss_test)
                	coord.request_stop()
			coord.join(threads)
   

	    
	    	

if __name__ == "__main__":
    NbodySimuDataBatch64, NbodySimuLabelBatch64 = readDataSet(filenames = [hp.Path['train_data']+str(i)+'.tfrecord' for i in range(0,400)])
    NbodySimuDataBatch32, NbodySimuLabelBatch32 = tf.cast(NbodySimuDataBatch64,tf.float32),tf.cast(NbodySimuLabelBatch64,tf.float32)
    valDataBatch64, valLabelbatch64 = readDataSet(filenames=[hp.Path['val_data']+str(i)+".tfrecord" for i in range(400,450)]);
    valDataBatch32, valLabelbatch32 = tf.cast(valDataBatch64,tf.float32),tf.cast(valLabelbatch64,tf.float32)
    testDataBatch64, testLabelbatch64 = readTestSet(filenames=[hp.Path['test_data']+str(i)+".tfrecord" for i in range(450,499)]);
    testDataBatch32, testLabelbatch32 = tf.cast(testDataBatch64,tf.float32),tf.cast(testLabelbatch64,tf.float32)


    trainCosmo = CosmoNet(train_data=NbodySimuDataBatch32,train_label=NbodySimuLabelBatch32,val_data=valDataBatch32,val_label=valLabelbatch32,test_data=testDataBatch32,test_label=testLabelbatch32,is_train=True, is_test=False)

    trainCosmo.train()
    #np.savetxt("losses4.txt",losses)
    #np.savetxt("accuracy4.txt",val_accuracys)
    #np.savetxt("data_accuracy4.txt",data_accuracys)
