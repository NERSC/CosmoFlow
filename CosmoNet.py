# for this one, change the order between relu and batch

import tensorflow as tf
import numpy as np
from io_Cosmo import *
import hyper_parameters_Cosmo as hp
import time
import math

from tensorflow.python.ops import math_ops ##

#import the Cray PE ML Plugin
import ml_comm as mc
import os
import sys
if (sys.version_info > (3, 0)):
    from functools import reduce

if "cori" in os.environ['HOST']:
    os.unsetenv('OMP_NUM_THREADS')
    os.environ['KMP_AFFINITY']  = "compact,norespect"
    os.environ['KMP_HW_SUBSET'] = "64C@2,1T"

#def weight_variable(shape):
#        initial = tf.truncated_normal(shape, stddev=0.1)
#        return tf.Variable(initial)

zscored_average = hp.DATAPARAM['zsAVG']
zscored_std = hp.DATAPARAM['zsSTD']

model_save_interval   = 10 #every 20 epochs
loss_average_interval = 1  #every 5 epochs
verbose               = 0  #print out model info
extra_timers          = 1  #extra perf timers

opt_type = 'ADAM' #  options are: 'ADAM', 'ADAM_LARC', 'ADAM_LARS'
enable_weight_decay = 0 #set to 1 to enable weight decay

if (verbose == 1):
    print('opt_type is: ',opt_type)

cpe_plugin_pipeline_enabled = 0  #set to 1 to enable high performance comm pipeline
cpe_plugin_comm_threads     = 4

if opt_type == 'ADAM':
    adam_beta_corrections       = 0
    lr_scaling_mode             = 1  # 0 = no scaling, 1 = sqrt() scaling, 2 = linear scaling
    adam_lr_decay               = 1
elif (opt_type == 'ADAM_LARC') or (opt_type == 'ADAM_LARS'):
    hp.Model['LEARNING_RATE'] = np.maximum(0.0005, hp.Model['LEARNING_RATE'])
    adam_beta_corrections       = 0
    lr_scaling_mode             = 0
    cpe_plugin_pipeline_enabled = 0 

def weight_variable(shape,name):
    W = tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return W

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CosmoNet:
    def __init__(self,train_data = None,train_label = None, val_data = None, val_label = None, test_data = None, test_label = None, is_train = None, is_test = None, save_path = None):
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
        self.test_data = test_data
        self.test_label = test_label
        self.is_train = is_train
        self.is_test = is_test
        self.save_path = save_path
        #self.num_parameters = 3*3*3*1*2+4*4*4*2*12+4*4*4*12*64+3*3*3*64*64+2*2*2*64*128+2*2*2*128*12+1024*1024+1024*256+256*2
        if enable_weight_decay:
            self.num_parameters = 7041443 # hard code for 3-param topology
        else: 
            self.num_parameters = 1
        self.learning_rate= None ########

        #initialize weight and bias
        self.W = {}
        self.b = {}
        self.bn_param = {}

        self.W['W_conv1'] = weight_variable([3, 3, 3, 1, 16],'w1')
        self.b['b_conv1'] = bias_variable([16])
        self.W['W_conv2'] = weight_variable([4, 4, 4, 16, 32],'w2')
        self.b['b_conv2'] = bias_variable([32])
        self.W['W_conv3'] = weight_variable([4,4,4,32,64],'w3')
        self.b['b_conv3'] = bias_variable([64])
        self.W['W_conv4'] = weight_variable([3,3,3,64,128],'w4')
        self.b['b_conv4'] = bias_variable([128])

        self.W['W_conv5'] = weight_variable([3,3,3,128,256],'w5')
        self.b['b_conv5'] = bias_variable([256])

        self.W['W_conv6'] = weight_variable([2,2,2,256,256],'w6')
        self.b['b_conv6'] = bias_variable([256])

        self.W['W_conv7'] = weight_variable([2,2,2,256,256],'w7')
        self.b['b_conv7'] = bias_variable([256])

        self.W['W_fc1'] = weight_variable([2048,2048],'w8')
        self.b['b_fc1'] = bias_variable([2048])
        self.W['W_fc2'] = weight_variable([2048,256],'w9')
        self.b['b_fc2'] = bias_variable([256])

        self.W['W_fc3'] = weight_variable([256,hp.DATAPARAM['output_dim']],'w10')
        self.b['b_fc3'] = bias_variable([hp.DATAPARAM['output_dim']])

    #Define some fuctions that might be used   
    
    def BatchNorm(self,inputT, IS_TRAINING, scope,reuse=None):
        return inputT
        #with tf.variable_scope(scope,'model',reuse = reuse):
        #    return tf.contrib.layers.batch_norm(inputT, is_training=IS_TRAINING,center = True, scale = True,epsilon=0.0001,decay=0.99,scope=scope)
             #tf.layers.batch_normalization(inputT,training=training,epsilon=0.0001,axis=-1,name=scope)

    def deepNet(self,inputBatch,IS_TRAINING,keep_prob,scope,reuse):
        # First convolutional layer
        with tf.name_scope('conv1'):
            h_conv1 = lrelu(self.BatchNorm(tf.nn.conv3d(inputBatch,self.W['W_conv1'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv1'],IS_TRAINING = IS_TRAINING, scope = scope+str(1), reuse = reuse),hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv1', h_conv1.shape)
        
        with tf.name_scope('pool1'):
            h_pool1 = tf.nn.avg_pool3d(h_conv1, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
            if (verbose == 1): print('h_pool1', h_pool1.shape)
            
        #Second convoluational layer
        with tf.name_scope('conv2'):
            h_conv2 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool1, self.W['W_conv2'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv2'],IS_TRAINING=IS_TRAINING,scope = scope+str(2),reuse = reuse),hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv2', h_conv2.shape)
            
        with tf.name_scope('pool2'):
            h_pool2 = tf.nn.avg_pool3d(h_conv2, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
            if (verbose == 1): print('h_pool2', h_pool2.shape)
        
        #Third convoluational layer
        with tf.name_scope('conv3'):
            h_conv3 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool2, self.W['W_conv3'],strides = [1,2,2,2,1],padding = 'VALID') + self.b['b_conv3'],IS_TRAINING=IS_TRAINING, scope = scope+str(3),reuse=reuse),hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv3', h_conv3.shape)
       
        with tf.name_scope('pool3'):
            h_pool3 = tf.nn.avg_pool3d(h_conv3, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
            if (verbose == 1): print('hpool3', h_pool3.shape)

        #Fourth convoluational layer
        with tf.name_scope('conv4'):
            h_conv4 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv3, self.W['W_conv4'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv4'],IS_TRAINING=IS_TRAINING,scope = scope+str(4),reuse=reuse),hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv4', h_conv4.shape)
       
        #Fifth convolutional layer
        with tf.name_scope('conv5'):
            h_conv5 = lrelu(tf.nn.conv3d(h_conv4, self.W['W_conv5'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv5'],hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv5', h_conv5.shape)
            
        #Sixth convolutional layer
        with tf.name_scope('conv6'):
            h_conv6 = lrelu(tf.nn.conv3d(h_conv5, self.W['W_conv6'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv6'],hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv6', h_conv6.shape)

        #Seventh convolutional layer
        with tf.name_scope('conv7'):
            h_conv7 = lrelu(tf.nn.conv3d(h_conv6, self.W['W_conv7'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv7'],hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hconv7', h_conv7.shape)
       
        with tf.name_scope('fc1'):
            h_conv7_flat = tf.reshape(h_conv7,[-1,2048])
            if (verbose == 1): print('hconv7_flat', h_conv7_flat.shape)
            h_fc1 = lrelu(tf.matmul(tf.nn.dropout(h_conv7_flat,keep_prob), self.W['W_fc1']) + self.b['b_fc1'],hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hfc1', h_fc1.shape)
        
        with tf.name_scope('fc2'):
            h_fc2 = lrelu(tf.matmul(tf.nn.dropout(h_fc1,keep_prob), self.W['W_fc2']) + self.b['b_fc2'],hp.Model['LEAK_PARAMETER'])
            if (verbose == 1): print('hfc2', h_fc2.shape)
            
        with tf.name_scope('fc3'):
            h_fc3 = tf.matmul(tf.nn.dropout(h_fc2,keep_prob), self.W['W_fc3']) + self.b['b_fc3']
            if (verbose == 1): print('hfc3', h_fc3.shape)
            return h_fc3
    
            
    def loss(self):
        with tf.name_scope('loss'):
            predictions = self.deepNet(inputBatch = self.train_data,IS_TRAINING = True,keep_prob = hp.Model['DROP_OUT'],scope='conv_bn',reuse = None)
            lossL1 = tf.reduce_mean(tf.abs(self.train_label-predictions))
            if (enable_weight_decay) and (math_ops.not_equal(hp.Model["REG_RATE"], tf.constant(0.0))):############
                if (verbose == 1):
                    print("weight decay is turned on")
                for w in self.W:
                    lossL1 += hp.Model["REG_RATE"]*tf.nn.l2_loss(self.W[w])/self.num_parameters
            return lossL1
    
    def validation_loss(self):
        val_predict = self.deepNet(inputBatch = self.val_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        val_predict = val_predict*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        val_true = self.val_label*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        lossL1Val = tf.reduce_mean(tf.abs(val_true-val_predict)/val_true)
        return lossL1Val,val_true,val_predict

    def train_loss(self):
        train_predict = self.deepNet(inputBatch = self.train_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        train_predict = train_predict*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        train_true = self.train_label*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        lossL1Train = tf.reduce_mean(tf.abs(train_true-train_predict)/train_true)
        return lossL1Train,train_true,train_predict

    def test_loss(self):
        test_predict = self.deepNet(inputBatch = self.test_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        test_predict = test_predict*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        test_true = self.test_label*tf.constant(zscored_std,dtype = tf.float32)+tf.constant(zscored_average,dtype = tf.float32)
        lossL1Test = tf.reduce_mean(tf.abs(test_true-test_predict)/test_true)
        return lossL1Test,test_true,test_predict

    #lars optimizer:
    def get_lars_optimizer(self,opt_type,loss,global_step,learning_rate,momentum=0.9,LARS_mode="clip",LARS_eta=0.002,LARS_epsilon=1./16000.):
        #set up optimizers
        if opt_type == "Adam":
            optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt_type == "RMSProp":
            optim = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif opt_type == "SGD":
            optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        else:
            raise ValueError("Error, optimizer {} unsupported.".format(opt_type))

        if (verbose == 1):
            print("LARS_mode=",LARS_mode)

        #compute gradients
        grads_and_vars = optim.compute_gradients(loss)
        grads          = mc.gradients([gv[0] for gv in grads_and_vars], 0)
        gs_and_vs      = [(g,v) for (_,v), g in zip(grads_and_vars, grads)]
        for idx, (g, v) in enumerate(gs_and_vs):
            if g is not None:
                v_norm = tf.norm(v, ord='euclidean')
                g_norm = tf.norm(g, ord='euclidean')

                lars_local_lr = tf.cond(
                    pred = math_ops.logical_and( math_ops.not_equal(v_norm, tf.constant(0.0)),
                                            math_ops.not_equal(g_norm, tf.constant(0.0)) ),
                                            true_fn = lambda: LARS_eta * v_norm / g_norm,
                                            false_fn = lambda: LARS_epsilon)
                if LARS_mode == "scale":
                    effective_lr = lars_local_lr
                else:
                    effective_lr = math_ops.minimum(lars_local_lr, 1.0)
                #multiply gradients
                gs_and_vs[idx] = (math_ops.scalar_mul(effective_lr, g), v)

        #apply gradients:
        grad_updates = optim.apply_gradients(gs_and_vs,global_step=global_step)

        # Ensure the train_tensor computes grad_updates.
        with tf.control_dependencies([loss]):
            return grad_updates


    def optimize(self, num_nodes, global_step, beta1, beta2):
        loss = self.loss()
        #opt_type = 'ADAM_LARC' # 'ADAM'  #'ADAM_LARS' #'LARS'
        if opt_type =='ADAM':
            with tf.name_scope('adam_optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):

                    num_batches_per_epoch = 4 #(float(hp.RUNPARAM['num_train']) *hp.magic_number/hp.Input['BATCH_SIZE']/num_nodes) 

                    #use the CPE ML Plugin to average gradients across processes
                    lr = hp.Model['LEARNING_RATE']
                    if (lr_scaling_mode > 0 and adam_lr_decay == 1):
                        boundaries = [np.int64(num_batches_per_epoch * x) for x in [6, 10]]
                        lr1 = lr
                        lr2 = lr
                        if (lr_scaling_mode == 1):
                            lr1 = lr / (0.5*math.sqrt(num_nodes))
                            lr2 = lr / math.sqrt(num_nodes)
                        elif (lr_scaling_mode == 2):
                            lr1 = lr / (0.5*num_nodes)
                            lr2 = lr / num_nodes
                        values = [lr, lr1, lr2]
                        lr = tf.train.piecewise_constant(global_step, boundaries, values)
                        lr = tf.Print(lr, [lr], message="LR = ")

                    optimizer      = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
                    grads_and_vars = optimizer.compute_gradients(loss)
                    grads          = mc.gradients([gv[0] for gv in grads_and_vars], 0)
                    gs_and_vs      = [(g,v) for (_,v), g in zip(grads_and_vars, grads)]
                    train_step     = optimizer.apply_gradients(gs_and_vs, global_step=global_step)
        elif opt_type == 'ADAM_LARC':
            with tf.name_scope('ADAM_LARC_optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.learning_rate = hp.Model['LEARNING_RATE']
                    train_step = self.get_lars_optimizer("Adam",loss,global_step, self.learning_rate,LARS_mode="clip")

        elif opt_type =='ADAM_LARS':
            with tf.name_scope('ADAM_LARS_optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):

                    num_batches_per_epoch = (float(hp.RUNPARAM['num_train']) *hp.magic_number/hp.Input['BATCH_SIZE']/num_nodes)## avoid using the existing hyperparameter as it will be overwritten!!
                    decay_steps = int(num_batches_per_epoch * hp.RUNPARAM['num_epochs_per_decay'])
                    num_steps_for_warmup = int(num_batches_per_epoch * hp.RUNPARAM['LR_warmup_epoch_num'])
                    base_LR=hp.Model['LEARNING_RATE']
                    warmup_start_LR=0.0001 if (num_nodes*hp.Input['BATCH_SIZE'] <=8192) else 0.001

                    def learning_rate_fn(global_step):
                        global_step_val = math_ops.cast(global_step, tf.float32)
                        def b1(): return  (global_step_val * base_LR + (num_steps_for_warmup - global_step_val) * warmup_start_LR)/num_steps_for_warmup
                        def b2(): return  math_ops.cast(base_LR, tf.float32)
                        return tf.cond(tf.less(global_step_val, num_steps_for_warmup), lambda: b1(), lambda: b2())
                    self.learning_rate = learning_rate_fn(global_step)

                    train_step = self.get_lars_optimizer("Adam",loss,global_step, self.learning_rate,LARS_mode="scale")

        lossL1Train,train_true,train_predict = self.train_loss()    
        return train_step, loss,lossL1Train,train_true,train_predict
    
    def train(self):
        global_step = tf.train.get_or_create_global_step()#

        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4

        ### taking config from the MKL benchmarks. 
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = 1 ## for March12 wheel
        config.inter_op_parallelism_threads = 1 ## for March12 wheel
 
        #used to save the model
        saver = tf.train.Saver()
        global best_validation_accuracy
        global last_improvement
        global total_iterations
        best_validation_accuracy = 1.0         #Best validation accuracy seen so far
        last_improvement = 0                   #Iteration-number for last improvement to validation accuracy.
        require_improvement = hp.RUNPARAM['require_improvement']               #Stop optimization if no improvement found in this many iterations.
        total_iterations = 0                   #Counter for total number of iterations performed so far.        

        #Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        
        #initialize the CPE ML Plugin with one team (single thread for now) and the model size
        totsize = sum([reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()])
        mc.init(cpe_plugin_comm_threads, 1, totsize, "tensorflow")
        hp.RUNPARAM['batch_per_epoch']     = int(hp.RUNPARAM['batch_per_epoch'] / mc.get_nranks())
        hp.RUNPARAM['batch_per_epoch_val'] = int(hp.RUNPARAM['batch_per_epoch_val'] / mc.get_nranks())
        totsteps = int(hp.RUNPARAM['num_epoch'] * hp.RUNPARAM['batch_per_epoch'])
        if (cpe_plugin_pipeline_enabled == 1):
            cool_down = - int(0.45 * totsteps)
            mc.config_team(0, 0, cool_down, totsteps, 2, 100)
            beta2 = 0.95
        else:
            mc.config_team(0, 0, totsteps, totsteps, 2, 100)

        #correct the Adam parameters based on the number of ranks, which accounts for effective batch size
        #base values determine at global batch size = 512
        if (adam_beta_corrections == 1 and mc.get_nranks() > 512):
            scale = (hp.Input['BATCH_SIZE'] * mc.get_nranks())/512.0
            beta1 = pow(beta1, scale)
            beta2 = pow(beta2, scale)

        # scale the learning rate by global batch size if requested with 512 global batch size baseline
        if (lr_scaling_mode == 1 and mc.get_nranks() > 512):
            hp.Model['LEARNING_RATE'] = math.sqrt(mc.get_nranks()/512.) * hp.Model['LEARNING_RATE']
        elif (lr_scaling_mode == 2 and mc.get_nranks() > 512):
            hp.Model['LEARNING_RATE'] = (mc.get_nranks()/512.) * hp.Model['LEARNING_RATE'] 
        if (verbose == 1):
            print('Effective learning rate=', hp.Model['LEARNING_RATE'])

        train_step, loss, lossL1Train,train_true,train_predict = self.optimize(mc.get_nranks(), global_step, beta1, beta2)
        lossL1Val,val_true,val_predict = self.validation_loss()
        lossL1Test,test_true,test_predict = self.test_loss()

        if (mc.get_rank() == 0):
            print("+------------------------------+")
            print("| CosmoFlow                    |")
            print("| # Ranks = {:5d}              |".format(mc.get_nranks()))
            print("| Global Batch = {:6d}        |".format(mc.get_nranks() * hp.Input['BATCH_SIZE']))
            print("| # Parameters = {:9d}     |".format(totsize))
            if (cpe_plugin_pipeline_enabled == 1):
                print("| CPE Plugin Pipeline Enabled  |")
            else:
                print("| CPE Plugin Pipeline Disabled |")
            print("+------------------------------+") 
 
        #use the CPE ML Plugin to broadcast initial model parameter values
        new_vars = mc.broadcast(tf.trainable_variables(),0)
        bcast    = tf.group(*[tf.assign(v,new_vars[k]) for k,v in enumerate(tf.trainable_variables())])
     
        if(self.is_train):
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
                    save_path = os.path.join(hp.Path['Model_path'], 'best_validation_epoch_' + str(epoch))
                    total_iterations += 1
                    start_time = time.time()
                    loss_per_epoch_val = 0
                    loss_per_epoch_train = 0
                    for i in range(hp.RUNPARAM['batch_per_epoch']):  
                        step_start_time = time.time()
                        _,lossTrain,lossL1Train_,train_true_,train_predict_, gstep = sess.run([train_step,loss,lossL1Train,train_true,train_predict, global_step])
                        step_finish_time = time.time()
                        
                        elapsed_time += (step_finish_time-step_start_time)
                        samps_per_sec = mc.get_nranks() * (epoch * hp.RUNPARAM['batch_per_epoch'] * hp.Input['BATCH_SIZE'] + (i+1) * hp.Input['BATCH_SIZE']) / elapsed_time
                        samps_per_sec_inst = mc.get_nranks() * hp.Input['BATCH_SIZE'] / (step_finish_time-step_start_time)
                        if (mc.get_rank() == 0):
                            print("Train Step: " + str(i) + ", global step: " + str(gstep) + ", Samples/Sec = " + str(samps_per_sec) + ", Samples/Sec(inst) = " + str(samps_per_sec_inst) + ", Loss = " + str(lossTrain))
                        loss_per_epoch_train +=lossL1Train_

                    if (mc.get_rank() == 0 and extra_timers == 1):
                        print("Training in Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))

                    average_start_time=time.time()
                    if (epoch % loss_average_interval == 0):
                        global_loss = np.array([loss_per_epoch_train],dtype=np.float32)
                        mc.average(global_loss)
                        loss_per_epoch_train = global_loss / hp.RUNPARAM['batch_per_epoch']
                        losses.append(loss_per_epoch_train)
                        losses_train.append(loss_per_epoch_train)
                    if (mc.get_rank() == 0 and extra_timers == 1):
                        print("Training loss averaging in Epoch {} took {:.3f}s".format(epoch, time.time() - average_start_time))

                    val_start_time=time.time()
                    for i in range(hp.RUNPARAM['batch_per_epoch_val']):
                        if (mc.get_rank() == 0):
                            print("Val Step = " + str(i))
                        loss_,val_true_,val_predict_ = sess.run([lossL1Val,val_true,val_predict])
                        loss_per_epoch_val += loss_
                    if (mc.get_rank() == 0 and extra_timers == 1):
                        print("validation in Epoch {} took {:.3f}s".format(epoch, time.time() - val_start_time))
                        
                    average_start_time=time.time()
                    if (epoch % loss_average_interval == 0):
                        global_loss = np.array([loss_per_epoch_val],dtype=np.float32)
                        mc.average(global_loss)
                        loss_per_epoch_val = global_loss / hp.RUNPARAM['batch_per_epoch_val']
                        losses_val.append(loss_per_epoch_val)
                    if (mc.get_rank() == 0 and extra_timers == 1):
                        print("validation loss averaging in Epoch {} took {:.3f}s".format(epoch, time.time() - average_start_time))

                    save_start_time=time.time()
                    if (epoch % model_save_interval == 0) and (loss_per_epoch_val < best_validation_accuracy):
                        best_validation_accuracy  = loss_per_epoch_val
                        last_improvement = total_iterations
                        if (mc.get_rank() == 0):
                            saver.save(sess=sess, save_path=save_path)
                            print("model saving in Epoch {} took {:.3f}s".format(epoch, time.time() - save_start_time))
                                 
                    if (mc.get_rank() == 0):
                        print("Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))
                        print("  training loss: %.3f" %(loss_per_epoch_train))
                        print("  validation loss: %.3f" %(loss_per_epoch_val))
                        print("  best loss: %.3f"%best_validation_accuracy)
                        np.savetxt(os.path.join(hp.Path['train_result'],'loss_train.txt'),losses_train)
                        np.savetxt(os.path.join(hp.Path['val_result'],'loss_val.txt'),losses_val)
                        np.savetxt(os.path.join(hp.Path['train_result'],'losses.txt'),losses)
                        #np.savetxt(os.path.join(hp.Path['train_result'],'train_pred'+str(epoch)+'.txt'),np.c_[train_true_,train_predict_])
                        #np.savetxt(os.path.join(hp.Path['val_result'],'val_pred'+str(epoch)+'.txt'),np.c_[val_true_,val_predict_])
                    if(total_iterations - last_improvement > require_improvement):
                        if (mc.get_rank() == 0):
                            print("No improvement found in a while, stopping optimization.")
                        mc.finalize()
                        break
                
            coord.request_stop()
            coord.join(threads)

        if(self.is_test):
            save_path = os.path.join(hp.Path['Model_path'], 'best_validation')
            if self.save_path != None:
                save_path = self.save_path

            with tf.Session() as sess:
                saver.restore(sess=sess,save_path=save_path)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                loss_test = []
                for i in range(0,int(hp.RUNPARAM['iter_test'])):
                    start_time = time.time()
                    lossL1Test_,test_true_,test_predict_ = sess.run([lossL1Test,test_true,test_predict])
                    loss_test.append(lossL1Test_)
                    if (mc.get_rank() == 0):
                        print("Box {} took {:.3f}s".format(i, time.time() - start_time))
                        print("  test loss: %.3f"%lossL1Test_)
                        np.savetxt(os.path.join(hp.Path['test_result'],'test_batch_'+str(i)+'.txt'),np.c_[test_true_,test_predict_])
                if (mc.get_rank() == 0):
                    np.savetxt(os.path.join(hp.Path['test_result'],'loss_test.txt'),loss_test)
                coord.request_stop()
                coord.join(threads)

        #cleanup the CPE ML Plugin
        mc.finalize()

if __name__ == "__main__":
    """ 
    #use dummy data
    s = 128   
    batch=1 
    NbodySimuDataBatch32 = tf.random_normal([batch, s, s, s, 1], seed=1) 
    NbodySimuLabelBatch32 = tf.random_normal([batch,3], seed=1)  
    valDataBatch32 = tf.random_normal([batch, s, s, s, 1], seed=2) 
    valLabelbatch32 = tf.random_normal([batch,3], seed=2)   
    testDataBatch32 = tf.random_normal([batch, s, s, s, 1], seed=3) 
    testLabelbatch32 = tf.random_normal([batch,3], seed=3) 
    trainCosmo = CosmoNet(train_data=NbodySimuDataBatch32,train_label=NbodySimuLabelBatch32,val_data=valDataBatch32,val_label=valLabelbatch32,test_data=testDataBatch32,test_label=testLabelbatch32,is_train=True, is_test=False)
    trainCosmo.train() 
    """ 

    #use real data
    NbodySimuDataBatch32, NbodySimuLabelBatch32 = readDataSet(filenames = [hp.Path['train_data']+str(i)+'.tfrecord' for i in range(0,(hyper_parameters_Cosmo.RUNPARAM["num_train"]))])
    ###NbodySimuDataBatch32, NbodySimuLabelBatch32 = tf.cast(NbodySimuDataBatch64,tf.float32),tf.cast(NbodySimuLabelBatch64,tf.float32)
    valDataBatch32, valLabelbatch32 = readDataSet(filenames=[hp.Path['val_data']+'/'+str(i)+".tfrecord" for i in range((hyper_parameters_Cosmo.RUNPARAM["num_train"]),(hyper_parameters_Cosmo.RUNPARAM["num_train"]+hyper_parameters_Cosmo.RUNPARAM["num_val"]))]);
    ###valDataBatch32, valLabelbatch32 = tf.cast(valDataBatch64,tf.float32),tf.cast(valLabelbatch64,tf.float32)
    testDataBatch32, testLabelbatch32 = readTestSet(filenames=[hp.Path['test_data']+'/'+str(i)+".tfrecord" for i in range((hyper_parameters_Cosmo.RUNPARAM["num_train"]+hyper_parameters_Cosmo.RUNPARAM["num_val"]),(hyper_parameters_Cosmo.RUNPARAM["num_train"]+hyper_parameters_Cosmo.RUNPARAM["num_val"]+hyper_parameters_Cosmo.RUNPARAM["num_test"]))]);
    ###testDataBatch32, testLabelbatch32 = tf.cast(testDataBatch64,tf.float32),tf.cast(testLabelbatch64,tf.float32)

    trainCosmo = CosmoNet(train_data=NbodySimuDataBatch32,train_label=NbodySimuLabelBatch32,val_data=valDataBatch32,val_label=valLabelbatch32,test_data=testDataBatch32,test_label=testLabelbatch32,is_train=True, is_test=False)
    trainCosmo.train()

    #np.savetxt("losses4.txt",losses)
    #np.savetxt("accuracy4.txt",val_accuracys)
    #np.savetxt("data_accuracy4.txt",data_accuracys)

    
    
            
