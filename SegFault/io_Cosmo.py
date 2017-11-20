import numpy as np
import tensorflow as tf
import hyper_parameters_Cosmo
import os
import itertools

def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()    
    _,single_example = reader.read(filename_queue)
    parsed_example = tf.parse_single_example(
    single_example,
    features = {
        "data_raw": tf.FixedLenFeature([],tf.string),
        "label_raw": tf.FixedLenFeature([],tf.string)
    }
    )
    
    NbodySimuDecode = tf.decode_raw(parsed_example['data_raw'],tf.float64)
    labelDecode = tf.decode_raw(parsed_example['label_raw'],tf.float64)
    NbodySimus = tf.reshape(NbodySimuDecode,[64,64,64])
        
    #augment 
    NbodySimus = tf.cond(tf.random_uniform([1],maxval=1)[0] < tf.constant(.5),lambda:NbodySimus,lambda:NbodySimus[::-1,:,...])
    NbodySimus = tf.cond(tf.random_uniform([1],maxval=1)[0] < tf.constant(.5),lambda:NbodySimus,lambda:NbodySimus[:,::-1,...])
    NbodySimus = tf.cond(tf.random_uniform([1],maxval=1)[0] < tf.constant(.5),lambda:NbodySimus,lambda:NbodySimus[:,:,::-1])
    
    prand = tf.random_uniform([1],maxval=1)[0]
    NbodySimus = tf.cond(prand < tf.constant(1./6),lambda:tf.transpose(NbodySimus, perm = (1,2,0)),lambda:NbodySimus)
    NbodySimus = tf.cond(tf.logical_and(prand < tf.constant(2./6) , prand > tf.constant(1./6)), lambda:tf.transpose(NbodySimus, perm = (1,0,2)),lambda:NbodySimus)
    NbodySimus = tf.cond(tf.logical_and(prand < tf.constant(3./6) , prand > tf.constant(2./6)), lambda:tf.transpose(NbodySimus, perm = (0,2,1)),lambda:NbodySimus)
    NbodySimus = tf.cond(tf.logical_and(prand < tf.constant(4./6) , prand > tf.constant(3./6)), lambda:tf.transpose(NbodySimus, perm = (2,0,1)),lambda:NbodySimus)
    NbodySimus = tf.cond(tf.logical_and(prand < tf.constant(5./6) , prand > tf.constant(4./6)), lambda:tf.transpose(NbodySimus, perm = (2,1,0)),lambda:NbodySimus)
    
    #normalize
    NbodySimus /= (tf.reduce_sum(NbodySimus)/64**3+0.)
    NbodySimuAddDim = tf.expand_dims(NbodySimus,axis = 3)
    label = tf.reshape(labelDecode,[2])
    label = (label - tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float64))/tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float64)
    return NbodySimuAddDim,label
    
def readDataSet(filenames):
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=None,shuffle=True)
    NbodySimus,label= read_tfrecord(filename_queue)
    #NbodyList = [read_tfrecord(filename_queue) for _ in range(hyper_parameters_Cosmo.Input["NUM_THREADS"])]
    NbodySimus_batch, label_batch = tf.train.shuffle_batch(
    	[NbodySimus,label],
	#NbodyList,
    	batch_size = hyper_parameters_Cosmo.Input["BATCH_SIZE"],
    	num_threads = hyper_parameters_Cosmo.Input["NUM_THREADS"],
    	capacity = hyper_parameters_Cosmo.Input["CAPACITY"],
    	min_after_dequeue = hyper_parameters_Cosmo.Input["MIN_AFTER_DEQUEUE"],
	allow_smaller_final_batch=True)
    
    return  NbodySimus_batch, label_batch


def read_test_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _,single_example = reader.read(filename_queue)
    parsed_example = tf.parse_single_example(
    single_example,
    features = {
        "data_raw": tf.FixedLenFeature([],tf.string),
        "label_raw": tf.FixedLenFeature([],tf.string)
    }
    )

    NbodySimuDecode = tf.decode_raw(parsed_example['data_raw'],tf.float64)
    labelDecode = tf.decode_raw(parsed_example['label_raw'],tf.float64)
    NbodySimus = tf.reshape(NbodySimuDecode,[64,64,64])
    NbodySimus /= (tf.reduce_sum(NbodySimus)/64**3+0.)
    NbodySimuAddDim = tf.expand_dims(NbodySimus,3)
    label = tf.reshape(labelDecode,[2])
    labelAddDim = (label - tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float64))/tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float64)
    print NbodySimuAddDim.shape

    return NbodySimuAddDim,labelAddDim
    
def readTestSet(filenames):
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=None,shuffle=False)
    NbodySimus,label= read_test_tfrecord(filename_queue)
    NbodySimus_batch, label_batch = tf.train.batch(
        [NbodySimus,label],
        #NbodyList,
        batch_size = hyper_parameters_Cosmo.Input_Test["BATCH_SIZE"],
        num_threads = hyper_parameters_Cosmo.Input_Test["NUM_THREADS"],
        capacity = hyper_parameters_Cosmo.Input_Test["CAPACITY"],
	enqueue_many=False,
        allow_smaller_final_batch=True)

    return  NbodySimus_batch, label_batch
       
        
