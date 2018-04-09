import numpy as np
import tensorflow as tf
import hyper_parameters_Cosmo
import os
import itertools
import random

def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class loadNpyData:
    def __init__(self,data,label,num):
        ### suggestion from James to cast as 32-bit
        self.data = data.astype(dtype = np.float32) ##data
        self.label = label.astype(dtype = np.float32) ##label
        self.num = num
    
    def convert_to(self):
        filename = str(self.num)+'.tfrecord'
        #print('Writing ', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(len(self.data)):
            data_raw = self.data[index].tostring()
            label_raw = self.label[index].tostring()
            example = tf.train.Example(features = tf.train.Features(feature={'label_raw': _bytes_feature(label_raw),'data_raw': _bytes_feature(data_raw)}))
            writer.write(example.SerializeToString())
        writer.close()

class loadTfrecordData:
    def __init__(self,fileBuffer,num):
        self.fileBuffer = fileBuffer
    
    
    def reconstruct_from(self):
        for filename in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(filename)
            data_raw = (example.features.feature['data_raw'].bytes_list.value[0])
            data = np.fromstring(data_raw, dtype=np.float).reshape([-1,128,128,128,1])
            label_raw = (example.features.feature['label_raw'].bytes_list.value[0])
            label = np.fromstring(label_raw,dtype=np.float).reshape([-1,hyper_parameters_Cosmo.DATAPARAM["output_dim"] ])
            
        return data,label

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

    NbodySimuDecode = tf.decode_raw(parsed_example['data_raw'],tf.float32)
    labelDecode = tf.decode_raw(parsed_example['label_raw'],tf.float32)

    NbodySimus = tf.reshape(NbodySimuDecode,[128,128,128])
 
    #normalize
    NbodySimus /= (tf.reduce_sum(NbodySimus)/128**3+0.)
    NbodySimuAddDim = tf.expand_dims(NbodySimus,axis = 3)
    label = tf.reshape(labelDecode,[hyper_parameters_Cosmo.DATAPARAM["output_dim"] ])


    label = (label - tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsAVG'],dtype = tf.float32))/tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsSTD']
                                                                                                            ,dtype = tf.float32)
    return NbodySimuAddDim,label
    
def readDataSet(filenames):
    #print "---readDataSet-ioCosmo------"
    #print filenames
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=None,shuffle=True)
    NbodySimus,label= read_tfrecord(filename_queue)

    NbodySimus_batch, label_batch = tf.train.shuffle_batch(
    	[NbodySimus,label],
	
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

    NbodySimuDecode = tf.decode_raw(parsed_example['data_raw'],tf.float32)
    labelDecode = tf.decode_raw(parsed_example['label_raw'],tf.float32)
    NbodySimus = tf.reshape(NbodySimuDecode,[128,128,128])
    NbodySimus /= (tf.reduce_sum(NbodySimus)/128**3+0.)
    NbodySimuAddDim = tf.expand_dims(NbodySimus,3)
    #label = tf.reshape(labelDecode,[2])
    label = tf.reshape(labelDecode,[hyper_parameters_Cosmo.DATAPARAM["output_dim"] ])
    
    labelAddDim = (label - tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsAVG'],dtype = tf.float32))/tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsSTD']
                                                                                                                  ,dtype = tf.float32)

    #print NbodySimuAddDim.shape
   
    return NbodySimuAddDim,labelAddDim
    
def readTestSet(filenames):
    #print "----readTestSet-io_cosmo----"
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
       
        

if __name__ == '__main__':

    
    
    label_path = os.path.join('/global/cscratch1/sd/djbard/MUSIC_pyCola/egpbos-pycola-672c58551ff1/OmSiNs/twothousand/','list-500-noCiC-128from256.txt')
    labels = np.loadtxt(label_path,delimiter=',')    
       
    
    ### How many tensorflow files do we want to make? 
    ### Assuming 500 here, with teh first 400 a raondom mix, 
    ### and the last 100 NOT mixed for val/test sets. 
    for i in range(400,500):
        data = []
        label = []
        for j in range(64):
            if i<400:
              numDirectory = random.randrange(1000,1400) ###
            else:
              numDirectory = (i)+1000 ## don't want this to be random!!
            numFile = random.randrange(8)
            dirname = numDirectory

            #print i, j, numDirectory
            ## pull a sub-volumes from the 2000 dir
            data_path = os.path.join('/global/cscratch1/sd/djbard/MUSIC_pyCola/egpbos-pycola-672c58551ff1/OmSiNs/twothousand/128from256-500/',str(dirname).rjust(3,'0'),str(numFile)+'.npy')
            #print data_path
            data = np.append(data,np.load(data_path))
            label = np.append(label,labels[ (numDirectory-1000)][[1,2,3]])
            

        loadNpyData(data.reshape(-1,128,128,128,1),label.reshape(-1,3),i).convert_to()
    
   
