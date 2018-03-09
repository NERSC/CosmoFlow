import numpy as np
import tensorflow as tf
import hyper_parameters_Cosmo
import os
import itertools



def _float64_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class loadNpyData:
    def __init__(self,data,label,num):
        self.data = data
        self.label = label
        self.num = num
    
    def convert_to(self):
        filename = str(self.num)+'.tfrecord'
        print('Writing ', filename)
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
            data = np.fromstring(data_raw, dtype=np.float).reshape([-1,64,64,64,1])
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
    label = tf.reshape(labelDecode,[hyper_parameters_Cosmo.DATAPARAM["output_dim"] ])
    ###
    ### 0.3, 0.02853, 0.8628, 0.04887, 0.701,0.05691
    label = (label - tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsAVG'],dtype = tf.float64))/tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsSTD']
,dtype = tf.float64)
    return NbodySimuAddDim,label
    
def readDataSet(filenames):
    print "---readDataSet-ioCosmo------"
    print filenames
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
    label = tf.reshape(labelDecode,[hyper_parameters_Cosmo.DATAPARAM["output_dim"] ])
    ### 0.3, 0.02853, 0.8628, 0.04887, 0.701,0.05691
    labelAddDim = (label - tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsAVG'],dtype = tf.float64))/tf.constant(hyper_parameters_Cosmo.DATAPARAM['zsSTD']
,dtype = tf.float64)
    print NbodySimuAddDim.shape
   


    return NbodySimuAddDim,labelAddDim
    
def readTestSet(filenames):
    print "----readTestSet-io_cosmo----"
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
       
        


######## for genererating data
if __name__ == '__main__':
    order = np.random.permutation(64*400)
    order = np.split(np.append(order,np.arange(64*400,64*499)),499)
    
    label_path = os.path.join('/global/cscratch1/sd/djbard/MUSIC_pyCola/egpbos-pycola-672c58551ff1/OmSiH/','basic_info_3.txt')
    labels = np.loadtxt(label_path,delimiter=',')    
       
    for i in range(0,499):
        data = []
        label = []
        for j in order[i]:
            numDirectory = int(j/64)
            numFile = j%64
            data_path = os.path.join('/global/cscratch1/sd/djbard/cosmoML/NbodySimu/',str('01')+str(numDirectory).rjust(3,'0'),str(numFile)+'.npy')
            data = np.append(data,np.load(data_path))
            label = np.append(label,labels[numDirectory][[1,4]])
        loadNpyData(data.reshape(-1,64,64,64,1),label.reshape(-1,3),i).convert_to()
    
