#!/usr/bin/python
# encoding=utf-8
import tensorflow as tf
from tensorflow import keras as k
from tensorflow import feature_column as fc
# from tensorflow.python.feature_column.feature_column_v2 import embedding_column,categorical_column_with_hash_bucket,numeric_column
import numpy as np
import datetime
import random,os,sys,json
import tensorflow.keras.backend as K
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)

# import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding,Dropout,Dense,Input,Layer
from tensorflow.keras.layers import Input, Embedding, Layer, Dense, Dropout, Flatten, Add, BatchNormalization, Activation, Concatenate
from tensorflow.python.keras import regularizers

class FM(Layer):
    def __init__(self,k=10,w_reg=0.001,v_reg=0.001):
        super(FM, self).__init__()
        self.k=k;
        self.w_reg=w_reg;
        self.v_reg=v_reg
        pass
    def build(self, input_shape):
        self.w0=self.add_weight(name='w0',shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True
                                )
        self.w=self.add_weight(name='w',shape=(input_shape[-1],1),
                               initializer='random_uniform',
                               regularizer=l2(self.w_reg),
                               trainable=True
                               )
        self.v=self.add_weight(name='v',shape=(self.k,input_shape[-1]),
                               initializer='random_uniform',
                               regularizer=l2(self.v_reg),
                               trainable=True
                               )
        pass

    def call(self, inputs, **kwargs):
        first_order=self.w0+tf.matmul(inputs,self.w) #inputs [batch,featurenumber]
        second_order= 0.5*tf.reduce_sum(tf.pow(tf.matmul(inputs,tf.transpose(self.v)),2)
                                        -tf.matmul(tf.pow(inputs,2),tf.pow(tf.transpose(self.v),2))
                                        ,
                                        axis=1,
                                        keepdims=True
                                        );
        return first_order+second_order;
        pass


embedding_size=16
deep_layers=[128,64,32]
lr = 0.001
deep_l2 = 0.001
log_tag = 'default'
epochs=20

feat_bins = ""
with open('./deepmodel/conf/feat_bins_fm.json') as file:
    feat_bins = json.load(file)


class ReadTFData:
    def __init__(self, schema_path):
        self.feature_schema = self.get_schema(schema_path)
    def get_schema(self, schema_path):
        schema = {}
        with open(schema_path,'r') as load_f:
             schema_json = json.load(load_f)
        for conf in schema_json: 
            if conf['feat_type'] == 'conti':
                schema[conf['featureName']] = tf.io.FixedLenFeature([], tf.float32, default_value = -1.0) # defaulte -1
            elif conf['feat_type'] == 'cate':
                if conf['length'] == 1:
#                     schema[conf['featureName']] = tf.io.VarLenFeature(tf.string)
                    schema[conf['featureName']] = tf.io.FixedLenFeature([], tf.string, default_value = "-1")
                else:
                    schema[conf['featureConf']] = tf.io.VarLenFeature(tf.string)
        schema['label']= tf.io.FixedLenFeature([], tf.float32, default_value = 0)
        return schema

    def load_from_tfrecord(self, file_name, batch_size, shuffle_size):
        print("load: " + file_name)
        if "train" in file_name:
            return tf.data.experimental.make_batched_features_dataset(
                file_pattern = file_name,
                batch_size = batch_size,
                features = self.feature_schema,
                label_key = "label",
                shuffle = True,
                shuffle_buffer_size = shuffle_size,
                shuffle_seed = 2021,#random.randint(0, 1000000),
#这个ecpoch在fit的时候传入，在这里不设置                
#                 num_epochs = 10, 
                drop_final_batch = True)
        elif "test" in file_name:
            return tf.data.experimental.make_batched_features_dataset(
                file_pattern = file_name,
                batch_size = batch_size,
                features = self.feature_schema,
                label_key = "label",
                shuffle = True,
                shuffle_buffer_size = shuffle_size,
                shuffle_seed = 2021,#random.randint(0, 1000000),
                num_epochs = 1,
                drop_final_batch = True)


root_path = "${hdfs_root_path}"
model_root_path = "${hdfs_model_path}"
train_path = root_path + "train/*"
test_path = root_path + "test/*"
schema_path = "${schema_path}"
model_path = model_root_path + "saved_model"


class CustomizedModel:
    def __init__(self, schema_path, layer_path):
        tf.random.set_seed(6)
        feature_columns, feature_input = self.__build_features(schema_path)
        self.model = self.__build_model(feature_columns, feature_input)
    
    def __get_emb(self, conf):
        feat_type = conf['feat_type']
        name = conf['featureName']
        if feat_type == "cate":
            if conf['length'] == 1:
                name = conf['featureName']
            else:
                name = conf['featureConf']
                
            feature_input = k.layers.Input(shape=(), name = name, dtype = tf.string)  # shape
            col = fc.categorical_column_with_hash_bucket(name, hash_bucket_size=conf['bucket'], dtype = tf.string) #fc
            feature_column = fc.embedding_column(col, dimension = embedding_size, trainable = True)  #fc
        elif feat_type == "conti":
            name = conf['featureName']
            if name in list(feat_bins.keys()):
                feature_column = fc.numeric_column(name,default_value=-1.0) #fc
                feature_column = fc.bucketized_column(feature_column,feat_bins[name])
                feature_column = fc.embedding_column(feature_column, dimension = embedding_size, trainable = True)
                feature_input = k.layers.Input(shape=(), name=name, dtype=tf.float32)
        else:
            print("error column_name: {0}".format(name))
        return feature_column, feature_input
    
    def __build_features(self, schema_info):
        cate=[]
        con = []
        feature_columns = []
        feature_inputs = {}
        with open(schema_path,'r') as load_f:
             schema_json = json.load(load_f)
        for conf in schema_json: 
            flag = 1
            if conf['feat_type'] == 'cate' and conf["length"] == 1:
                cate.append(conf['featureName'])
                flag = 0
            if conf['feat_type'] == 'conti':
                con.append(conf['featureName'])
                flag = 0
            if flag == 1:
                continue
            feature_column, feature_input = self.__get_emb(conf)
            feature_columns.append(feature_column)
            if conf["feat_type"] == 'cate' and conf["length"] != 1:
                name = conf["featureConf"]
            else:
                name = conf["featureName"]
            feature_inputs[name] = feature_input
        print("cate:",cate)
        print("con:",con)
        return feature_columns, feature_inputs

    def __build_model(self, feature_columns, feature_inputs):
        # input
        inputs = feature_inputs
        dense_features = [
            tf.keras.layers.DenseFeatures([feature_columns[i]], name='dense_feature_'+fc_key)(
                {fc_key: feature_inputs[fc_key]}
            ) for i, fc_key in enumerate(list(feature_inputs.keys()))
        ]
        dense_features = [
        K.expand_dims(fc, axis=1) for fc in dense_features
            ]
        fm_inputs = K.concatenate(dense_features, axis=-1)

        fm_outputs=FM()(Flatten()(fm_inputs))
        deep = Flatten()(fm_inputs)
        for units in deep_layers:
            deep_p = Dense(units, kernel_regularizer=regularizers.l2(deep_l2))(deep)
#             bn = BatchNormalization()(deep_p)
#             deep = Activation('relu')(bn)
            deep = Activation('relu')(deep_p)

        deep_output = Dense(1)(deep)

        merge = tf.add(fm_outputs,deep_output)
        output=tf.nn.sigmoid(merge)

        opt = tf.keras.optimizers.Adam(lr=lr)
        merge_model = tf.keras.Model(inputs, output)
        merge_model.summary()
        merge_model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )

        return merge_model
    
    def fit(self, train_dataset, test_dataset):
        print("Fit model on training data")
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
        )
#         early_stop = k.callbacks.EarlyStopping(monitor = 'val_auc',verbose=2)
        t1 = datetime.datetime.now()
        self.model.fit(
            train_dataset, epochs=epochs, steps_per_epoch=200, #100
            validation_data=test_dataset,
            validation_steps=50,
            validation_freq=1,
            callbacks=[earlystop]
        )
        t2 = datetime.datetime.now()
        print("done!")
        print(t2-t1)
        return self.model
def train():
    read_tfdata = ReadTFData(schema_path)
    
    train_dataset = read_tfdata.load_from_tfrecord(file_name = train_path, batch_size = 1000, shuffle_size = 200)
    test_dataset = read_tfdata.load_from_tfrecord(file_name = test_path, batch_size = 1000, shuffle_size = 200)
    #print(train_dataset)
    
    cm = CustomizedModel(schema_path,"layer_test")
    model = cm.fit(train_dataset,test_dataset)
#     model.save(model_path, save_format='tf')
    model.evaluate(test_dataset)


    
def main():
    train()
    print("finish")
    
    
if __name__ == '__main__':
    main()
