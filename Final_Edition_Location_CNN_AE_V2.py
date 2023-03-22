# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:52:36 2023

@author: u2171379
"""


import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import scipy.io as spio
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from matplotlib import pyplot
from keras.models import load_model
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import time
from keras.layers import Conv2D, Dense, Input, Reshape, Lambda, Layer, Flatten
from keras import backend as K
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
import sys

# Just for debugging tensor objects
#init_op = tf.initialize_all_variables()

#run the graph
#with tf.Session() as sess:
    #sess.run(init_op) #execute init_op
    #print (sess.run())
#%% Define Inverse One-hot 
def inv_one_hot(OH_data):
    max_index_col = np.argmax(OH_data[0])+1
    out = list ()
    for i in range(0,OH_data.shape[0]):
          max_index_col = np.argmax(OH_data[i]) +1
          out.append(max_index_col)
          
    out = np.array(out)
    return out
#%%
# reading data from Matlab files
# profiles = spio.loadmat('profiles_IEEE_39_Bus.mat', squeeze_me=True)
# labels = spio.loadmat('labels_IEEE_39_Bus.mat', squeeze_me=True)
# profiles = profiles ["profiles"]
# labels = labels ["labels"]

# profiles = spio.loadmat('profiles.mat', squeeze_me=True)
# labels = spio.loadmat('labels.mat', squeeze_me=True)
# profiles = profiles ["profile_rand"]
# labels = labels ["labels_rand"]
#%%

class Deep_Net_Structure:

    def __init__(self, CNN_layer_list, Dense_layer_list, Decoder_layer_list, input_shape, output_size, kernel_size, pool_size, stride_size):
        self.CNN_layer_list = np.array(CNN_layer_list)
        self.Dense_layer_list = np.array(Dense_layer_list)
        self.Decoder_layer_list = np.array(Decoder_layer_list)
        self.input_shape = input_shape
        self.output_size = np.array(output_size)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride_size = stride_size
        
    # CNN_layer_list= [512,256,128]
    # Dense_layer_list = [128]
    def Creat_Network(self):
        
        
        input_shape = Input(shape=self.input_shape)  # size of input data is 10 (number of generators) * 100 (number of samples) with 2 channels (delta and omega)     
        # a convolution layer output shape = 10*10*256
        conv1 = Conv2D( self.CNN_layer_list[0], self.kernel_size[0], strides = self.stride_size[0], activation = 'relu', padding = 'valid')(input_shape)
        #Second Convolutional Layer
    
        # convolution layer (256 filters with size 2x2 with and stride 2 without activation function)
        # Input data size is 10x10x256, output size will be 5x5x256.
        # After that squash function will be appllied so output vector values will be maintained between 0 and 1.
        pool1 = MaxPooling2D (pool_size =self.pool_size[0], strides =self.stride_size[1],padding= "valid")(conv1)
        dropout1= Dropout(0.1)(pool1)     
    
        # This dropout layer is added after eliminating the Auto-encoder of Hinton's model, the dropout value is also checked carefully.
        # Input data size is 5x5x256, output size will be 4x4x128.
        conv2 = Conv2D(self.CNN_layer_list[1], self.kernel_size[1], strides = self.stride_size[2], padding = 'same')(dropout1)
        #conv3 = Conv2D(128, (2,2), strides = (1,1), padding = 'valid')(conv2)
        # Input data size is 4x4x128, output size will be 2x2x128.
        pool2 = MaxPooling2D (pool_size =self.pool_size[1], strides =self.stride_size[3], padding= "valid")(conv2)
        dropout2= Dropout(0.1)(pool2) 
        # Flatten data
        flat = Flatten()(dropout2)
        hidden1 = Dense(self.Dense_layer_list[0],activation = 'relu')(flat)
        output = Dense (self.output_size, activation = 'softmax')(hidden1)
        
        
        decoder1 = Dense(self.Decoder_layer_list[0], activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(flat)
        decoder2 = Dense(self.Decoder_layer_list[1], activation = 'relu')(decoder1)
        decoder3 = Dense(self.Decoder_layer_list[2], activation = 'relu')(decoder2)
        decoderoutput = Reshape(self.input_shape)(decoder3)
        
        model = Model(inputs=input_shape, outputs=[output, decoderoutput])
        
        return model
    
class IEEE_14_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
        super().__init__(CNN_layer_list,Dense_layer_list,[256, 512, 5*100*2], (5,100,2),9, ((1,20),(2,2)),((2,2),(2,2)),((1,20),(1,1),(1,1),(1,1)) )    

class IEEE_39_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
        super().__init__(CNN_layer_list,Dense_layer_list,[512, 1024, 10*100*2], (10,100,2),29,((1,10),(2,2)), ((2,2),(2,2)), ((1,10),(2,2),(1,1),(2,2)))

class IEEE_57_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
        super().__init__(CNN_layer_list,Dense_layer_list,[512, 1024, 7*100*2], (7,100,2),50,((1,10),(2,2)), ((2,5),(2,2)), ((1,10),(1,2),(1,1),(2,2)))



#%% Switch case


class SwitchCase:

    def switch(self, Bus_Number):
        default = "Invalid Case"
        return getattr(self, 'IEEE_case_' + str(Bus_Number) + '_Bus', lambda: default)()

    def IEEE_case_14_Bus(self):

        profiles = spio.loadmat('profiles_IEEE_14_Bus_Lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_14_Bus_Lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        
        model = IEEE_14_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model

    def IEEE_case_39_Bus(self):
        
        profiles = spio.loadmat('profiles_IEEE_39_Bus_Lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_39_Bus_Lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        
        model = IEEE_39_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model

    def IEEE_case_57_Bus(self):
        
        profiles = spio.loadmat('profiles_IEEE_57_Bus_Lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_57_Bus_Lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        
        model = IEEE_57_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model


case = SwitchCase()

print('Enter the total bus number of the IEEE case:')
IEEE_case_bus_number = input()

Structure = case.switch(IEEE_case_bus_number)

if Structure == 'Invalid Case':
    sys.exit("Invalid Case, This program is designed for IEEE 14 Bus, IEEE 39 Bus, and IEEE 57 Bus cases")


profiles = Structure[0]
labels = Structure[1]
model = Structure[2]
#%%

X_total = profiles         # Input data (delta and omega of generator buses)
y = labels           # CLasses (number of attacked bus [load buses])
sample_window = 100
X = X_total[:,:,0:sample_window,:]

X = X.astype('float32')
y = y.astype('float32').reshape(y.shape[0],1)

y = np.array(to_categorical(y.astype('float32')))  # define categorical format for each lable (One-hot)
y = np.delete(y,0,1)          # remove the first column (zero class ) (the extra column is added due to the categorical procedure)
training_percentage = 0.8;    # dividing data set to training and test parts
validation_percentage = 0.1;    # dividing data set to training and test parts
divide_data1 = int(training_percentage*profiles.shape[0])  # define a dividing parameter
divide_data2 = int((validation_percentage+training_percentage)*profiles.shape[0])  # define a dividing parameter

x_train = X [0:divide_data1,:]
x_valid = X [divide_data1:divide_data2,:]
x_test = X [divide_data2:len(profiles),:]
y_train = y [0:divide_data1:]
y_valid = y [divide_data1:divide_data2:]
y_test = y [divide_data2:len(profiles),:]


#%%
#Training of model
# model = Model(inputs=[input_shape,decoder_input], outputs=[output, decoderoutput(masked)])
# test_model = Model(inputs=[input_shape,decoder_input], outputs=[output, decoderoutput(masked)])
#plot_model(model, to_file='Convolutional.png')
m = 4
epochs = 150
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model_2DCNN.h5', monitor='val_acc', mode='max', verbose=1, save_best_only = 'True')
#save_freq = 'epoch'
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss= ['categorical_crossentropy','mse'] ,loss_weights = [1., 0.0000005],metrics=['acc'])
history = model.fit([x_train],[y_train, x_train], batch_size = m, epochs = epochs, validation_data = ([x_valid],[y_valid, x_valid]),callbacks=[es, mc])


#%%
#Predict from above trained model
#with CustomObjectScope({'DigitCapsuleLayer': DigitCapsuleLayer},{'loss_fn': loss_fn}):
model = load_model('best_model_2DCNN.h5')

#saved_model = load_model('best_model.h5')
start_time = time.time()
label_predicted_test = model.predict(x_test)
print("--- %s seconds ---" % (time.time() - start_time))
label_predicted_train = model.predict(x_train)
label_predicted_test = model.predict(x_test)
final_accuracy_test = accuracy_score(inv_one_hot(y_test), inv_one_hot(label_predicted_test))
print(final_accuracy_test)
#%%
# plot training history
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

train_acc = np.array(history.history['acc'])
valid_acc = np.array(history.history['val_acc'])

train_loss = np.array(history.history['loss'])
valid_loss = np.array(history.history['val_loss'])

total_acc = np.array([train_acc,valid_acc])
total_acc = np.transpose(total_acc)

total_loss = np.array([train_loss,valid_loss])
total_loss = np.transpose(total_loss)

total_train_labels = np.array([inv_one_hot(y_train),inv_one_hot(label_predicted_train)])
total_train_labels = np.transpose(total_train_labels)

total_test_labels = np.array([inv_one_hot(y_test),inv_one_hot(label_predicted_test)])
total_test_labels = np.transpose(total_test_labels)

workbook = xlsxwriter.Workbook('Results_CNN_15%.xlsx')
worksheet = workbook.add_worksheet('Accuracy')   # training & validation

#write column names
worksheet.write(0, 0, "Traning")
worksheet.write(0, 1, "Validation")

row = 1
for col, data in enumerate(np.transpose(total_acc)):
    worksheet.write_column(row, col, data)

worksheet = workbook.add_worksheet('Loss')     # training & validation

#write column names
worksheet.write(0, 0, "Traning")
worksheet.write(0, 1, "Validation")

row = 1
for col, data in enumerate(np.transpose(total_loss)):
    worksheet.write_column(row, col, data)
    
    
worksheet = workbook.add_worksheet('Training labels')     # training & validation

#write column names
worksheet.write(0, 0, "Traning")
worksheet.write(0, 1, "Test")

row = 1
for col, data in enumerate(np.transpose(total_train_labels)):
    worksheet.write_column(row, col, data)
    


worksheet = workbook.add_worksheet('Test labels')     # training & validation

#write column names
worksheet.write(0, 0, "Traning")
worksheet.write(0, 1, "Test")

row = 1
for col, data in enumerate(np.transpose(total_test_labels)):
    worksheet.write_column(row, col, data)
    
workbook.close()
