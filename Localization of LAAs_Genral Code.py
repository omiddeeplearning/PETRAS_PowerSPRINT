# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:52:36 2023

@author: omiddeeplearning (h.r.jahangir@gmail.com)
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.utils import to_categorical
import scipy.io as spio
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
from keras import regularizers
import sys

    
    
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


# Defining a general class for designing specific network for all cases
# Note that the Deep_Net_Structure class is the main class and other classes are defined as subclasses of this class

class Deep_Net_Structure:
    
    # Initializing the parameters for Deep_Net_Structure class
    
    def __init__(self, CNN_layer_list, Dense_layer_list, Decoder_layer_list, input_shape, output_size, kernel_size, pool_size, stride_size):
        self.CNN_layer_list = np.array(CNN_layer_list)
        self.Dense_layer_list = np.array(Dense_layer_list)
        self.Decoder_layer_list = np.array(Decoder_layer_list)
        self.input_shape = input_shape
        self.output_size = np.array(output_size)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride_size = stride_size
    
    # Defining a function to create a deep neural network
    
    def Creat_Network(self):
        
         # size of input data is  (number of generators) *  (number of samples) * (number of features or channels) 
        input_shape = Input(shape=self.input_shape)     
        # First Convolutional Layer
        conv1 = Conv2D( self.CNN_layer_list[0], self.kernel_size[0], strides = self.stride_size[0], activation = 'relu', padding = 'valid')(input_shape)
        # Adding the pooling layer for the dimesion reduction
        pool1 = MaxPooling2D (pool_size =self.pool_size[0], strides =self.stride_size[1],padding= "valid")(conv1)
        # Adding the Dropout layer for the improving the training against the overfitting
        dropout1= Dropout(0.1)(pool1)     
    
        # Second Convolutional Layer
        conv2 = Conv2D(self.CNN_layer_list[1], self.kernel_size[1], strides = self.stride_size[2], padding = 'same')(dropout1)
        # Adding the pooling layer for the dimesion reduction
        pool2 = MaxPooling2D (pool_size =self.pool_size[1], strides =self.stride_size[3], padding= "valid")(conv2)
        # Adding the Dropout layer for the improving the training against the overfitting
        dropout2= Dropout(0.1)(pool2) 
        # Flatten data
        flat = Flatten()(dropout2)
        
        # Classification Output Layer (Localization task)
        hidden1 = Dense(self.Dense_layer_list[0],activation = 'relu')(flat)
        output = Dense (self.output_size, activation = 'softmax')(hidden1)
        
        # Reconstruction Network (using AE framework with MLP structure)
        decoder1 = Dense(self.Decoder_layer_list[0], activation = 'relu', activity_regularizer=regularizers.l1(10e-5))(flat)
        decoder2 = Dense(self.Decoder_layer_list[1], activation = 'relu')(decoder1)
        decoder3 = Dense(self.Decoder_layer_list[2], activation = 'relu')(decoder2)
        decoderoutput = Reshape(self.input_shape)(decoder3)
        
        # Defining the multi output model (Localization layer and Recunstruction layer)
        model = Model(inputs=input_shape, outputs=[output, decoderoutput])
        
        return model


# Defining the IEEE cases including the IEEE 14-bus system, IEEE 39-bus system, and IEEE 57-bus system
# Note that the following is a subclass of the the Deep_Net_Structure class

class IEEE_14_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
                                                        # Decoder_layers   ,  input,  output, kernel_size,  pool_size,   stride_size
        super().__init__(CNN_layer_list,Dense_layer_list,[256, 512, 5*100*2], (5,100,2),9, ((1,20),(2,2)),((2,2),(2,2)),((1,20),(1,1),(1,1),(1,1)))    

class IEEE_39_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
                                                        # Decoder_layers   ,  input,  output, kernel_size,  pool_size,   stride_size
        super().__init__(CNN_layer_list,Dense_layer_list,[512, 1024, 10*100*2], (10,100,2),29,((1,10),(2,2)), ((2,2),(2,2)), ((1,10),(2,2),(1,1),(2,2)))

class IEEE_57_bus_model(Deep_Net_Structure):

    def __init__(self, CNN_layer_list , Dense_layer_list):
                                                        # Decoder_layers   ,  input,  output, kernel_size,  pool_size,   stride_size
        super().__init__(CNN_layer_list,Dense_layer_list,[512, 1024, 7*100*2], (7,100,2),50,((1,10),(2,2)), ((2,5),(2,2)), ((1,10),(1,2),(1,1),(2,2)))


#%% 

# Defining a Swith class for subclasses of the Deep_Net_Structure 

class SwitchCase:


    # Intialization of the switch class
    
    def switch(self, Bus_Number):
        default = "Invalid Case"
        return getattr(self, 'IEEE_case_' + str(Bus_Number) + '_Bus', lambda: default)()

    # Defining the parameters of the IEEE 14-bus switch class
    
    def IEEE_case_14_Bus(self):
        
        # Reading the input data from the .mat files

        profiles = spio.loadmat('profiles_IEEE_14_Bus_Lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_14_Bus_Lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        model_name = 'IEEE_case_14_Bus'
        
        # Defining the training parameters of this case
        
        reconstruction_training_loss_parameter = 0.0005
        patience_rate = 2
        learning_rate_value = 0.00005
        batch_size = 10
        
        # Creating the network for this case using the subclass IEEE 14-bus case from Deep_Net_Structure class
        
        model = IEEE_14_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model,model_name, reconstruction_training_loss_parameter, patience_rate, learning_rate_value, batch_size

    def IEEE_case_39_Bus(self):
        
        # Reading the input data from the .mat files
        
        profiles = spio.loadmat('profiles_IEEE_39_Bus_lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_39_Bus_lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        model_name = 'IEEE_case_39_Bus'
        
        # Defining the training parameters of this case
        
        reconstruction_training_loss_parameter = 0.0005
        patience_rate = 30
        learning_rate_value = 0.00005
        batch_size = 20
        
        # Creating the network for this case using the subclass IEEE 39-bus case from Deep_Net_Structure class
        
        model = IEEE_39_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model,model_name, reconstruction_training_loss_parameter, patience_rate,learning_rate_value, batch_size

    def IEEE_case_57_Bus(self):
        
        # Reading the input data from the .mat files
        
        profiles = spio.loadmat('profiles_IEEE_57_Bus_lim.mat', squeeze_me=True)
        labels = spio.loadmat('labels_IEEE_57_Bus_lim.mat', squeeze_me=True)
        profiles = profiles ["profiles"]
        labels = labels ["labels"]
        model_name = 'IEEE_case_57_Bus'
        
        # Defining the training parameters of this case
        
        reconstruction_training_loss_parameter = 0.0005
        patience_rate = 10
        learning_rate_value = 0.00005
        batch_size = 20
        
        # Creating the network for this case using the subclass IEEE 57-bus case from Deep_Net_Structure class

        model = IEEE_57_bus_model([512,256,128], [128]).Creat_Network()
        return profiles, labels, model,model_name, reconstruction_training_loss_parameter, patience_rate, learning_rate_value, batch_size


# running the switch class
case = SwitchCase()

# selecting the IEEE case class

print('Enter the total bus number of the IEEE case (14, 39, or 57):')
IEEE_case_bus_number = input()

Structure = case.switch(IEEE_case_bus_number)

if Structure == 'Invalid Case':
    sys.exit("Invalid Case, This program is designed for IEEE 14 Bus, IEEE 39 Bus, and IEEE 57 Bus cases")

# initilizing the parameters based on the seelected class
# Note that for understanding the parameters of the selected class, please refer to the subfunctions in the switch class

profiles = Structure[0]
labels = Structure[1]
model_name = Structure[3]
reconstruction_loss_parameter = Structure[4]
patience_rate = Structure[5]
learning_rate_value = Structure[6]
batch_size_parameter = Structure[7]
     

#%%

X_total = profiles         # Input data (delta and omega of generator buses)
y = labels                 # CLasses (number of attacked bus [load buses])
sample_window = 100        # Number of monitored samples per each observation

X = X_total[:,:,0:sample_window,:]
X = X.astype('float32')
y = y.astype('float32').reshape(y.shape[0],1)

y = np.array(to_categorical(y.astype('float32')))  # define categorical format for each lable (One-hot)
y = np.delete(y,0,1)                               # remove the first column (zero class ) (the extra column is added due to the categorical procedure)
training_percentage = 0.8;                         # dividing data set to training and test parts
validation_percentage = 0.1;                       # dividing data set to training and test parts
divide_data1 = int(training_percentage*profiles.shape[0])                          # define a dividing parameter
divide_data2 = int((validation_percentage+training_percentage)*profiles.shape[0])  # define a dividing parameter

# Defining training and test parts

x_train = X [0:divide_data1,:]
x_valid = X [0:divide_data2,:]
x_test = X [divide_data2:len(profiles),:]
y_train = y [0:divide_data1,:]
y_valid = y [0:divide_data2,:]
y_test = y [divide_data2:len(profiles),:]


#%%


epochs = 1500   # Defining the maximum number of epochs

# Defining early stopping criteria

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= patience_rate )
mc = ModelCheckpoint(str(model_name)+ '_CNN.h5', monitor='val_loss', mode='min', verbose=1, save_best_only = 'True')
#save_freq = 'epoch'


# Definng the operation mode (we have two modes: training and testing)

print('Please select the operation mode (enter test for testing and enter train for training):')
operation_mode = input()


if operation_mode == 'train':
    model =  Structure[2]
    print('Training')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_value),loss= ['categorical_crossentropy','mse'] ,loss_weights = [1., reconstruction_loss_parameter],metrics=['acc'])
    history = model.fit([x_train],[y_train, x_train], batch_size = batch_size_parameter, epochs = epochs, validation_data = ([x_valid],[y_valid, x_valid]),callbacks=[es, mc])
    
    # plot training based on accuracy

    plt.plot(history.history[list(history.history.keys())[3]], label='train')
    plt.plot(history.history[list(history.history.keys())[8]], label='validation')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # plot training based on loss

    plt.plot(history.history[list(history.history.keys())[0]])
    plt.plot(history.history[list(history.history.keys())[5]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    label_predicted_test = model.predict(x_test)
    label_predicted_train = model.predict(x_train)
    label_predicted_test = model.predict(x_test)
    
    # Apply the defined inverse one-hot function for calculating the accuracy
    
    final_accuracy_test = accuracy_score(inv_one_hot(y_test), inv_one_hot(label_predicted_test[0]))
    
    print(final_accuracy_test)
    #%%

    #Train Acc
    train_acc = np.array(history.history[list(history.history.keys())[3]])

    #Valid Acc

    valid_acc = np.array(history.history[list(history.history.keys())[8]])

    #Train Loss
    train_loss = np.array(history.history[list(history.history.keys())[0]])

    #Valid Loss
    valid_loss = np.array(history.history[list(history.history.keys())[5]])

    # preparation for writing the results into the excel file
    
    total_acc = np.array([train_acc,valid_acc])
    total_acc = np.transpose(total_acc)

    total_loss = np.array([train_loss,valid_loss])
    total_loss = np.transpose(total_loss)

    total_train_labels = np.array([inv_one_hot(y_train),inv_one_hot(label_predicted_train[0])])
    total_train_labels = np.transpose(total_train_labels)

    total_test_labels = np.array([inv_one_hot(y_test),inv_one_hot(label_predicted_test[0])])
    total_test_labels = np.transpose(total_test_labels)

    # write the results into the excel file 
    workbook = xlsxwriter.Workbook(str(model_name)+'_CNN.h5'+'.xlsx')
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
    worksheet.write(0, 0, "Actual")
    worksheet.write(0, 1, "Estimated")

    row = 1
    for col, data in enumerate(np.transpose(total_train_labels)):
        worksheet.write_column(row, col, data)
        


    worksheet = workbook.add_worksheet('Test labels')     # training & validation

    #write column names
    worksheet.write(0, 0, "Actual")
    worksheet.write(0, 1, "Estimated")

    row = 1
    for col, data in enumerate(np.transpose(total_test_labels)):
        worksheet.write_column(row, col, data)
        
    workbook.close()
    
elif operation_mode == 'test':
    model = load_model(str(model_name)+ '_CNN.h5')
else:
    sys.exit("Invalid Input data")
    
    
    
#%%
def plot_confusion_matrix(true_labels, predicted_labels):
    classes = np.unique(np.concatenate((true_labels, predicted_labels)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    
    # Create a figure with higher resolution
    fig = plt.figure(figsize=(8, 6), dpi=100)
    
    # Create axes for the plot
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot the confusion matrix as an image
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set the colorbar label
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    # Set the x and y axis labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    
    # Set the tick labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    
    # Set the threshold for text color in the cells
    threshold = cm.max() / 2.0
    
    # Loop over the data dimensions and create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    
    # Set the title of the plot
    ax.set_title("Confusion Matrix")
    
    # Show the plot in a new window
    plt.show()

#%%

def plot_time_series(a,b):
    
    
    profiles_set1 =a[:,0:100]*50+50
    profiles_set2 =b[:,0:100]
    # Determine the x-axis values (assuming equal spacing)
    
    x = np.arange(len(profiles_set1[0]))

    # Create a figure with higher resolution
    fig = plt.figure(figsize=(8, 6), dpi=100)

    # Create subplots for each set of profiles
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Plot the profiles on separate subplots
    for subplot in profiles_set1:
        ax1.plot(x, subplot)

    for subplot in profiles_set2:
        ax2.plot(x, subplot)

    # Set the y-axis limits based on the profiles
    min_val1 = (np.min(profiles_set1))
    max_val1 = (np.max(profiles_set1))
    
    min_val2 = (np.min(profiles_set2))
    max_val2 = (np.max(profiles_set2))
    
    ax1.set_ylim(min_val1, max_val1)
    ax2.set_ylim(min_val2, max_val2)

    # Set the titles for each subplot
    ax1.set_title('Frequency')
    ax2.set_title('Voltage Phase Angle')

    # Set the x-axis label
    fig.text(0.5, 0.04, 'Time', ha='center')
    
    # Set the y-axis label for the first subplot
    fig.text(0.0, 0.75, 'Hz', va='center', rotation='vertical', fontsize=12, color='black')
    ax1.yaxis.set_label_coords(-0.08, 0.5)

    # Set the y-axis label for the second subplot
    fig.text(0.0, 0.25, 'Voltage Phase Angle', va='center', rotation='vertical', fontsize=12, color='black')
    ax2.yaxis.set_label_coords(1.08, 0.5)

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot in a new window
    plt.show()




#%%

# Testing mode
# calculate the accuracy and time

start_time = time.time()
label_predicted_test = model.predict(x_test)
print("--- needed time for the localization task is %s seconds ---" % (time.time() - start_time))
label_predicted_train = model.predict(x_train)
label_predicted_test = model.predict(x_test)
final_accuracy_test = accuracy_score(inv_one_hot(y_test), inv_one_hot(label_predicted_test[0]))
print('localizaion accuracy is =', final_accuracy_test)

#%%

# writing the test outcome into an excel file

total_test_labels = np.array([inv_one_hot(y_test),inv_one_hot(label_predicted_test[0])])
total_test_labels = np.transpose(total_test_labels)

workbook = xlsxwriter.Workbook(str(model_name)+'_CNN.h5_'+'test'+'.xlsx')


worksheet = workbook.add_worksheet('Test labels')     # training & validation
#write column names
worksheet.write(0, 0, "Actual")
worksheet.write(0, 1, "Estimated")

row = 1
for col, data in enumerate(np.transpose(total_test_labels)):
    worksheet.write_column(row, col, data)
    
workbook.close()




#plot_confusion_matrix(total_test_labels[:,0], total_test_labels[:,1])

#plot_time_series(np.reshape(profiles [9,:,:,0], (profiles.shape[1],profiles.shape[2])), np.reshape(profiles [9,:,:,1], (profiles.shape[1],profiles.shape[2])))

