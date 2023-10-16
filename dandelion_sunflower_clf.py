# import required libraries 
import os
import zipfile
import random
import tensorflow as tf #tf is the library used for ML
from tensorflow.keras.optimizers import RMSprop #Keras is the framework inside the Tensorflow library which makes making neural networks easier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# jrc: relative paths would make this more portable
local_zip = 'C:/Users/mhayat22.THADENSCHOOL/Downloads/ai/archivedataset.zip' #connect variable to a path to the dataset file
zip_ref   = zipfile.ZipFile(local_zip, 'r') #zipfile.zipfile is a class for reading/writing zip files, first parameter is the file itself, the second should always be r to read an existing file 
zip_ref.extractall('/ai') #extracts everything in the folder "ai"
zip_ref.close() #calling close is important to let new data to be written/zipped out in the folder

'''mkdir makes directories, it will make new folders in the ai folder'''
try: 
    os.mkdir('/ai/dandelion-sunflower')
    os.mkdir('/ai/dandelion-sunflower/training')
    os.mkdir('/ai/dandelion-sunflower/testing')
    os.mkdir('/ai/dandelion-sunflower/training/dandelion')
    os.mkdir('/ai/dandelion-sunflower/training/sunflower')
    os.mkdir('/ai/dandelion-sunflower/testing/dandelion')
    os.mkdir('/ai/dandelion-sunflower/testing/sunflower')
except OSError: #exception called when program detecs a system-related error
    pass



#Function to split the data to be put inside the new folders
def split_data(DATA_SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = [] #create list of files
    for filename in os.listdir(DATA_SOURCE): #for every file in chosen folder for the new data to be put in(os.listdir returns a list of all files in the current directory)
        file = DATA_SOURCE + filename #create a variable called file for every file scanned in a particular folder, assign it the value of the name of the folder and the filename
        if os.path.getsize(file) > 0: #check if the byte size of the file in the path is > 0 
            files.append(filename) #if so, add the file to the files list
        else:
            print(filename + " is zero length, so ignoring.") #if file is less than zero, than it means its corrupted, doesn't exist or inaccessible

    training_length = int(len(files) * SPLIT_SIZE)#size of the folder of the training set, takes all files in the files list, divides it by the SPLIT_SIZE, this is important so training set doesnt have all the files with no new files for testing set to use.
    testing_length = int(len(files) - training_length) #the size of the training_length - the total length of files would be the length of the testing set
    shuffled_set = random.sample(files, len(files)) #uses random.sample to shuffle the data set, random.sample takes a sequence like a list first, and then the second parameter is the length of the sample, which is the total length of the list files
    training_set = shuffled_set[0:training_length] #the training set will consist of shuffled data of range of 0 to the training length
    testing_set = shuffled_set[-testing_length:] #testing set will be the the negative of the training length, this value will the starting range, it will pick files not picked by the testing set

    for filename in training_set: #for every file in training set
        this_file = DATA_SOURCE + filename 
        destination = TRAINING + filename 
        copyfile(this_file, destination) #copies the file in training set to the destination, which will be training directory for sunflowers and dandelions

    for filename in testing_set:
        this_file = DATA_SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


#connect newly created folders to variables to use them in the split_data function to put the files in.
DANDELION_SOURCE_DIR = "/ai/flowers/dandelion/"
TRAINING_DANDELION_DIR = "/ai/dandelion-sunflower/training/dandelion/"
TESTING_DANDELION_DIR = "/ai/dandelion-sunflower/testing/dandelion/"
SUNFLOWER_SOURCE_DIR = "/ai/flowers/sunflower/"
TRAINING_SUNFLOWER_DIR = "/ai/dandelion-sunflower/training/sunflower/"
TESTING_SUNFLOWER_DIR = "/ai/dandelion-sunflower/testing/sunflower/"

split_size = .9 #will split data into 90% training data and 10% testing data

#a main dandelion or sunflower directory source will have training and testing directories with unique data(because of the split size) in it respectively
split_data(DANDELION_SOURCE_DIR, TRAINING_DANDELION_DIR, TESTING_DANDELION_DIR, split_size) 
split_data(SUNFLOWER_SOURCE_DIR, TRAINING_SUNFLOWER_DIR, TESTING_SUNFLOWER_DIR, split_size)

# initalizing a convolution neural networks, these are primarily used to analyze visual imagery, structure is the same as a multi layer perceptron neural network(forward and fully connected layers to each other)
model = tf.keras.Sequential(), #sequential tells us we are going to define a sequence of layers, as evidenced below
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu')) # adds 2d convolutional image filtering layer to the neural network model of layers
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2)),
model.add(tf.keras.layers.Flatten()) #converting the data into a 1-dimensional array by multiplying the remaining pixel values to later input it to the main first dense layer.
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 
#Sigmoid is the only activation function compatible with the binary crossentropy loss function. Returns value between 0 and 1 for the data
#Dense implementation is based on a large 512 unit layer followed by a dense that for this project would go to base 2 and would output only one thing


#model uses RMS prop optimizer with the loss function of the neural network being binary corssentrophy with the network's accuracy results presented by the metric 'accuracy'
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy']) 


'''ImageDataGenerator basically lets you maniuplate pixel size of every image in each folder to make all the images standard 
flow_from_directory is a function that lets you put in a path and further augment image data'''

TRAINING_DIR = "/ai/dandelion-sunflower/training/"#connects the testing dataset folder of dandelion and sunflowers to the variable training_datagen
train_datagen = ImageDataGenerator(rescale=1.0/255.)#a process called "normalization" it rescales every image pixel from the range 0 to 255 to the range between 0 and 1. main advantage is that it helps train/test images in the same manner since every image has varying pixel values

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=60, #batches put multiple images into a batch to be processed instead of loading all the data at once, a process called "batch loading"
                                                    class_mode='binary',
                                                    target_size=(150, 150))


VALIDATION_DIR = "/ai/dandelion-sunflower/testing/" #connects the testing dataset folder of dandelion and sunflowers to the variable VALIDATION_DIR
validation_datagen = ImageDataGenerator(rescale=1.0/255.) #same as line 93 but this time for the validation dataset
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=60, 
                                                              class_mode='binary',
                                                              target_size=(150, 150))


''' .fit() used for supervised machine learning(making a function learn through mapping input to the output), 
  .fit trains the model by running the all the data from the path of the training_generator, running it for 13 epochs, 
  with processing 30 batches per epoch, then after the model learns the patterns it gets tested by the data stored 
  in the path of the validation_generator, validation steps is the same as steps_per_epoch, but for the validation data
  steps_per_epoch: number of batches(samples) to be selected for one epoch
'''
history = model.fit(train_generator, epochs=13, steps_per_epoch=30,
                    validation_data=validation_generator, validation_steps=6) 
                                                
model.save("dandyvsunflower.h5") #H5 is a file format to store structured data, Keras saves models in this format so it can save model configuration in a single file.


#results I got when I ran for 13 epochs: 97% accuracy 