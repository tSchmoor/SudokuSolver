import os
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from LoadImagesFromFolder import LoadImagesFromFolder
"""
Generate an augmented train set and an augmented dev set
    INPUTS :    numberOfBatchGenerated : the number of batch to generate, the training set will be composed of 32*numberOfBatchGenerated samples
                dev_ratio : the ratio of samples to save for the dev set, the dev set will be composed of 32*numberOfBatchGenerated samples*dev_ratio samples

    OUTPUTS :   x_train_generated : the training samples
                y_train_generated : the one-hot encoded lables of the training samples
                x_dev_generated : the dev samples
                y_dev_generated : the one-hot encoded labels of the dev samples
                weights : the sample weights to apply on the training samples
"""
def getAugmentedDigitalSet(numberOfBatchGenerated=1000, dev_ratio=0.1):
#    Load the digital database
    DigitalDatabasePath = os.getcwd()+r'\ImageTrainDatabase'
    DigitalImages, DigitalNames = LoadImagesFromFolder(DigitalDatabasePath)
    m = len(DigitalImages)
    if m==0:
        return
    
    (h,w,_) = np.shape(DigitalImages[0])
    m_extended = 2**8
    if dev_ratio>0:
        m_dev = int(m_extended*dev_ratio)
        m_train = m_extended-m_dev
    else: m_train = m_extended
    
    
    x_digital = np.uint8(20*np.random.random((m_extended,h,w)))
    y_digital = np.zeros(m_extended,np.uint8)
    for index in range(m):
        image, name = DigitalImages[index], DigitalNames[index]
        Sample = 255-cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        for digit in range(0,10):
            if str(digit) in name:
                Label = digit
                x_digital[index] = np.uint8(Sample)
                y_digital[index] = np.uint8(Label)
    x_digital = x_digital[:,:,:,np.newaxis]
    
    
#    Shuffle the database and separate the train and dev sets
    order = np.arange(m_extended)
    np.random.shuffle(order)
    x_digital = x_digital[order]
    y_digital = y_digital[order]
    y_digital_encoded = to_categorical(y_digital, num_classes=10, dtype='float32')
    
    x_train, x_dev = x_digital[:m_train], x_digital[m_train:]
    y_train, y_dev = y_digital_encoded[:m_train], y_digital_encoded[m_train:]
    
#    Instanciate the Data Generator for the train set
    BatchSize = 32
    dataGenerator = ImageDataGenerator(rotation_range=15,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       zoom_range=0.1)
    trainDataIterator = dataGenerator.flow(x_train,y_train,batch_size=BatchSize)

#    Generate articifial samples for the train set
    x_train_generated = np.uint8(20*np.random.random((numberOfBatchGenerated*BatchSize,h,w,1)))
    y_train_generated =to_categorical(np.zeros(numberOfBatchGenerated*BatchSize), num_classes=10, dtype='float32')
    for batch in range(numberOfBatchGenerated):
        generatedBatch = trainDataIterator.next()
        for index in range(generatedBatch[0].shape[0]):
            x_train_generated[index+batch*BatchSize] = generatedBatch[0][index]
            y_train_generated[index+batch*BatchSize] = generatedBatch[1][index]
    
    if dev_ratio>0:
#    Instanciate the Data Generator for the dev set
        devDataIterator = dataGenerator.flow(x_dev,y_dev,batch_size=BatchSize)

#    Generate articifial samples for the dev set
        x_dev_generated = np.uint8(20*np.random.random((int(numberOfBatchGenerated*BatchSize*dev_ratio),h,w,1)))
        y_dev_generated =to_categorical(np.zeros(int(numberOfBatchGenerated*BatchSize*dev_ratio)), num_classes=10, dtype='float32')
        for batch in range(int(numberOfBatchGenerated*dev_ratio)):
            generatedBatch = devDataIterator.next()
            for index in range(generatedBatch[0].shape[0]):
                x_dev_generated[index+batch*BatchSize] = generatedBatch[0][index]
                y_dev_generated[index+batch*BatchSize] = generatedBatch[1][index]
    else:
        x_dev_generated = x_dev
        y_dev_generated = y_dev
        
#    Add new dimentions to the labels
    y_train_generated = y_train_generated[:,np.newaxis,np.newaxis,:]
    y_dev_generated = y_dev_generated[:,np.newaxis,np.newaxis,:]
    
#    Set sample weights for the training
    weights = np.ones(len(x_train_generated))
    weights[np.argmax(y_train_generated)==0] = 1
    weights[np.argmax(y_train_generated)==1] = 2
    weights[np.argmax(y_train_generated)==3] = 2
    weights[np.argmax(y_train_generated)==6] = 2
    weights[np.argmax(y_train_generated)==8] = 2
    weights[np.argmax(y_train_generated)==9] = 2

    
    return x_train_generated, y_train_generated, x_dev_generated, y_dev_generated, weights