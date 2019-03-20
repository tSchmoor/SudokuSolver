from keras import Input
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation
from keras.regularizers import l2

"""
Generates a CNN model with 3 convolutional layers and 2 fully connected layers
    OUTPUT:    model : the model generated
"""
def getModel():
#    Input (28x28x1)
    x = Input(shape=(28,28,1))
    a0 = Dropout(0.2)(x)
    
#    1st Convolutional Layer -> (14x14x16)
    a1 = Conv2D(16, 7, padding='same', activation='relu')(a0)
    a1p = MaxPooling2D()(a1)
    a1pd = Dropout(0.2)(a1p)
    
#    2nd Convolutional Layer -> (7x7x32)
    a2 = Conv2D(32, 7, padding='same', activation='relu')(a1pd)
    a2p = MaxPooling2D()(a2)
    a2pd = Dropout(0.2)(a2p)
    
#    3rd Convolutional Layer -> (1x1x64)
    a3 = Conv2D(64, 7, padding='valid', activation='relu')(a2pd)
    
#    4th Dense Layer -> (1x1x64)
    a4 = Dense(64, activation='relu',kernel_regularizer=l2(0.01))(a3)
    
#    5th Dense Layer -> (1x1x10)
    z5 = Dense(10,kernel_regularizer=l2(0.01))(a4)
    y = Activation('softmax')(z5)
    
#    Instanciate and return the model
    model = Model(inputs=x, outputs=y)
    return model
