## Next few lines to uncomment if there is a problem with CUDA
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.optimizers import Adam
from getAugmentedDigitalSet import getAugmentedDigitalSet
from getModel import getModel

# =============================================================================
# Load Samples
# =============================================================================
trainSamples, trainLabels, devSamples, devLabels, Weights = getAugmentedDigitalSet(2000)

# =============================================================================
# Load model
# =============================================================================
model = getModel()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# =============================================================================
# Train the model
# =============================================================================
model.fit(trainSamples, trainLabels,
              batch_size=256,
              epochs=40,
              validation_data=(devSamples, devLabels),
              sample_weight=Weights,
              shuffle=True)

model.save('digitsModelTest.h5')
