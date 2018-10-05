
import numpy as np
import pdb
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
import pandas as pd 
import os
# from keras.utils import plot_model
import matplotlib.pyplot as plt
import time

 
exp_dir = './inception_test/'
x_test = np.load('./test_img_list.npy')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=360)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './train/',
        target_size=(224, 224),
        batch_size=50,
        class_mode='categorical')
        
validation_generator = test_datagen.flow_from_directory(
        './val/',
        target_size=(224, 224),
        batch_size=59,
        class_mode='categorical')

# model1 = MobileNet(input_shape= (224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights=None)
model1 = InceptionV3(input_shape= (224, 224, 3), weights='imagenet', include_top = False)

for layer in model1.layers:
    layer.trainable = False
    
model2 = Sequential()
model2.add(model1)
# model2.add(Dropout(0.5))
model2.add(Flatten())
model2.add(Dense(1000, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(512, activation='relu'))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(12, activation = 'softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model2.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])
# plot_model(model2, to_file='model-inceptionv3.png')

filepath=exp_dir + "weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',period = 10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=0.0001)
tensorboard = TensorBoard(log_dir=exp_dir+'logs/{}'.format(time.time()))
callbacks_list = [checkpoint,tensorboard,reduce_lr]

model2.fit_generator(train_generator,epochs=50,shuffle= True, validation_data=validation_generator, validation_steps=12, callbacks=callbacks_list) #sample/batch

######################################Test Prediction#######################################################
preds = model2.predict(x_test, batch_size=1, verbose=0)
np.save('./preds_all2.npy',preds) 
l1=[]
pdb.set_trace()
classes = os.listdir('./train')
preds  =np.load('./preds_all2.npy')
for i in preds:
    l1.append(np.where(i==i.max()))
class_preds = [classes[i] for i in l1]
fnames = np.load('./test_img_name.npy')
fname = [i for i in fnames]
df = pd.DataFrame(data={"file": fname, "species": class_preds})
df.to_csv("preds2.csv",sep=',',index=False) 
############################################################################################################

'''
Supplementary code for creating val data

import os
import shutil
import numpy as np
######################################Cretae Val Folders####################################
# classes = os.listdir('./train/')
# for cl in classes:
    # os.makedirs('./val/'+ cl)
############################################################################################


##########################################Create Val Data Method 1################################
classes = os.listdir('./train/')
file_count  = 0
for cl in classes:
    files_per_class= os.listdir('./train/'+cl)
    for n,img in enumerate(files_per_class):
        if n> int (0.85*len(files_per_class)):
            shutil.move('./train/'+cl +'/'+ img,'./val/'+cl +'/'+ img)
            file_count+=1
            
print ("Files Moved  = %d" % (file_count))


##########################################Create Val Data Method 2################################
classes = os.listdir('./train/')
file_count  = 0
for cl in classes:
    files_per_class= os.listdir('./train/'+cl)
    file_indices = np.random.choice(len(files_per_class),int (0.85*len(files_per_class)))
    files_to_transfer = [files_per_class[i] for i in file_indices]
    for img in files_to_transfer:
        shutil.move('./train/'+cl +'/'+ img,'./val/'+cl +'/'+ img)
        file_count+=1
print ("Files Moved  = %d" % (file_count))
''' 
