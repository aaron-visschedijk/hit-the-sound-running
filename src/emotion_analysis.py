# modified neural network training based on:
# https://github.com/neha01/Realtime-Emotion-Detection

import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

df=pd.read_csv('../data/fer2013.csv')

X_train,train_y,X_test,test_y = [], [], [], []

#remove emotion 0
df = df[df.emotion != 0]

#remove emotion 1
df = df[df.emotion != 1]

#remove emotion 2
df = df[df.emotion != 2]

#remove emotion 2
df = df[df.emotion != 5]

#remove emotion 2
df = df[df.emotion != 6]

train_counts = [0,0,0,0,0,0,0]
test_counts = [0,0,0,0,0,0,0]
train_limit = 5000
test_limit = 1000

#remap emotion 3 and 4 to 0 and 1 respectively
for index, row in df.iterrows():
    emotion = row['emotion']
    val=row['pixels'].split(" ")
    if emotion == 3:
        emotion = 0
    if emotion == 4:
        emotion = 1


    try:
        if 'Training' in row['Usage'] and train_counts[emotion] < train_limit:
            train_counts[emotion] += 1
            X_train.append(np.array(val,'float32'))
            train_y.append(emotion)
        elif 'PublicTest' in row['Usage'] and test_counts[emotion] < test_limit:
            test_counts[emotion] += 1
            X_test.append(np.array(val,'float32'))
            test_y.append(emotion)
    except:
        print(f"error occured at index :{index} and row:{row}")


print(str(train_counts))
print(str(test_counts))

num_features = 64
num_labels = 5
batch_size = 64
epochs = 1
width, height = 48, 48


X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#cannot produce
#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

##designing the cnn
#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

#Compiling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#Training the model
model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)


#Saving the  model to  use it later on
fer_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model.h5")
