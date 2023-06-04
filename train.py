import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = '/Users/noura/Documents/Face Recognition with ML App/src/data/train'
val_dir = '/Users/noura/Documents/Face Recognition with ML App/src/data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')

cv2.ocl.setUseOpenCL(False)
'''

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    bounding_box = cv2.CascadeClassifier(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2gray_frame)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



import numpy as np 
import cv2 
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator 

# to get our data 
train_dir = '/Users/noura/Documents/Face Recognition with ML App/src/data/train'
val_dir = '/Users/noura/Documents/Face Recognition with ML App/src/data/test'
train_datagen = ImageDataGenerator(rescale=1./255) # generate a data set from the image files  in our dir
# our data is images, which has pixels and consist of an array between 1 and 255  
val_datagen = ImageDataGenerator(rescale=1./255)
# create our generator from the data imported. 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48), # target size 48x48 pixels
    batch_size=64, # refers number of training sample utilized in one iteration
    color_mode='grayscale', # black and white images
    class_mode='categorical') # since we have categories of facial expressions
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

# Creating model to prepare for training
emotion_model = Sequential() # initializing a sequential model (keras gives two options: sequential and functional)
# sequential allows us to create models layer by layer but it is limited as it doesn't let you create models that share layers
# or have multiple outputs/inputs 
# functional has more flexibility, can connect a layer to ANY other layer but it is more complex

# these are the layers 
# more than two is deep neural network
# conv neural network is a special type where you use for image identificaion and classification. 
# "a convolution is a simple application of a filter that results in an activation" 
# when input meets certain thereshold, there is an activation
# if there is a certain type of input that goes in, and that input repeats itself in a certain way, then we can 
# see a map forming (feature map) that indicates the location and strengths of the detect input
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))) # this layer creates conv kernel that produces a tensor of outputs 
# provide input shape, 32, height of conv network, activation('relu'== rectified, the function will output the input directly if positive and 0 otherwise, very common )
# input_shape represents our images (48x48 and grayscaled)
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2))) # downsampling layer to include less data
emotion_model.add(Dropout(0.25)) # randomely sets input values to 0 during training time to prevent over-fitting
# over fitting means when a model provides accurate predictions for training data but not for the new/unseen data 
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten()) # flatten layer which we need to flatten o
emotion_model.add(Dense(1024, activation='relu')) # hidden layer which is going to contain 1024 neurons/units
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) # hidden layer that contains exact number of possibilities we can have, meaning emotions.  
# have to convert allll those 1024 units into the 7 
# compile our model:
# loss is degree of error -> models try to minimize loss but dont try to maximize accuracy -> so we use the 'Adam' optimizer which is very standard
# categorical_crossentropy, the type metric of how we are specfying out losses
# since we need to specify a metric we want to track, we specificy 'accuracy' as our metric
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

# train our model 
emotion_model_info = emotion_model.fit_generator(
                     train_generator,
                     steps_per_epoch=28709// 64, # since our batch size is 64
                     epochs=2, # number of iterations we want
                     validation_data=validation_generator,
                     validation_steps=7178 // 64)

emotion_model.save_weights('model.h5') # save our weights here
'''