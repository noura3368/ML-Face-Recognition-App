# deep neural network has a certain level or complexities
# more than 2 hidden layers
# data/information that goes into our network, first layer is called entry layer
#hidden layer has certain amount of neurons that the first layer is connected to second by their neurons
# each layer of a neural connected is connected  each other
# then we have output that our last hidden layer is connected to 
# hidden layer figures out some type of mapping / trains the data   
# conv means a mathematical operations on two functions that produces a third function
# some shape that goes in, two values have been operated on and a third value is produced
'''
import tkinter as tk 
from tkinter import *
import cv2  
from PIL import Image, ImageTk # using to import images
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading # for multi-threading
from train import emotion_model

emotion_model.load_weights('model.h5') # loading our emotion_model into this file
cv2.ocl.setUseOpenCL(False) 

emotion_dict = {0: "Angry", 1:"Fearful", 2: "Happy", 3: "Neutral", 4:"Sad", 5:"Surprised", 6: "Disgusted"}

current_path = os.path.dirname(os.path.abspath(__file__)) # current absolute path 

emoji_dist={0: current_path + "/data/emojis/angry_woman.png", 1: current_path + "/data/emojis/fearful_woman.png", 2: current_path + "/data/emojis/happy_woman.png", 3: current_path + "/data/emojis/default_woman.png", 4: current_path + "/data/emojis/sad_woman.png", 5: current_path + "/data/emojis/surprised_woman.png", 6: current_path + "/data/emojis/disguted_woman.png"}

global last_frame
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global camera
show_text=[0]
global frame_number


def show_subject():
    camera = cv2.VideoCapture("C:/Users/noura/Downloads/pexels-artem-podrez-5137640-3840x2160-30fps.mp4") # use webcam live
    print(camera)
    if not camera.isOpened():
        print("Can't open the camera")
    global frame_number
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT)) # length of frame, maximum number of frame to know when to exit
    frame_number += 1 
    if frame_number >= length:
        exit()
    camera.set(1, frame_number) # set the next frame we should read (camera), to the value we just incremented to
    flag, frame = camera.read() # read frame by frame, returns array frame and a flag (specifies if we have actually read something or not)
    frame = cv2.resize(frame, (600, 500))
    bounding_box = cv2.CascadeClassifier('C:/Users/noura/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') # represents a box around person's face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn our image to gray, most classifications can be done using grayscale images + faster processing time as less memory is used
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for(x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # refine our image a bit 
        roi_gray_frame = gray_frame[y:y + h, x:x + w] # crop our image
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img) # predict the emotion based on image
        maxindex = int(np.argmax(prediction)) # from 0 - 6
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex
    if flag is None: # mean that we haven't created a frame
        print("ERROR")
    # update main window
    elif flag: 
        global last_frame
        last_frame = frame.copy() # get frame
        pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic) # create an image (arrays that create values from 0 to 255) to create an img, use Image from PIL
        imgtk = ImageTk.PhotoImage(image=img) # imagetk that can be used with tklabel
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        root.update() # update the main thread where tk GUI is running 
        lmain.after(10, show_subject) 
    if cv2.waitKey(1) & 0XFF == ord('q'): # press q or type any value, we will exit
        exit()


def show_avatar():
    second_frame = cv2.imread(emoji_dist[show_text[0]]) # read the emoji
    pic2 = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB) # 
    img2 = Image.fromarray(second_frame)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(10, show_avatar)
'''
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading # for multi-threading
#from train import emotion_model

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

global emoji_dist, last_frame, cap
emoji_dist = {}
emotion_model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
current_path = os.path.dirname(os.path.abspath(__file__)) # current absolute path 

def change_avatar(gender):
    global emoji_dist
    show = False
    if emoji_dist == {}:
        show = True
    emoji_dist={0: current_path + "/data/emojis/angry_" + gender + ".png", 1: current_path + "/data/emojis/disgusted_" + gender + ".png", 2: current_path + "/data/emojis/fearful_" + gender + ".png", 3: current_path + "/data/emojis/happy_" + gender + ".png", 4: current_path + "/data/emojis/default_" + gender + ".png", 5: current_path + "/data/emojis/sad_" + gender + ".png", 6: current_path + "/data/emojis/surprised_" + gender + ".png"}                                 
    if show:
        show_avatar()
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)   
show_text=[0]

def show_live_video(): 
    global cap                               
    if not cap.isOpened():                             
        print("cant open the camera1")
    flag, frame = cap.read()
    frame = cv2.resize(frame,(600,500))
    bounding_box = cv2.CascadeClassifier('C:/Users/noura/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction)) 
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # put emotion text on screen
        show_text[0]=maxindex
        
    if flag is None:
        raise RuntimeError("Can't read camera footage properly!")
    global last_frame
    last_frame = frame.copy()
    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_live_video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def show_avatar():
    global emoji_dist
    second_frame=cv2.imread(emoji_dist[show_text[0]])
    img2=Image.fromarray(second_frame)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_avatar)

if __name__ == '__main__':
    frame_number = 0 # increment frame by frame 
    # front-end portion
    root=tk.Tk() 
    root.title("Emojify Yourself!")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'

    heading2=Label(root,text="Emojify yourself!",pady=20, font=('arial',45,'bold'),bg='black', fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)
    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    
    root.title("Emojify Yourself!")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    girl_button = Button(root, text="Girl", bg='pink', command=lambda: change_avatar('woman'), font=('arial', 25, 'bold')).place(x = 700, y = 100)
    boy_button = Button(root, text="Boy", bg='blue', command=lambda: change_avatar('man'),font=('arial', 25, 'bold')).place(x=600, y=100)
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    
    # both run indefinitely 
    #threading.Thread(target=show_subject).start() # use threading because the main loop is blocking 
    #threading.Thread(target=show_avatar).start()
    threading.Thread(target=show_live_video).start()
    threading.Thread(target=show_avatar).start()

    root.mainloop()
