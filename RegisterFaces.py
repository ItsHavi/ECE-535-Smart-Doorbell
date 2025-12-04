# Havi Patel, Nicholas Nguyen, Angad Singh
# ECE 535
# 11/29/2025
# REGISTERING FACES for tflite. For main smart doorbell code


import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import os


#options

#here we took the file face_rec.tflite, which is a facial recognition model we got from github:
#https://github.com/MCarlomagno/FaceRecognitionAuth/blob/master/assets/mobilefacenet.tflite
rec_model_path="face_rec.tflite"
db_file="known_faces.pkl"




# this function is the brain of all this, because it is what takes an image of a face and turns it into a vector of 128 numbers for the TFlite model
def get_face_embedding(model, image):
    #get input and output details
    in_dets=model.get_input_details()
    out_dets=model.get_output_details()
   
    #resizes the image to match model as the model we have only can take certain sizes of images
    in_shape=in_dets[0]['shape']
    img=cv2.resize(image, (in_shape[1], in_shape[2]))
   
    #normalize the pixels
    img=(np.float32(img)-127.5)/128.0
    img=np.expand_dims(img, axis=0)
    model.set_tensor(in_dets[0]['index'], img) #set tensor here runs our indexing files and then this invokes the model
    model.invoke()
    return model.get_tensor(out_dets[0]['index'])[0]




#looads the AI and the face finder, opens the known faces and turns on the webcam.
def main():
    name=input("Enter name: ")
    model=tflite.Interpreter(model_path=rec_model_path)
    model.allocate_tensors()
    #this is the face finder and this loads what calculates the litte boxes for each face, we can change
    # _default to _alt2 instead of _default because that will change this to
    #disregard anything with not enough human features
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


    #open database if it exists, which lets us add onto a persons data on another run of the registering
    if os.path.exists(db_file):
        with open(db_file, 'rb') as f:
            database=pickle.load(f)
    else:
        database={}


    #make a new list for this person if they are new
    if name not in database:
        database[name]=[]
    cam=cv2.VideoCapture(0)
    count=0
    print(f"\n--- Training {name} ---")
    print("Press 's' to snap a pic 'q' to quit.")
    print("Make sure to capture a lot of angles")
    while True:
        ret, frame=cam.read()
        if not ret: break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray, 1.1, 4)
        big_face=None
        max_area=0
        #draw boxes ourselves, only green because we aren't really detecting anything here
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if w*h>max_area:
                max_area=w*h
                big_face=(x, y, w, h)


        cv2.putText(frame, f"Samples: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Trainer', frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord('s'):
            if big_face is not None:
                (x, y, w, h)=big_face
                #this part crops only the 'frame' so just the face
                crop=frame[y:y+h, x:x+w]
                print(f"Snapping photo {count+1}...")
                embedding=get_face_embedding(model, crop)
                #save to our list
                database[name].append(embedding)
                count+=1
                print("Picture taken")
            else:
                print("No face visible")
        elif key==ord('q'):
            break


    #save everything back to file
    with open(db_file, 'wb') as f:
        pickle.dump(database, f)
    print(f"\nDone. Saved {count} photos for {name}.")
    cam.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()

