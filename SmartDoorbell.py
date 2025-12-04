# Havi Patel, Nicholas Nguyen, Angad Singh
# ECE 535
# 11/30/2025
# MAIN RUNNING FILE FOR SMART DOORBELL




import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle


# options
rec_model_path="face_rec.tflite"
db_file="known_faces.pkl"
threshold=0.9 #lower is stricter, higher is looser,
    # threshold is what determines the level that it can look at you and "see" you before it thinks you're not a recognized person


#simple helped that we got using pickle, basically opens the known_faces.pkl fiile that we made from
#register_faces.py so it doesn't just crash the program if it's not there
def load_db():
    try:
        with open(db_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("No database found, run register_face.py")
        return {}


#same as trainer scripts
def get_face_embedding(model, image):
    in_dets=model.get_input_details()
    out_dets=model.get_output_details()
   
    in_shape=in_dets[0]['shape']
    img=cv2.resize(image, (in_shape[1], in_shape[2]))
   
    #normalize the images
    img=(np.float32(img)-127.5)/128.0
    img=np.expand_dims(img, axis=0)
    model.set_tensor(in_dets[0]['index'], img)
    model.invoke()
    return model.get_tensor(out_dets[0]['index'])[0]


#main loop
# NOTE: This will basically be the same thing as the register as the register calls the same method to try and recognize people
#first finds a face, then crops it. Converts it to numbers, and then matches it against all of our training photos in our training, also calculates distance and who is who visibly to the user
def main():
    database=load_db()
    model=tflite.Interpreter(model_path=rec_model_path)
    model.allocate_tensors()
   
    #so we took the simplest face finder from cv2, we can change this to _alt2 instead of _default because that will change this to
    #disregard anything with not enough human features
    #\/
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    print("Smart Doorbell is running /n Press 'q' to quit while in program")
    while True:
        ret, frame=cam.read()
        if not ret: break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            #grab the face and crop it
            crop=frame[y:y+h, x:x+w]
            curr_embed=get_face_embedding(model, crop)
            #if we don't have them saved, they're unrecognized
            match_name="Unknown"
            MinDist=float('inf')


            #check every person in the book
            for name, photo_list in database.items():
                #check every photo we have of them
                for saved_photo in photo_list:
                    #math to see how similar they are
                    dist=np.linalg.norm(curr_embed-saved_photo)
                   
                    #keep the best match found so far
                    if dist<MinDist:
                        MinDist=dist
                        if dist<threshold:
                            match_name=name


            #draw the box
            if match_name!="Unknown": color=(0, 255, 0) #green
            else: color=(0, 0, 255) #red
           
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{match_name} ({MinDist:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Doorbell', frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()



