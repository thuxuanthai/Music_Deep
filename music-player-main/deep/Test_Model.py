import numpy as np
import argparse
import cv2
from keras.models import load_model
import statistics as st

GR_dict={0:(0,255,0),1:(0,0,255)}
model = load_model('model.hdf5')
i = 0
output = []
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
while (True):
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        predicted_emotion = emotion_dict[maxindex]
        output.append(predicted_emotion)
        cv2.rectangle(frame,(x,y),(x+w,y+h),GR_dict[1],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),GR_dict[1],-1)
        cv2.putText(frame, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    i = i + 1

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
print(output)
cap.release()

# return render_template("buttons.html", final_output=final_output1)
