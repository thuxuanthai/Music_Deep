from __future__ import division, print_function
# import sys
import os
import cv2
# import re
import numpy as np
import tensorflow as tf
from gevent.pywsgi import WSGIServer
from keras.models import load_model
from keras.utils.image_utils import img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index1.html")


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    # i=0
    #
    # GR_dict={0:(0,255,0),1:(0,0,255)}
    # model = tf.keras.models.load_model('model_train.hdf5')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # output=[]
    # cap = cv2.VideoCapture(0)
    # while (i<=30):
    #     ret, img = cap.read()
    #     faces = face_cascade.detectMultiScale(img,1.05,5)
    #
    #     for x,y,w,h in faces:
    #
    #         face_img = img[y:y+h,x:x+w]
    #
    #         resized = cv2.resize(face_img,(224,224))
    #         reshaped=resized.reshape(1, 224,224,3)/255
    #         predictions = model.predict(reshaped)
    #
    #         max_index = np.argmax(predictions[0])
    #
    #         emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad",
    #                         6: "Surprised"}
    #         # emotions = ('angry', 'disgusted', 'fearful', 'happy', 'sad', 'neutral', 'surprised')
    #         predicted_emotion = emotion_dict[max_index]
    #         output.append(predicted_emotion)
    #
    #
    #
    #         cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
    #         cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
    #         cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    #     i = i+1
    #
    #     cv2.imshow('LIVE', img)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         break
    # print(output)
    # cap.release()
    # cv2.destroyAllWindows()
    # final_output1 = st.mode(output)

    model = load_model(r'E:\TG_MT\view_music\emotion-based-music-player-master\model_train.hdf5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

    # start the webcam feed
    # cap = cv2.VideoCapture(0)
    # image = cv2.imread('.\\data\\train\\angry\\tucgian.jpg')
    image = cv2.imread(r'E:\TG_MT\view_music\emotion-based-music-player-master\tucgian.jpg')
    # Find haar cascade to draw bounding box around face
    facecasc = cv2.CascadeClassifier(r'E:\TG_MT\view_music\emotion-based-music-player-master\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        predicted_emotion = emotion_dict[maxindex]
        cv2.putText(image, predicted_emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        # show the output frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    print(predicted_emotion)
    # final_output1 = st.mode(predicted_emotion)
    return render_template("buttons.html", final_output=predicted_emotion)


@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")


@app.route('/movies/surprise', methods=['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")


@app.route('/movies/angry', methods=['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")


@app.route('/movies/sad', methods=['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")


@app.route('/movies/disgust', methods=['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")


@app.route('/movies/happy', methods=['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")


@app.route('/movies/fear', methods=['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")


@app.route('/movies/neutral', methods=['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")


@app.route('/songs/surprise', methods=['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")


@app.route('/songs/angry', methods=['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")


@app.route('/songs/sad', methods=['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")


@app.route('/songs/disgust', methods=['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")


@app.route('/songs/happy', methods=['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")


@app.route('/songs/fear', methods=['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")


@app.route('/songs/neutral', methods=['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")


@app.route('/templates/join_page', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='localhost', port=5000, debug=True, use_reloader=True)
