import os
import cv2
import numpy as np
from profile import Profile
from keras.models import model_from_json
from keras.preprocessing import image
from statistics import mode

# Tolerance is used to see if the observed face has been happy with the music
# that has been playing
tolerance = 1
tolerance_change_rate = 0.05
tolerance_indicator_width = 10

# we use different music profiles for different group sizes
profiles = []
profiles.append(Profile(0, []))
profiles.append(Profile(1, ["ukulele.mp3", "anewbeginning.mp3"]))
profiles.append(Profile(2, ["jazzyfrenchy.mp3", "creativeminds.mp3"]))
profiles.append(Profile(3, ["ukulele.mp3", "anewbeginning.mp3"]))
profiles.append(Profile(4, ["jazzyfrenchy.mp3", "creativeminds.mp3"]))
profile_index = 0
profiles[profile_index].start()

# load model
model = model_from_json(open("../data/models/model.json", "r").read())
# load weights
model.load_weights('../data/models/model.h5')

# facial recognition cascade classifier from opencv to recognize faces
face_haar_cascade = cv2.CascadeClassifier('../data/models/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

# buffer so we dont instantly switch profiles when a face is missed during 1 frame
nr_of_people_buffer = 7
nr_of_people_queue = [0,0,0,0,0,0,0]


while True:
    ret, test_img = cap.read()
    # skip loop if no frame from the camera is received
    if not ret:
        continue

    # convert image to grayscale
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    # update buffer with new value
    if len(nr_of_people_queue) >= nr_of_people_buffer:
        nr_of_people_queue.pop(0)
    nr_of_people_queue.append(len(faces_detected))

    # find the currently appropriate profile
    most_common = mode(nr_of_people_queue)

    # possibly change profile and thus the music
    if profile_index != most_common:
        profiles[profile_index].stop()
        profile_index = most_common
        profiles[profile_index].start()

    # add information about the current porfile and song to the screen
    cv2.putText(test_img, "Profile: " + str(most_common), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_img, "Now playing: " + profiles[profile_index].get_current_song(), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # draw rectangles on faces
    for (x, y, w, h) in faces_detected:
        # resize image for prediction model
        cv2.rectangle(test_img, (x, y),(x + w, y + h),(255, 255, 255), thickness = 1)
        gray = gray_img[y:y + w, x:x + h]
        gray = cv2.resize(gray, (48,48))
        img_pixels = image.img_to_array(gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        # predict using the cropped image
        predictions = model.predict(img_pixels)

        emotions = ('positive', 'negative')

        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # change tolerance rate based on emotion
        if predicted_emotion == 'negative':
            tolerance -= tolerance_change_rate
        elif predicted_emotion == 'positive':
            tolerance += tolerance_change_rate
            tolerance = min(tolerance, 1)

        # display predicted emotion
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # tolerance indicator code
    height, width = test_img.shape[:2]
    indicator = np.zeros((height, tolerance_indicator_width, 3), np.uint8)
    indicator[(int((tolerance/-1) * height)):,:] = (0,0,255)
    numpy_horizontal = np.hstack((test_img, indicator))

    # if tolereance drops below 0, play the next song
    if tolerance <= 0:
        tolerance = 1
        profiles[profile_index].next_song()


    # display image
    cv2.imshow('Facial emotion analysis ', numpy_horizontal)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# close processes
profiles[profile_index].stop()
cap.release()
cv2.destroyAllWindows()
