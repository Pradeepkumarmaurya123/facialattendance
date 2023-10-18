import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

dhoni_image = face_recognition.load_image_file("photo/dhoni.jpg")
dhoni_encoding = face_recognition.face_encodings(dhoni_image)[0]

ronaldo_image = face_recognition.load_image_file("photo/ronaldo.jpg")
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

virat_image = face_recognition.load_image_file("photo/virat.jpg")
virat_encoding = face_recognition.face_encodings(virat_image)[0]

known_face_encoding = [dhoni_encoding, ronaldo_encoding, virat_encoding]

known_face_names = ["dhoni", "ronaldo", "virat"]

students = known_face_names.copy()

face_location = []
face_encoding = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(current_date+'.csv', 'w+', newline='')
Inwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name is known_face_names:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H-%M-%S")
                Inwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
