#!/usr/bin/env python
# coding: utf-8

import dlib
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

abhadana_image = face_recognition.load_image_file("Amit_Bhadana.jpg")
ab_encoding = face_recognition.face_encodings(abhadana_image)[0]

bbam_image = face_recognition.load_image_file("bhuvan_bam.jpg")
bb_encoding = face_recognition.face_encodings(bbam_image)[0]

achanchlani_image = face_recognition.load_image_file("Ashish_Chanchlani.jpg")
as_encoding = face_recognition.face_encodings(achanchlani_image)[0]

cminati_image = face_recognition.load_image_file("carry_minati.jpeg")
cm_encoding = face_recognition.face_encodings(cminati_image)[0]

known_face_encoding = [ab_encoding, bb_encoding, as_encoding, cm_encoding]
known_faces_names = ["Amit Bhadana", "Bhuvan Bam", "Ashish Chanchlani", "Carry Minati"]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y:%m:%d")

with open('attendance.csv', 'w', newline='') as csvfile:
    lnwriter = csv.writer(csvfile)
    lnwriter.writerow(['Name', 'Time'])

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)

            if name in known_faces_names:
                if name in students:
                    print("Present:", name)
                    students.remove(name)
                    print("Absent:", students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
