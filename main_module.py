import numpy as np
import face_recognition as fr
import cv2
import os
from datetime import datetime
import pyttsx3 as tts
import winsound
from emailHelpers import Mailer, Email

            # fromaddr = "noreply.cctv.notification@gmail.com"
			# toaddr = "mkmthebest111@gmail.com"
			# mailer = Mailer(fromaddr, "oultqfmeipylghoz")
			# text = email.as_string()
			# mailer.send_mail(text, toaddr)
			# print("Email Sent")

# def attendance(name):
#     with open('sus.csv', 'r+') as f:
#         myDatalist = f.readlines()
#         nameList = []
#         for line in myDatalist:
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:
#             time_now = datetime.now()
#             tstr = time_now.strftime('%H: %M: %S')
#             dstr = time_now.strftime('%d / %m / %Y')
#             f.writelines(f'\n{name}, {tstr}, {dstr}')
#
#             engine.say('A suspect is detected' + name)
#             engine.runAndWait()
#
#             email = Email(fromaddr)
#             email.set_to(toaddr)
#             email.set_subject("Do not reply to this notification")
#             email.set_body("A new suspect has been detected  " + name)
#
#             mailer = Mailer(fromaddr, "oultqfmeipylghoz")
#             text = email.as_string()
#             mailer.send_mail(text, toaddr)

#########################################################

class FaceDetector(object):
    def __init__(self):
        self.fromaddr = "noreply.cctv.notification@gmail.com"
        self.toaddr = "mkmthebest111@gmail.com"
        self.video  = cv2.VideoCapture(0)

        self.path = 'project/Image'
        self.myList = os.listdir(self.path)

    def __del__(self):
        #releasing camera
        self.video.release()

    def imgdatabase(self):
        images = []
        for cu_img in self.myList:
            current_Img = cv2.imread(f'{self.path}/{cu_img}')
            images.append(current_Img)

        return images

    def namedatabase(self):
        personName = []
        for cu_img in self.myList:
            personName.append(os.path.splitext(cu_img)[0])

        return personName

    def faceEncodings(self, images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.encode = fr.face_encodings(img)[0]
                encodeList.append(self.encode)
            return encodeList

    def model( self):
        detector = FaceDetector()
        images = detector.imgdatabase()
        personName = detector.namedatabase()
        frame = self.video.read()
        engine = tts.init()

        knownEncodings = detector.faceEncodings(images)
        duration = 1000
        freq = 440
        # print("All Encodings Completed! ")
        # engine.say('All Encodings Completed ')
        # engine.runAndWait()
        # cap = cv2.VideoCapture(0)

        while True:
            ret, frame = self.video.read()
            faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

            currentFace = fr.face_locations(faces)
            encodeCurrent = fr.face_encodings(faces, currentFace)

            for encodeFace, faceLoc in zip(encodeCurrent, currentFace):
                matches = fr.compare_faces(knownEncodings, encodeFace)
                faceDis = fr.face_distance(knownEncodings, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = personName[matchIndex].upper()
                    print(name)

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    winsound.Beep(freq, duration)
                    # attendance(name)
                    break

            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        # buffer = cv2.imencode('.jpg', frame)
		# frame = buffer.tobytes()
		# yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow("CCTV", frame)
        # if cv2.waitKey(1) == 13:
        #     break
       
        # buffer = cv2.imencode('.jpg', frame)
		# frame = buffer.tobytes()
		# yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
# start = FaceDetector().model()



