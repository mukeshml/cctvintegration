import mimetypes
from operator import ge
from sre_constants import SUCCESS
from flask import Flask, render_template, Response, Request, request
import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
import pyttsx3 as tts
import winsound
from emailHelpers import Mailer, Email
import csv
import pandas as pd
from os.path import relpath


app = Flask(__name__, template_folder='templates')

camera  = cv2.VideoCapture('project/video1.avi')
# data = 'project/sus'
# datacsv = os.listdir(data)
engine = tts.init()

class FaceDetector(object):
    
	def __init__(self):
		# self.camera  = cv2.VideoCapture('project/video1.avi')
		self.path = 'project/Image'
		# self.data = 'project/sus'
		# self.path = 'Image'
		# self.vpath = 'project/footage'
		self.myList = os.listdir(self.path)
		# self.datacsv = os.listdir(self.data)

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
		

    # def __del__(self):
    #     #releasing camera
    #     self.video.release()

    

    




# def attendance(name):
# 	relpath('C:\Users\Mukesh Kumar\OneDrive\Desktop\new\sus.csv')
# 	data = os.getcwd() + 'C:\Users\Mukesh Kumar\OneDrive\Desktop\new\sus.csv'
# 	f = open(data, 'r+')
# 	d = pd.read_csv(data)
# 	f = pd.DataFrame(data = d)
# 	myDatalist = f.readlines()
# 	nameList = []
# 	nameList.append(f[0])
# 	for line in f:
# 		entry = line.split(',')
# 		nameList.append(f[0])
# 		target = os.path.join(app.static_folder, 'sus.csv')
# 		target = request.form('sus')

	
# 	if name not in f.values:
# 		time_now = datetime.now()
# 		tstr = time_now.strftime('%H: %M: %S')
# 		dstr = time_now.strftime('%d / %m / %Y')
# 		f.writelines(f'\n{name}, {tstr}, {dstr}')
# 		adding_data = pd.DataFrame([name,tstr,dstr])
# 		lst = [{name,tstr,dstr}]
# 		f.append(lst)
# 		f.to_csv('sus.csv',mode ='a',index= False, header= False)

		

        


            
# def attendance(name, sus):
# 	s_file = sus + ".csv"
# 	s_file_open = open(s_file, "r+")
# 	f = s_file_open.readlines
# 	nameList = []

# 	for line in f:
# 		entry = line.split(',')
# 		nameList.append(entry[0])
#     	# entry = line.split(',')
#         # nameList.append(entry[0])
# 		if name not in nameList:
# 			time_now = datetime.now()
# 			tstr = time_now.strftime('%H: %M: %S')
# 			dstr = time_now.strftime('%d / %m / %Y')
# 			f.writelines(f'\n{name}, {tstr}, {dstr}')
# 			engine.say('A suspect is detected' + name)
# 			engine.runAndWait()
			
# 			email = Email(fromaddr)
# 			email.set_to(toaddr)
# 			email.set_subject("Do not reply to this notification")
# 			email.set_body("A new suspect has been detected  " + name)
			
# 			mailer = Mailer(fromaddr, "oultqfmeipylghoz")
# 			text = email.as_string()
# 			mailer.send_mail(text, toaddr)
# 



detector = FaceDetector()
images = detector.imgdatabase()
personName = detector.namedatabase()
knownEncodings = detector.faceEncodings(images)
duration = 1000
freq = 440
print(personName)
print("All Encodings Completed! ")
engine.say('All Encodings Completed ')
engine.runAndWait()

@app.route('/')

def home():
	return render_template('home.html')

database = {'mkm':'123','mukesh':'thebest'}

@app.route('/form_login', methods=['POST', 'GET'])
def login():
	name1= request.form['username']
	key = request.form['password']

	if name1 not in database:
		return render_template('home.html', info='Invalid User')
	
	else:
		if database[name1]!=key:
			return render_template('home.html', info='Invalid Passsword')
		else:
				return render_template('index.html', name = name1)

# camera = cv2.VideoCapture(0)

# def model(main_module):

# 	while True:

#     	frame = main_module.model()
#         yield (b'--frame\r\n'
# 				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen():
	while True:
		success, frame = camera.read()
		if not success:
			break
		else:
			
			# frame = self.video.read()
			# engine = tts.init()

			# knownEncodings = detector.faceEncodings(images)
			# duration = 1000
			# freq = 440
			# # print("All Encodings Completed! ")
			# # engine.say('All Encodings Completed ')
			# # engine.runAndWait()
			# # cap = cv2.VideoCapture(0)

			while True:
				ret, frame = camera.read()
				faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
				faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

				currentFace = fr.face_locations(faces)
				encodeCurrent = fr.face_encodings(faces, currentFace)

				for encodeFace, faceLoc in zip(encodeCurrent, currentFace):
					matches = fr.compare_faces(knownEncodings, encodeFace)
					faceDis = fr.face_distance(knownEncodings, encodeFace)
					matchIndex = np.argmin(faceDis)

					if matches[matchIndex]:
						best_match = personName[matchIndex].upper()
						# print(best_match)
				
						for (x1, y1, x2, y2), name in zip(currentFace, personName):
							y1, x2, y2, x1 = faceLoc
							y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
							cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
							cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
							cv2.putText(frame, best_match, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
							
							
							# engine.say('A suspect is detected' + best_match)
							# engine.runAndWait()
							# email = Email(fromaddr)
							# email.set_to(toaddr)
							# email.set_subject("Do not reply to this notification")
							# email.set_body("A new suspect has been detected  " + best_match)
							# fromaddr = "noreply.cctv.notification@gmail.com"
							# toaddr = "mkmthebest111@gmail.com"
							# mailer = Mailer(fromaddr, "oultqfmeipylghoz")
							# text = email.as_string()
							# mailer.send_mail(text, toaddr)
							# print("Email Sent")
							

							# winsound.Beep(freq, duration)
							#attendance(best_match )
			
				ret, buffer = cv2.imencode('.jpg', frame)
				frame = buffer.tobytes()
				yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')

def video_feed():
	return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')

# def home():
# 	return render_template('index.html')



if __name__ == '__main__':


	app.run(debug= True)
