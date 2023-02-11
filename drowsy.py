from threading import Thread
import time
import numpy as np
import cv2
import time
import os
# from imutils.video import VideoStream
# from imutils import face_utils
# import imutils
# import dlib
import cv2
# import argparse


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="Enter the path to the shape predictor.")
# ap.add_argument("-a", "--alarm", type=str, default="",
# 	help="Enter the path to the alarm file.")
# ap.add_argument("-w", "--webcam", type=int, default=0,
# 	help="Webcam (can change it to an external webcam)")
# args = vars(ap.parse_args())

'''
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	overlay = image.copy()
	output = image.copy()
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
	for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
		(j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		if name == "jaw":
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
			# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output
'''

# threshold = 0.3
# threshold_frames = 45


# count = 0
# Al_on = False


# print("Finding facial predictor!")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])


# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# (aStart, bEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
# (bStart, aEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
# (xStart, yEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# print("Starting Video Stream!")
# vs = VideoStream(src=args["webcam"]).start()
# time.sleep(1.0)


# while True:

# 	frame = vs.read()
# 	frame = imutils.resize(frame, width=450)
# 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	size = frame.shape
# 	rects = detector(gray, 0)

# 	for rect in rects:

# 		shape = predictor(gray, rect)
# 		shape = face_utils.shape_to_np(shape)
# 		nose = shape[aStart:bEnd]
# 		jaw = shape[bStart:aEnd]
# 		mouth = shape[xStart:yEnd]
# 		jawHull = cv2.convexHull(jaw)
# 		mouthHull = cv2.convexHull(mouth)
# 		noseHull = cv2.convexHull(nose)
# 		leftEye = shape[lStart:lEnd]
# 		rightEye = shape[rStart:rEnd]
# 		leftEAR = functions.eye_aspect_ratio(leftEye)
# 		rightEAR = functions.eye_aspect_ratio(rightEye)


# 		ear = (leftEAR + rightEAR) / 2.0


# 		leftEyeHull = cv2.convexHull(leftEye)
# 		rightEyeHull = cv2.convexHull(rightEye)
# 		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
# 		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
# 		cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
# 		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
# 		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

# 		#visualize_facial_landmarks(frame, face_utils.FACIAL_LANDMARKS_IDXS)

# 		if ear < threshold:
# 			count += 1


# 			if count >= threshold_frames:

# 				if not Al_on:
# 					Al_on = True


# 					if args["alarm"] != "":
# 						t = Thread(target=functions.sound_alarm,
# 							args=(args["alarm"],))
# 						t.deamon = True
# 						t.start()


# 				cv2.putText(frame, "WAKE UP!", (10, 30),
# 					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# 		else:
# 			count = 0
# 			Al_on = False


# 		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF


# 	if key == ord("e"):
# 		break

# cv2.destroyAllWindows()
# vs.stop()



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
a=0
b=0
c=time.time()
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No face detected")

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if (time.time()-c) >= 15:
            if a/(a+b)>=0.2:
                os.system('buzz.mp3')
                print("****ALERT*****",a,b,a/(a+b))
                
            else:
                print("safe",a/(a+b))
            c=time.time()
            a=0
            b=0
        
        if len(eyes) == 0:
            a=a+1
            print('no eyes!!!')
            cv2.putText(img, "WAKE UP!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            b=b+1
            print('eyes!!!')
            
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        print("noeyes: ",a)
        print("total: ",(a+b))
        break

cap.release()
cv2.destroyAllWindows()