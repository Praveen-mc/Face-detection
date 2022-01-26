import cv2
import os
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
capture = cv2.VideoCapture(0) #capturing video

while (True):
    # Capture frame-by-frame
    ret, frames = capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(image ,start_point, end_point, color, thickness)
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(frames, 'FACE',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,2),1,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
