import cv2
import dlib
import keyboard
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:/Users/hp/Desktop/game-opencv/shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        landmark=predictor(gray, face)
        lipUp=landmark.parts()[62]
        lipDown=landmark.parts()[66]
        keyboard.press("up")
        if(lipDown.y-lipUp.y>4):
            keyboard.press("left")
        
        else:
            keyboard.press("right")           
    if ret:
        cv2.imshow("User",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

