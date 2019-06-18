#REad video fram by frame

import cv2
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    #gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret== False:
        continue
    #cv2.imshow("Video Frame",frame)
    #cv2.imshow("GRay",gray_frame)
    cv2.imshow("VId ",frame[::,-1::-1,:])# y ,x,z?
# Wait for user to press q for quit

    key_pressed=cv2.waitKey(1) & 0xFF # Bitwise AND to convert 32 bit into 8 bit to get char which is of 8bit
    if key_pressed==ord('q'): #AScii val
        break


cv2.destroyAllWindows()
