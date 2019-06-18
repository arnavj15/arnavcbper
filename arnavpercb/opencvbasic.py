import cv2
import matplotlib.pyplot as plt
img=cv2.imread('dog.png')
newImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.imshow('Dog',img)
#cv2.imshow('Gray dog',newImg)
plt.imshow(img)
plt.show()
#plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
