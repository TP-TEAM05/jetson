
import cv2
import numpy as np
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (290, 0), (350, 480), (0,0,0), -1)
cv2.imwrite('test_line.jpg', img)
print('test_line.jpg vytvorený')
