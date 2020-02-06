from processor import *


result = run_detection('images/img.jpg')
cv2.imwrite('result.jpg', result)
cv2.imshow('Result', result)
cv2.waitKey(0)