import depthai as dai
import numpy as np
import cv2

# read a png image
img = cv2.imread('/workspaces/depthai/calib_data/left_1.png', cv2.IMREAD_UNCHANGED)

# check if the image is of type uint16
print("dtype: ", img.dtype)
cv2.imshow("img", img)
cv2.waitKey(0)