#!/usr/bin/env python3

import cv2
import depthai as dai
import numba as nb
import numpy as np

# Optimized with 'numba' as otherwise would be extremely slow (55 seconds per frame!)
@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

   #for i in np.arange(input.size // 5): # around 25ms per frame (with numba)
    for i in nb.prange(input.size // 5): # around  5ms per frame
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

# Properties
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
# Linking
camRgb.raw.link(xoutRgb.input)

#--------------------------------------------------
# Define source and output
camLeft = pipeline.create(dai.node.ColorCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName("left")

# Properties
camLeft.setInterleaved(False)
camLeft.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
camLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)

# Linking
camLeft.raw.link(xoutLeft.input)

# Define source and output
camRight = pipeline.create(dai.node.ColorCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName("right")

# Properties
camRight.setInterleaved(False)
camRight.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
camRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)

# Linking
camRight.raw.link(xoutRight.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    i = 0

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        inLeft = qLeft.get()  # blocking call, will wait until a new data has arrived
        inRight = qRight.get()
        """ print(inRgb.getType())
        print("width: ", inRgb.getWidth())
        print("height: ", inRgb.getHeight())
        print("data size: ", inRgb.getData().size)
        # Retrieve 'bgr' (opencv format) frame
        leftRaw = inLeft.getFrame()
        print(leftRaw.shape)
        print(leftRaw.dtype) """

        # unpack raw10 data
        width = inRgb.getWidth()
        height = inRgb.getHeight()
        shape = (height, width)
        # Input data
        rgbData = inRgb.getData()
        leftData = inLeft.getData()
        rightData = inRight.getData()

        # Output data holder
        unpackedRgb = np.empty(rgbData.size * 4 // 5, dtype=np.uint16)
        unpackedLeft = np.empty(leftData.size * 4 // 5, dtype=np.uint16)
        unpackedRight = np.empty(rightData.size * 4 // 5, dtype=np.uint16)

        # Unpack Process
        unpack_raw10(rgbData, unpackedRgb, True)
        unpack_raw10(leftData, unpackedLeft, True)
        unpack_raw10(rightData, unpackedRight, True)

        # Reshape to 2d image
        unpackedRgb = unpackedRgb.reshape(shape).astype(np.uint16)
        unpackedLeft = unpackedLeft.reshape(shape).astype(np.uint16)
        unpackedRight = unpackedRight.reshape(shape).astype(np.uint16)

        # Convert to bgr
        rgbImg = cv2.cvtColor(unpackedRgb, cv2.COLOR_BayerGB2BGR)
        leftImg = cv2.cvtColor(unpackedLeft, cv2.COLOR_BayerGB2BGR)
        rightImg = cv2.cvtColor(unpackedRight, cv2.COLOR_BayerGB2BGR)
        #  COLOR_BayerBG2BGR
        # rgbImg = inRgb.getCvFrame()
        # leftImg = inLeft.getCvFrame()
        # rightImg = inRight.getCvFrame()

        #resize image to 0.5 scale
        rgbImgVis = cv2.resize(rgbImg, (0, 0), fx=0.5, fy=0.5)
        leftImgVis = cv2.resize(leftImg, (0, 0), fx=0.5, fy=0.5)
        rightImgVis = cv2.resize(rightImg, (0, 0), fx=0.5, fy=0.5)

        # rgbImgVis = rgbImg
        # leftImgVis = leftImg
        # rightImgVis = rightImg

        #show image
        cv2.imshow("rgb", rgbImgVis)
        cv2.imshow("left", leftImgVis)
        cv2.imshow("right", rightImgVis)

        print("dtype: ", rgbImg.dtype)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("capture image")
            i+=1
            cv2.imwrite(f'./calib_data/rgb_{i}.png', rgbImg.astype(np.uint16))
            cv2.imwrite(f'./calib_data/left_{i}.png', leftImg.astype(np.uint16))
            cv2.imwrite(f'./calib_data/right_{i}.png', rightImg.astype(np.uint16))
            # cv2.imwrite(f'rgb.png', rgbImg)
            # cv2.imwrite(f'left.png', leftImg)
            # cv2.imwrite(f'right.png', rightImg)
