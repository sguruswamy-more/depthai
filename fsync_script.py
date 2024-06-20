#!/usr/bin/env python3

import cv2
import math
import depthai as dai
import contextlib
import argparse
from datetime import timedelta
import time

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')

args = parser.parse_args()

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.CAM_B,
    'left' : dai.CameraBoardSocket.CAM_A,
    'right': dai.CameraBoardSocket.CAM_C,
}
cam_instance = {
    'rgb'  : 0,
    'left' : 1,
    'right': 2,
}


def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=100))
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName('syncout')
    
    def create(c):

        if c =='rgb':
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
            # cam.setIspScale(1, 2)  # 400P
            # cam.preview.link(xout.input)
            cam.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.OUTPUT)
    
        else:
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
            cam.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)

        cam.isp.link(sync.inputs[c])
        cam.initialControl.setManualExposure(1000, 800)
        cam.setBoardSocket(cam_socket_opts[c])
        cam.setFps(args.fps)



    create('left')
    create('rgb')
    create('right')
    sync.out.link(xout.input)
    return pipeline

ips = [
    '19443010F19D712700'
]

# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0: raise RuntimeError("No devices found!")
    else: print("Found", len(device_infos), "devices")
    queues = []

    for ip in ips:
        device_info = dai.DeviceInfo(ip)
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        cam_list = {'left', 'right', 'rgb'}
        # cam_list = {'left', 'right'}

        print('Starting pipeline for', device.getMxId())
        # Get a customized pipeline based on identified device type
        device.startPipeline(create_pipeline())

        queue = device.getOutputQueue(name="syncout", maxSize=4, blocking=False)

        start_time = time.time()
        frame_count = 0
        while True:
            new_msg = queue.get()
            frame_count += 1
            prev_ts = None
            for name, msg in new_msg:
                print(f"Got {name} frame, seq: {msg.getSequenceNum()} TS: {msg.getTimestamp()}")
                if prev_ts is not None:
                    print(f"Diff microsec: {(msg.getTimestamp() - prev_ts).microseconds}")
                if prev_ts is None:
                    prev_ts = msg.getTimestamp()
                imgFrame: dai.ImgFrame = msg
                frame = imgFrame.getCvFrame()
                # Write text
                frame = cv2.resize(frame, (640, 400))
                frame = cv2.putText(frame, f"TS {imgFrame.getTimestamp(dai.CameraExposureOffset.MIDDLE)}", (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color=(127, 255, 0))
                frame = cv2.putText(frame, f"Seq {imgFrame.getSequenceNum()}", (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color=(127, 255, 0))
                cv2.imshow(name, frame)

            if frame_count % 20 == 0:
                frame_count = 0
                start_time = time.time()
            end_time = time.time()
            seconds = end_time - start_time
            fps  = frame_count / seconds
            print(f"FPS: {fps}")
            if cv2.waitKey(1) == ord('q'):
                break
