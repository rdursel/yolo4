from detector import *
import os

def main():
    #videoPath = "test.mp4"
    #With mpegstreamer (https://apps.microsoft.com/store/detail/mjpeg-streamer/9N7G34WVVPNK?hl=en-us&gl=us)
    videoPath = "http://192.168.3.44:8000/"
    configPath=os.path.join('model_data','ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    modelPath=os.path.join('model_data','frozen_inference_graph.pb')
    classesPath=os.path.join('model_data','coco.names')

    myDetector = Detector(videoPath, configPath, modelPath, classesPath)
    myDetector.onVideo()

if __name__ == '__main__':
    main()
