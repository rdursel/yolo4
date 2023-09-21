import cv2
import numpy as np
import time

np.random.seed(123)

class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath ):
        self.videoPath=videoPath
        self.configPath = configPath
        self.modelPath=modelPath
        self.classesPath=classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()



    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0,'__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))
        print(self.classesList)

    def get_centers(x, y, w, h):
        centerX = (w + x) / 2
        centerY = (h + y) / 2
        return centerX, centerY
    
    def is_circle_inside_rectangle(circle, rectangle):
        circle_x, circle_y, circle_radius = circle
        rect_x1, rect_y1, rect_x2, rect_y2 = rectangle
        
        if (circle_x - circle_radius >= rect_x1) and \
        (circle_x + circle_radius <= rect_x2) and \
        (circle_y - circle_radius >= rect_y1) and \
        (circle_y + circle_radius <= rect_y2):
            return True
        return False



    def onVideo(self):

        def is_circle_inside_rectangle(circle, rectangle):
            circle_x, circle_y, circle_radius = circle
            rect_x1, rect_y1, rect_x2, rect_y2 = rectangle
            
            if (circle_x - circle_radius >= rect_x1) and \
            (circle_x + circle_radius <= rect_x2) and \
            (circle_y - circle_radius >= rect_y1) and \
            (circle_y + circle_radius <= rect_y2):
                return True
            return False
    
        cap=cv2.VideoCapture(self.videoPath)
        if (cap.isOpened()==False):
            print('Error loading file....')
            return
        

        (success, image) = cap.read()

        centroid = (320, 220)

        # Calculate top-left and bottom-right points for a 30x30 square centered at the centroid
        half_side = 50 // 2
        top_left = (centroid[0] - half_side, centroid[1] - half_side)
        bottom_right = (centroid[0] + half_side, centroid[1] + half_side)
        print(top_left, bottom_right)

        while success:
            classLabelIDs, confidences, bboxs = self.net.detect(image,confThreshold=0.5)

            bboxs=list(bboxs)
            #print(bboxs)
            confidences=list(np.array(confidences).reshape(1,-1)[0])
            confidences=list(map(float,confidences))

            bboxsIdx = cv2.dnn.NMSBoxes(bboxs,confidences, score_threshold=0.5, nms_threshold=0.2)
            
            #cv2.circle(image, (320, 220), 7, (255, 255, 255), -1) #where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

            if len(bboxsIdx) !=0:
                for i in range(0, len(bboxsIdx)):
                    bbox = bboxs[np.squeeze(bboxsIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxsIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    displayText = '{}:{:.2f}'.format(classLabel,classConfidence)
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.circle(image, (int(x+w//2), int(y+h//2) ), 7, (255, 0, 0), -1)
                    cv2.putText(image,displayText,(x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor,2)
                    if(is_circle_inside_rectangle((int(x+w//2), int(y+h//2) , 7),(*top_left,*bottom_right))):
                       print("Yeahhh !!!")

            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()
