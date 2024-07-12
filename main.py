import csv
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)  # Initialize hand detector

# Define MCQ class
class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None

    def updates(self, cursor, bboxs):
        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1<cursor[0]<x2 and y1<cursor[1]<y2:
                self.userAns = x+1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)

# Import questions from CSV file
pathCSV = 'Mcqs.csv'
with open(pathCSV, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# Create MCQ objects
mcqList = [MCQ(q) for q in dataAll]
qNo = 0
qTotal = len(dataAll)

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image
    hands, img = detector.findHands(img, flipType=False)  # Detect hands

    if qNo < qTotal:
        mcq = mcqList[qNo]
        # Display question and choices
        img, bbox = cvzone.putTextRect(img, mcq.question, [100, 100], 2, 2, colorR=(0, 0, 0), offset=50, border=5, colorB=(128, 128, 128))
        img, bbox1 = cvzone.putTextRect(img, mcq.choice1, [100, 250], 2, 2, colorR=(0, 0, 0), offset=50, border=5, colorB=(128, 128, 128))
        img, bbox2 = cvzone.putTextRect(img, mcq.choice2, [400, 250], 2, 2, colorR=(0, 0, 0), offset=50, border=5, colorB=(128, 128, 128))
        img, bbox3 = cvzone.putTextRect(img, mcq.choice3, [100, 400], 2, 2, colorR=(0, 0, 0), offset=50, border=5, colorB=(128, 128, 128))
        img, bbox4 = cvzone.putTextRect(img, mcq.choice4, [400, 400], 2, 2, colorR=(0, 0, 0), offset=50, border=5, colorB=(128, 128, 128))

        if hands:
            lmList = hands[0]['lmList']
            cursor = tuple(lmList[8][0:2])
            p1 = tuple(lmList[8][0:2])
            p2 = tuple(lmList[12][0:2])
            length, info, img = detector.findDistance(p1, p2, img)

            if length < 60:
                mcq.updates(cursor, [bbox1, bbox2, bbox3, bbox4])
                if mcq.userAns is not None:
                    time.sleep(1)
                    qNo += 1
    else:
        score = sum(1 for mcq in mcqList if mcq.answer == mcq.userAns)
        score = round((score / qTotal) * 100, 2)
        img, _ = cvzone.putTextRect(img, "Quiz is completed", [250, 300], 2, 2, colorR=(0, 0, 0), offset=50, border=5)
        img, _ = cvzone.putTextRect(img, f'Your score is: {score}%', [700, 300], 2, 2, colorR=(0, 0, 0), offset=50, border=5)

    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (0, 0, 0), 5)
    img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, colorR=(0, 0, 0), offset=16)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
