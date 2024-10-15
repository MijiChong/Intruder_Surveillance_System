import cv2
import torch
from human_recognition import FaceRecognition
import os
from notification import TelegramBot
import time
from playsound import playsound
import threading


class YOLOv5Detector:
    def __init__(self, intruder_dir='intruder', model_path='yolov5s', device='cpu'):
        # Load the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', model_path, device=device)

        # Define the custom classes you want to detect
        self.CLASSES = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
            'bear', 'zebra', 'giraffe', 'butterfly', 'rabbit', 'fish', 'monkey', 'person'
        ]


        self.DANGER = ['bird', 'cat', 'dog', 'cow']

        # Create a set of allowed class names for quick lookup
        self.allowed_classes = set(self.CLASSES)

        #Create FaceRecognition object
        self.face_recognition = FaceRecognition()

        self.intruder_dir = intruder_dir
        
        #Ensure the intruder directory exists
        os.makedirs(intruder_dir, exist_ok=True)

        #Parameter required for notification
        self.chat_id = "5169219801"
        self.bot = TelegramBot(self.chat_id)
        self.current_time = time.time()
        self.sent = False

    def detect_object(self, img):
        # Perform inference
        results = self.model(img)
        identity = ""

        # Draw bounding boxes and labels on the image
        for *box, conf, cls in results.xyxy[0]:
            class_id = int(cls.item())
            class_name = self.model.names[class_id]  # Get class name from model

            # Only process if the detected class is in our custom list
            if conf > 0.5 and class_name in self.CLASSES:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if class_name != "person" else (255, 0, 0)  # Green for animals, blue for humans
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                 # If a person is detected, perform face recognition
                if class_name == "person":
                    person_image = img[y1:y2, x1:x2]
                    # Recognize the face
                    identity = self.face_recognition.recognize_face(person_image)
                    cv2.putText(img, f'Face: {identity}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if identity == "Unknown" or class_name in self.DANGER:
                    intruder_file_path = os.path.join(self.intruder_dir, f"intruder_{x1}_{y1}.jpg")
                    cv2.imwrite(intruder_file_path, img)

                    #Reset notification after selected time
                    if time.time() > self.current_time + 30:
                        self.sent = False
                        self.current_time = time.time()

                    #Send notification
                    if self.sent == False:
                        if identity == "Unknown":
                            message = "Human intruder detected!"
                        else:
                            message = f"Animal intruder detected! It is a/an {class_name}"

                        t1 = threading.Thread(target=self.bot.send_message, kwargs={'message':message})
                        t2 = threading.Thread(target=self.bot.send_photo, kwargs={'photo_path':intruder_file_path})

                        t1.start()
                        t2.start()
                
                        self.sent = True
                    

                    
        return img
