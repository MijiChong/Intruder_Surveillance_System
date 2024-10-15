import cv2
from object_detection import YOLOv5Detector

class MotionDetector:
    def __init__(self, min_contour_area=5000, blur_size=(21, 21), threshold_value=20):
        self.min_contour_area = min_contour_area
        self.blur_size = blur_size
        self.threshold_value = threshold_value
        self.frame1 = None
        self.gray1 = None

    def initialize(self, frame):
        """Initialize the first frame for comparison."""
        self.gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray1 = cv2.GaussianBlur(self.gray1, self.blur_size, 0)

    def process_frame(self, frame):
        """Process a frame to detect motion and draw rectangles around moving objects."""
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, self.blur_size, 0)

        # Compute the absolute difference between the first and current frames
        diff = cv2.absdiff(self.gray1, gray2)

        # Apply a binary threshold to the difference image
        _, thresh = cv2.threshold(diff, self.threshold_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                self.gray1 = gray2.copy()
                return True
                

        # Update the first frame to be the current frame
        self.gray1 = gray2.copy()

        return False
