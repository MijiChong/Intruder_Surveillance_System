import cv2
from motion_detection import MotionDetector
from object_detection import YOLOv5Detector


def main():
    motion_detector = MotionDetector(min_contour_area=2500, blur_size=(21, 21), threshold_value=10)

    object_detector = YOLOv5Detector()

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Capture the first frame to initialize the motion detector
    ret, frame = cap.read()
    if ret:
        motion_detector.initialize(frame)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect motion and draw bounding boxes
        motion_detected = motion_detector.process_frame(frame)

        if(motion_detected == True):
            frame=object_detector.detect_object(frame)
        
        # Display the frame with motion detection in a window
        cv2.imshow('Intruder Detection System', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    main()
