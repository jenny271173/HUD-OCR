# Import necessary libraries and packages
import cv2
import pytesseract
import re

# Pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# File path for video
# Video from https://www.youtube.com/watch?v=WkZGL7RQBVw&ab_channel=AviationWeek
video = "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened1.mp4"  

# Preprocess the frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blur)
    thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing


# Run OCR on the video frames for the specified ROI
# inspired by https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# Note to define ROI's: 
# x = move left/right, y = move up/down
# w = width box, h = height box
def run_OCR(filename, ROI, num_frames):
    cap = cv2.VideoCapture(filename)
    numbers_array = []
    frame_count = 0
    exit = False 
    
    while cap.isOpened() and not exit:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            if frame_count % num_frames == 0:
                for ROI_name, (x, y, w, h) in ROI.items():
                    roi = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    preprocessed = preprocess_frame(roi)
                    text = pytesseract.image_to_string(preprocessed, config='')
                    numbers = re.findall(r'\d+', text)
                    
                    if text:
                        print(text)
                
                    # Append the numbers to the array
                    if numbers:
                        numbers_array.append(numbers)
                    
                    # Display the numbers on the frame
                    cv2.putText(frame, ','.join(numbers), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                exit = True 
                
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return numbers_array

if __name__ == "__main__":
    ROI = {
        'ROI 1': (525, 540, 95, 35)
    }
    result = run_OCR(video, ROI, num_frames=10)
    
    ROI_2 = {
        'ROI 2': (630, 540, 95, 35)
    }
    result_2 = run_OCR(video, ROI_2, num_frames=10)
    
print(result)
print(result_2)
