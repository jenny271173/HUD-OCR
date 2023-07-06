# Import streamlit
import streamlit as st

# Local URL: http://localhost:8501
# Network URL: http://192.168.0.17:8501

st.title("Head Up Display (HUD) Optical Character Recognition (OCR) Model")
st.markdown('Jennifer Doan')
st.markdown('**Bottom line up front**: Scroll down to the bottom of this page to implement the HUD OCR now!')

st.header('Introduction')
st.markdown('This is a web-based application to implement an OCR model on HUD videos.')
st.markdown('The main objective for this project is to recognize text and/or numbers in streaming aircraft HUD recordings.')
st.header('Background Information')
st.markdown('When analyzing fighter aircraft, collecting data from head-up display videos can be very time consuming. From personal experience, the data collection process occurs by rewatching these videos during a debrief session and then taking notes at particular time stamps. This process proves to be insufficient when the videos are lengthy enough that it becomes difficult to review in detail, or there are just too many to analyze in the bigger picture. This can be a familiar situation for those who have explored flight tests, such as target location error (TLE) or sensor testing, in the past. In addition, human error comes to play when pilots are tasked with performing data collection during their flights. According to Wilson and others (2019), fatigue was shown to be a contributing factor in at least 20% of accidents that were related to transportation. This is already an indication that there are implications within aviation as well. Drowsiness and fatigue related symptoms have become such a topic of interest that a study was done to collect pilot physiological information, such as photoplethysmogram and electrocardiograms, in flight to confirm the negative effects (Wilson, 2019). Thus, the majority of the data collection work load is on analysts on the ground, which also deal with issues of complacency with redundancy. Therefore, it is necessary to find a new method of data collection that is much more efficient and accurate. In this project, an attempt will be made at performing optical character recognition to collect data from declassified head-up display videos from fighter aircraft.')
st.markdown('In other instances, there have been artificial intelligence (AI) related errors in the military theater that have led to tragedies and loss of life. This can, in part, be explained by inaccurate data fed into AI machines that were intended to provide help. In a particular incident on March 22, 2003, the Royal Air Force lost two soldiers because of errors that were attributed to misclassification and the improper definition of rules and autonomous behaviors of a potential enemy weapon (Atherton, 2022). Utilizing other programs, such as OCR, to assist in the reduction of human error in input data can significantly reduce AI related errors, and perform work more efficiently overall. There is very much evidence backing the need for the ability to perform OCR on aircraft data to reduce human error and overall workload.')

st.header('HUD Video Example')
st.markdown('This is the HUD video taken directly from YouTube and downloaded as an .mp4 file. It can be found here: https://www.youtube.com/watch?v=aSzXqlnT7nQ&ab_channel=NASAArmstrongFlightResearchCenter')

def main():
    F16_HUD = "https://github.com/jenny271173/HUD-OCR/raw/main/F16_HUD.mp4"
    
    # Display the video
    st.video(F16_HUD)

if __name__ == "__main__":
    main()
    
st.header('HUD Video Shortened')
st.markdown('The shortened 30 second version of the HUD is shown below. Notice how the beginning and end title pages have been cut out, and the focus has been set on a consistent chunk of video.')
def main1():
    shortened = "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened1.mp4"  
    
    # Display the video
    st.video(shortened)

if __name__ == "__main__":
    main1()

st.header('OCR Model Code')
initial_imports = '''
# Import streamlit 
import streamlit as st

# Import necessary libraries and packages
# pip install opencv-python
# pip install pytesseract 
import cv2
import pytesseract
import re
import pandas as pd
import numpy as np
'''
st.code(initial_imports, language='python')

more_path = '''
# Pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# File path for video
# Video from https://www.youtube.com/watch?v=WkZGL7RQBVw&ab_channel=AviationWeek
video = "/Users/jenniferdoan/Desktop/Shortened.mp4"
'''
st.code(more_path, language='python')

preprocessing = '''
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
'''
st.code(preprocessing, language='python')

ocr_model = '''
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
'''
st.code(ocr_model, language='python')

rois = '''
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
'''
st.code(rois, language='python')

st.header('Instructions to Operate HUD OCR Model')
st.markdown('1. Review the data source provided on this page. The videos are representative of the input data. The user is able to perform data exploration, visualization, and understand some of the cleaning measures that were taken.')
st.markdown('2. Review the source code for the OCR model. The preprocessing methods for the video are shown, as well the definitions of each function required to initialize and run the model.')
st.markdown('3. Simply click the **Auto Import Data** button. This will load the necessary libraries and packages, as well the video itself into the source code. The user will see a message that reads: "Loading libraries, packages, and data..." This will indicate the desired actions are happening successfully.')
st.markdown('4. Click the **Run OCR** button. This will run the rest of the model and print the output on the screen. The user will expect to see a message that reads: "OCR Action in Progress..." Furthermore, the user will see printed frames, as well as some output in a numbers array. There will be additional output that can be viewed in the terminal of the operating system.')
st.markdown('**Note**: Although it is not mandatory, it may be helpful to have the terminal open while using this software.')
st.markdown('**Note**: At any time after running the OCR model, the user will be able to click **Stop** at the top right corner of the page to stop the model from continuing.')

st.header('Implementation')
def run_code():
    st.write("Loading libraries, packages, and data...")
    # Import necessary libraries and packages
    # import cv2
    import pytesseract
    # import re
    # Pytesseract
    pytesseract.pytesseract.tesseract_cmd = 'https://drive.google.com/file/d/18QUqHKAlHyOwohoLwxxqViJg5bIkpM3d/view?usp=sharing'
    # File path for video
    # Video from https://www.youtube.com/watch?v=WkZGL7RQBVw&ab_channel=AviationWeek
    # video = "/Users/jenniferdoan/Desktop/Shortened.mp4"
if st.button("Auto Import Data"):
    run_code()

def run_code1():
    st.write("OCR Action in Progress...")
    # Preprocess the frame
    import cv2 
    import pytesseract
    import re
    video = "/Users/jenniferdoan/Desktop/Shortened.mp4"
    def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blur)
        thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closing
    
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
                
                        # Append the numbers to the array
                        if numbers:
                            numbers_array.append(numbers)

                        if text:
                            st.write(text)

                    # Show frame on streamlit
                    st.image(frame, channels="BGR", caption="Frame")
                
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

        # Print the results
        st.write("OCR Action Completed")
        
        st.write("Result:")
        for sublist in result:
            st.write('.'.join(sublist))

        st.write("Result 2:")
        for sublist in result_2:
            for item in sublist:
                st.write(item)
    
if st.button("Run OCR"):
    run_code1()
    
st.header('References')
references = "Atherton, K. (2022, May 6). Understanding the errors introduced by military AI applications. Brookings. https://www.brookings.edu/techstream/understanding-the-errors-introduced-by-military-ai-applications/ <br>[DontGetShot]. (2023, February 12). Michigan UFO Declassified F-16 HUD Footage [Video]. YouTube. https://www.youtube.com/watch?v=GZt-lordqBE&ab_channel=DontGetShot <br>Hamad, K. A., & Kaya, M. (2016). A detailed analysis of optical character recognition technology. International Journal of Applied Mathematics, Electronics and Computers, 244-249. https://doi.org/10.18100/ijamec.270374 <br>Wilson, N., Guragain, B., Verma, A., Archer, L., & Tavakolian, K. (2019). Blending human and machine: Feasibility of measuring fatigue through the aviation headset. Human Factors: The Journal of the Human Factors and Ergonomics Society, 62(4). https://doi.org/10.1177/0018720819849783"
st.markdown(references, unsafe_allow_html=True)
