import streamlit as st
import cv2
import pytesseract
import re

# Set pytesseract path
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Define the video path
video_path = "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened1.mp4"

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
                    roi = frame[y:y + h, x:x + w]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    preprocessed = preprocess_frame(roi)
                    text = pytesseract.image_to_string(preprocessed, config='')
                    numbers = re.findall(r'\d+', text)

                    if text:
                        st.write(text)

                    # Append the numbers to the array
                    if numbers:
                        numbers_array.append(numbers)

                    # Display the numbers on the frame
                    cv2.putText(frame, ','.join(numbers), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                2)

                st.image(frame, channels="BGR", caption="Frame")

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return numbers_array


# Define the main function
def main():
    st.title("Head Up Display (HUD) Optical Character Recognition (OCR) Model")
    st.markdown('Jennifer Doan')
    st.markdown('**Bottom line up front**: Scroll down to the bottom of this page to implement the HUD OCR now!')

    st.header('Introduction')
    st.markdown('This is a web-based application to implement an OCR model on HUD videos.')
    st.markdown(
        'The main objective for this project is to recognize text and/or numbers in streaming aircraft HUD recordings.')

    st.header('HUD Video Example')
    st.markdown(
        'This is the HUD video taken directly from YouTube and downloaded as an .mp4 file. It can be found here: '
        'https://www.youtube.com/watch?v=aSzXqlnT7nQ&ab_channel=NASAArmstrongFlightResearchCenter')

    # Display the video
    F16_HUD = "https://github.com/jenny271173/HUD-OCR/raw/main/F16_HUD.mp4"
    st.video(F16_HUD)

    st.header('HUD Video Shortened')
    st.markdown('The shortened 30-second version of the HUD is shown below. Notice how the beginning and end title pages have been cut out, and the focus has been set on a consistent chunk of video.')

    # Display the video
    shortened = "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened1.mp4"
    st.video(shortened)

    st.header('OCR Model Code')
    initial_imports = '''
    # Import necessary libraries and packages
    import cv2
    import pytesseract
    import re
    import pandas as pd
    import numpy as np
    '''
    st.code(initial_imports, language='python')

    more_path = '''
    # Set pytesseract path
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

    # Define the video path
    video_path = "https://github.com/jenny271173/HUD-OCR/raw/main/Shortened.mp4"
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
                        roi = frame[y:y + h, x:x + w]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        preprocessed = preprocess_frame(roi)
                        text = pytesseract.image_to_string(preprocessed, config='')
                        numbers = re.findall(r'\d+', text)

                        if text:
                            st.write(text)

                        # Append the numbers to the array
                        if numbers:
                            numbers_array.append(numbers)

                        # Display the numbers on the frame
                        cv2.putText(frame, ','.join(numbers), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    2)

                    st.image(frame, channels="BGR", caption="Frame")

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
        result = run_OCR(video_path, ROI, num_frames=10)

        ROI_2 = {
            'ROI 2': (630, 540, 95, 35)
        }
        result_2 = run_OCR(video_path, ROI_2, num_frames=10)

        st.write("Numbers Array:", result)
        st.write("Numbers Array 2:", result_2)
    '''
    st.code(rois, language='python')

    st.header('Instructions to Operate HUD OCR Model')
    st.markdown(
        '1. Review the data source provided on this page. The videos are representative of the input data. The user is able to perform data exploration, visualization, and understand some of the cleaning measures that were taken.')
    st.markdown('2. Review the source code for the OCR model. The preprocessing methods for the video are shown, as well the definitions of each function required to initialize and run the model.')
    st.markdown('3. Simply run the code by clicking the **Run OCR** button below. This will execute the OCR model on the video frames and display the processed frames along with the detected numbers.')
    st.markdown('4. View the output in the "Numbers Array" section below the code.')

    # Run the code on button click
    if st.button('Run OCR'):
        st.write("OCR Action in Progress...")
        st.write("Loading video...")

        # Define the ROIs
        ROI = {
            'ROI 1': (525, 540, 95, 35)
        }
        result = run_OCR(video_path, ROI, num_frames=10)

        ROI_2 = {
            'ROI 2': (630, 540, 95, 35)
        }
        result_2 = run_OCR(video_path, ROI_2, num_frames=10)

        st.write("Numbers Array:", result)
        st.write("Numbers Array 2:", result_2)

# Run the main function
if __name__ == '__main__':
    main()
