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

references = "Atherton, K. (2022, May 6). Understanding the errors introduced by military AI applications. Brookings. https://www.brookings.edu/techstream/understanding-the-errors-introduced-by-military-ai-applications/ <br>[DontGetShot]. (2023, February 12). Michigan UFO Declassified F-16 HUD Footage [Video]. YouTube. https://www.youtube.com/watch?v=GZt-lordqBE&ab_channel=DontGetShot <br>Hamad, K. A., & Kaya, M. (2016). A detailed analysis of optical character recognition technology. International Journal of Applied Mathematics, Electronics and Computers, 244-249. https://doi.org/10.18100/ijamec.270374 <br>Wilson, N., Guragain, B., Verma, A., Archer, L., & Tavakolian, K. (2019). Blending human and machine: Feasibility of measuring fatigue through the aviation headset. Human Factors: The Journal of the Human Factors and Ergonomics Society, 62(4). https://doi.org/10.1177/0018720819849783"
st.markdown(references, unsafe_allow_html=True)
