import cv2
from PIL import Image
import imutils
import numpy as np
import re
import os
import requests
import streamlit as st
import time

# Create a loading indicator for the easyocr import
with st.spinner('Loading EasyOCR module... This might take a minute.'):
    try:
        import easyocr
        st.success('EasyOCR loaded successfully!')
    except ImportError:
        st.error('Failed to import EasyOCR. Please make sure it\'s installed correctly.')
        st.info('Running: pip install easyocr')
        os.system('pip install easyocr')
        try:
            import easyocr
            st.success('EasyOCR installed and loaded successfully!')
        except ImportError:
            st.error('Failed to install EasyOCR. The app may not function properly.')

# Supabase credentials - consider using environment variables for these
SUPABASE_URL = "https://nyfaluazyaribgfqryxy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im55ZmFsdWF6eWFyaWJnZnFyeXh5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzA0NzQ1MCwiZXhwIjoyMDYyNjIzNDUwfQ.rbytQ-q5a8cN-A-LakAmtywl2VqXn-CiTeJXkhKJeIk"

def insert_data_to_supabase(plate, city):
    url = f"{SUPABASE_URL}/rest/v1/car"
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    data = {
        "plate_number": plate,
        "city": city
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)

@st.cache_data
def load_easyocr_reader():
    """Load the EasyOCR reader with caching to improve performance"""
    try:
        return easyocr.Reader(['en','th'])
    except Exception as e:
        st.error(f"Error loading EasyOCR reader: {str(e)}")
        return None

def process_image(image):
    try:
        # Convert to OpenCV format
        open_cv_image = np.array(image)
        # Convert RGB to BGR (OpenCV uses BGR)
        img = open_cv_image[:, :, ::-1].copy() 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
       
        keypoint = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoint)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        
        if location is None:
            return {"error": "No license plate contour found"}
            
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Get the cached reader or load it if not available
        reader = load_easyocr_reader()
        if reader is None:
            return {"error": "Failed to initialize EasyOCR. Please try again."}
            
        text = reader.readtext(cropped_image, detail = 0)

        text_str = ' '.join(text)
        
        plate_pattern = r"[ก-ฮ0-9]{1,}"  
        city_pattern = r"\b(กรุงเทพมหานคร|เชียงใหม่|ขอนแก่น|ลพบุรี|ลหบุรี)\b"

        plate = re.findall(plate_pattern, text_str)
        city = re.findall(city_pattern, text_str)

        response_data = {}
        
        if plate:
            response_data['plate'] = plate[0]
        
        if city:
            response_data['city'] = city[0]
            
        # Return the processed data
        return response_data
            
    except Exception as e:
        return {"error": str(e)}

# Set up the Streamlit UI
st.title('Thai License Plate Recognition')
st.write('Upload an image of a Thai license plate to recognize the plate number and city')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a button to process the image
    if st.button('Process Image'):
        st.write("Processing...")
        
        # Process the image
        with st.spinner('Analyzing license plate...'):
            result = process_image(image)
        
        # Check for errors
        if "error" in result:
            st.error(result["error"])
        else:
            # Display the results
            st.success("License plate recognized!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "plate" in result:
                    st.metric("License Plate", result["plate"])
                else:
                    st.warning("No license plate number detected")
            
            with col2:
                if "city" in result:
                    st.metric("City", result["city"])
                else:
                    st.warning("No city detected")
            
            # Add option to save to database
            if "plate" in result or "city" in result:
                if st.button("Save to Database"):
                    plate_val = result.get("plate", None)
                    city_val = result.get("city", None)
                    
                    success, db_result = insert_data_to_supabase(plate_val, city_val)
                    
                    if success:
                        st.success("Successfully saved to database!")
                    else:
                        st.error(f"Failed to save to database: {db_result}")