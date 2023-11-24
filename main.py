from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from crack import preprocess_image
from tensorflow.keras.models import load_model
import numpy as np
import re
import cv2
from PIL import Image
from pytesseract import image_to_string

# Set the URL of the webpage
webpage_url = "https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/index.php"

# Set the CSS selector or XPath for the image element
css_selector = "img:nth-child(13)"
xpath_selector = "//form/img"

# Create a WebDriver instance (you can use other browsers like Firefox as well)
driver = webdriver.Chrome()

# Open the webpage
driver.get(webpage_url)

try:
    # Find the image element by CSS selector
    img_element = driver.find_element(by=By.CSS_SELECTOR, value=css_selector)
    # img_element = driver.find_element(by=By.XPATH, value=xpath_selector)

    # Get the source URL of the image
    img_src = img_element.get_attribute("src")

    # Print or use the image source URL as needed
    print(f"Image source URL: {img_src}")

    full_image_url = f"{img_src}"

    # Send a GET request to fetch the image
    image_response = requests.get(full_image_url)

    # Check if the request for the image was successful
    if image_response.status_code == 200:
        # Save the image to a file
        image_filename = "img.png"
        with open(image_filename, "wb") as file:
            file.write(image_response.content)
        print(f"Image has been saved as {image_filename}")
        print(full_image_url)
    else:
        print("Failed to retrieve image")

finally:
    # Close the WebDriver
    driver.quit()


def preprocess_image_for_ocr(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Additional preprocessing steps like denoising can be added here

    # Save or return preprocessed image
    preprocessed_image_path = "preprocessed_" + image_path
    cv2.imwrite(preprocessed_image_path, thresh)
    return preprocessed_image_path


def extract_6_digit_number(image_path):
    preprocessed_image_path = preprocess_image_for_ocr(image_path)
    text = image_to_string(preprocessed_image_path, config="outputbase digits")

    if text:
        print(text)

    # Find all sequences of digits
    all_digits = re.findall(r"\d+", text)

    # Concatenate them
    concatenated = "".join(all_digits)

    # Check if the concatenated string has at least 6 digits
    if len(concatenated) >= 6:
        # Return the first 6 digits
        return concatenated[:6]
    else:
        return None


# Assuming you have the image saved as 'img.png'
captcha_text = extract_6_digit_number("img.png")

if captcha_text:
    print(f"Extracted 6-digit number: {captcha_text}")
    # You can now use this `captcha_text` in your further code logic
    # captcha.send_keys(captcha_text)
else:
    print("No 6-digit number found in the image.")
