from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from crack import preprocess_image
from tensorflow.keras.models import load_model
import numpy as np

# Set the URL of the webpage
webpage_url = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/index.php'
 
# Set the CSS selector or XPath for the image element
css_selector = 'img:nth-child(13)'
xpath_selector = '//form/img'
 
# Create a WebDriver instance (you can use other browsers like Firefox as well)
driver = webdriver.Chrome()
 
# Open the webpage
driver.get(webpage_url)
 
try:
    # Find the image element by CSS selector
    img_element = driver.find_element(by=By.CSS_SELECTOR, value=css_selector)
    #img_element = driver.find_element(by=By.XPATH, value=xpath_selector)
 
    # Get the source URL of the image
    img_src = img_element.get_attribute('src')
 
    # Print or use the image source URL as needed
    print(f"Image source URL: {img_src}")
    
    full_image_url = f'{img_src}'

    # Send a GET request to fetch the image
    image_response = requests.get(full_image_url)

    # Check if the request for the image was successful
    if image_response.status_code == 200:
        # Save the image to a file
        image_filename = 'img.png'
        with open(image_filename, 'wb') as file:
            file.write(image_response.content)
        print(f"Image has been saved as {image_filename}")
        print(full_image_url)
    else:
        print("Failed to retrieve image")
        
finally:
    # Close the WebDriver
    driver.quit()
    
# Load the saved model
model = load_model('my_model.keras')

# Now use the preprocess_image function on your saved image
image_path = 'img.png'  # The image you saved from the webpage
processed_digits = preprocess_image(image_path)

# Use the model to predict on the processed digits
if processed_digits.shape[0] == 6:
    predictions = model.predict(processed_digits)
    predicted_numbers = np.argmax(predictions, axis=1)
    number_string = ''.join(map(str, predicted_numbers))
    print(number_string)
else:
    print("The image does not contain 6 digits.")