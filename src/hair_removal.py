import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '../data/raw/images/ISIC_0024320.jpg'

# Write a function to execute the dull razor algorithm
def dull_razor(image_path, kernel_size=(17, 17), blur_ksize=(3, 3), inpaint_radius=10):

    """
    Implements the Dull Razor Algorithm for hair removal and displays step-by-step debugging.
    
    Parameters:
        image_path (str): Path to the input image.
        kernel_size (tuple): Size of the morphological structuring element.
        blur_ksize (tuple): Size of Gaussian blur kernel.
        inpaint_radius (int): Radius for inpainting.

    Returns:
        img_no_hair (unknown): An image with hair removed.
    """

    # Read the image
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image '{image_path}'. Check file path.")
        return None

    # Covert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Apply Blackhat Morphological Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # Use MORPH_RECT for straight hair edges
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Apply Gaussian blur to the image
    blackhat_blur = cv2.GaussianBlur(blackhat, blur_ksize, cv2.BORDER_DEFAULT)

    # Threshold the gradient image
    _, th = cv2.threshold(blackhat_blur, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the thresholded image
    img_no_hair = cv2.inpaint(img, th, inpaint_radius, cv2.INPAINT_TELEA)

    # Display the images
    plt.figure(figsize=(15, 15))
    plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(223), plt.imshow(blackhat, cmap='gray'), plt.title('Blackhat Morphological Operation')
    plt.subplot(222), plt.imshow(blackhat_blur, cmap='gray'), plt.title('Gaussian Blur')
    plt.subplot(224), plt.imshow(cv2.cvtColor(img_no_hair, cv2.COLOR_BGR2RGB)), plt.title('Hair Removed Image')
    plt.show()

    return img_no_hair

img_no_hair = dull_razor(image_path)

if img_no_hair is not None:
    print("Hair removal completed successfully.")
    cv2.imwrite("hair_removed_image.jpg", img_no_hair)  # Save the output image
else:
    print("Hair removal failed.")
