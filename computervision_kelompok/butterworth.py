import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread("depositphotos_392859592-stock-photo-leather-american-football-white-background.jpg", cv.IMREAD_GRAYSCALE)

center_x = 80
center_y = 100
rad = 30

# Butterworth notch filter
def butterworth_notch(image2, center_x, center_y, rad):
    rows, cols = image2.shape 
    mask = np.ones((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if distance < rad:
                mask[i, j] = 0
    return mask
butterworth_filter = butterworth_notch(image, center_x, center_y, rad)  
fft = np.fft.fft2(image)
butterworth_image = fft * butterworth_filter
butterworth_image = np.fft.ifft2(butterworth_image).real 
butterworth_image = cv.normalize(butterworth_image, None, 0, 255, cv.NORM_MINMAX)
butterworth_image = np.uint8(butterworth_image)
mean = 0
stddev = 30
noisy_image = image + np.random.normal(mean, stddev, image.shape).astype(np.uint8)
plt.subplot(121), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image')
plt.subplot(122), plt.imshow(butterworth_image, cmap='gray'), plt.title('Image after Butterworth notch filters')
plt.show()
