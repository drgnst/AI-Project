import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread("depositphotos_392859592-stock-photo-leather-american-football-white-background.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# lowpass filtering
gaussian_blur = cv.GaussianBlur(gray, (11,11), 0)


# Apply FFT
f_ori = np.fft.fft2(gray)
fshift_ori = np.fft.fftshift(f_ori)

# Calculate magnitude spectrum
magnitude_spectrum_ori = np.log(np.abs(fshift_ori) + 1)

f_gaus = np.fft.fft2(gaussian_blur)
fshift_gaus = np.fft.fftshift(f_gaus)
magnitude_spectrum_gaus = np.log(np.abs(fshift_gaus)+1)

res_img = [gray,magnitude_spectrum_ori, gaussian_blur, magnitude_spectrum_gaus]
res_title = ['Original Image',' Fourier Spectrum of Image', 'Image with gaussian lowpass filter', 'fourier spectrum of image with gaussian lowpass filter' ]
plt.figure(1, figsize=(10,10))
for i, (curr_img, curr_title) in enumerate(zip(res_img,res_title)):
    plt.subplot(2,2, (i+1))
    plt.imshow(curr_img, 'gray')
    plt.title(curr_title)
    plt.xticks([])
    plt.yticks([])
plt.show()



