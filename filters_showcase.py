import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
# nacitanie obrazku
original_image = mpimg.imread('FIIT_budova.jpg')


# Convert to grayscale 
grayscale = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)



f, figure = plt.subplots(1, 2, figsize=(15, 4))
figure[0].set_title('Original Image')
figure[0].imshow(original_image)

figure[1].set_title('Grayscale image')
figure[1].imshow(grayscale, cmap='gray')



plt.show()



# vertikalny filter
vertical = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])


# horizontalny filter
horizontal = np.array([[ -1, -1, -1], 
                       [ 0, 0, 0], 
                       [ 1, 1, 1]])

sharpening = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])

horizontal_filter = cv2.filter2D(grayscale, -1, horizontal)
vertical_filter = cv2.filter2D(grayscale, -1, vertical)

f, figure = plt.subplots(1, 2, figsize=(15, 4))
figure[0].set_title('detection of horizontal edges')
figure[0].imshow(horizontal_filter, cmap='gray')

figure[1].set_title('detection of vertical edges')
figure[1].imshow(vertical_filter, cmap='gray')
plt.show()



f, figure = plt.subplots(1, 2, figsize=(15, 4))
figure[0].set_title('Grayscale image')
figure[0].imshow(grayscale, cmap='gray')



sharpening_filter = cv2.filter2D(grayscale,-1,sharpening)
figure[1].set_title('Grayscale image - sharpened')
figure[1].imshow(sharpening_filter,cmap='gray')

plt.show()