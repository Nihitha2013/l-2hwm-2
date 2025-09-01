import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('example.jpg')
image_rbg=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

(h,w)=image.shape[:2]
center=(w//2,h//2)
M=cv2.getRotationMatrix2D(center,45,1.0)

rotated=cv2.warpAffine(image,M,(w,h))

rotated_rgb=cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB)
plt.imshow(rotated_rgb)
plt.show()

brightness_matrix=np.ones(image.shape,dtype="uint8")*50
brighter=cv2.add(image, brightness_matrix)

brighter_rgb=cv2.cvtColor(brighter,cv2.COLOR_BGR2RGB)
plt.imshow(brighter_rgb)
plt.title("Brighter Image")
plt.show()

image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(image_grey)
plt.title("Grayscale Image")
plt.show()

cropped_image=image[100:300,200:400]
cropped_image_rgb=cv2.cvtColor(cropped_image,cv2.COLOR_BGR2RGB)
plt.imshow(cropped_image_rgb)
plt.title("Cropped Image")
plt.show()

cv2.imwrite('rotated_image.jpg', rotated)

cv2.imwrite('brighter_image.jpg', brighter)

cv2.imwrite('grayscale_image.jpg', image_grey)

cv2.imwrite('cropped_image.jpg', cropped_image)