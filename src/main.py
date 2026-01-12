import cv2
import matplotlib.pyplot as plt

# test for now

img = cv2.imread("imgs/TUMO_0.JPG")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
