import cv2
import time

img1 = cv2.imread("test_img/10.jpg")
img2 = cv2.imread("test_img/11.jpg")
img3 = cv2.imread("test_img/13.jpg")

img1 = cv2.resize(img1, (112, 112))
img2 = cv2.resize(img2, (112, 112))
s = time.time()
img3 = cv2.resize(img3, (112, 112))

img1_var = cv2.Laplacian(img1, cv2.CV_64F).var()
print("blur detect time", time.time() - s)
print()
img2_var = cv2.Laplacian(img2, cv2.CV_64F).var()
img3_var = cv2.Laplacian(img3, cv2.CV_64F).var()

print(img1_var, img2_var, img3_var)
