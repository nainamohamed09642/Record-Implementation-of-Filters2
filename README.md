# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.

### Step2
Convert the image from BGR to RGB.

### Step3
Apply the required filters for the image separately.

### Step4
Plot the original and filtered image by using matplotlib.pyplot.

### Step5
End the program.

## Program:

### Name: JEEVAN ES
### Register Number: 212223230091



#### Convolution Result
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread('cat39.jpeg')

kernel = np.ones((5,5), dtype = np.float32) / 5**2
print (kernel)

image = cv2.imread('cat39.jpeg')

dst = cv2.filter2D(image, ddepth = -1, kernel = kernel)

plt.figure(figsize = [20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(image[:,:,::-1]); plt.title("Original Image")
plt.subplot(122); plt.axis('off'); plt.imshow(dst[:,:,::-1]);   plt.title("Convolution Result")


```
#### Output
<img width="1246" height="591" alt="image" src="https://github.com/user-attachments/assets/966b84db-c31c-4820-bd9b-b169002db59b" />

### 1. Smoothing Filters

#### i) Using Averaging Filter

```python
average_filter = cv2.blur(image, (30,30))

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(average_filter[:, :, ::-1]); plt.title('Output Image ( Average Filter)')
```

#### Output
<img width="1276" height="538" alt="image" src="https://github.com/user-attachments/assets/93b58a31-225d-4b98-863d-61eceaf4f6d3" />



#### ii) Using Weighted Averaging Filter
```
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])/16
weighted_average_filter = cv2.filter2D(image, -1, kernel)

plt.figure(figsize = (18, 6))
plt.subplot(121);plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122);plt.imshow(weighted_average_filter[:, :, ::-1]); plt.title('Output Image(weighted_average_filter)');plt.show()
```
#### Output
<img width="1300" height="524" alt="image" src="https://github.com/user-attachments/assets/d206539e-a6d3-4c30-a6bb-f7aa7d5a4aaa" />



#### iii) Using Gaussian Filter
```
gaussian_filter = cv2.GaussianBlur(image, (29,29), 0, 0)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(gaussian_filter[:, :, ::-1]); plt.title('Output Image ( Gaussian Filter)')
```
#### Output
<img width="1374" height="548" alt="image" src="https://github.com/user-attachments/assets/ed933c5c-dcf3-4fec-adbf-049db9d9a4d2" />



#### iv)Using Median Filter
```
median_filter = cv2.medianBlur(image, 19)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(median_filter[:, :, ::-1]); plt.title('Output Image ( Median_filter)')
```
#### Output
<img width="1322" height="553" alt="image" src="https://github.com/user-attachments/assets/080aa64e-b6a1-4941-a5d2-fe30234eacce" />



### 2. Sharpening Filters

#### i) Using Laplacian Linear Kernal
```
# i) Using Laplacian Kernel (Manual Kernel)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_laplacian_kernel = cv2.filter2D(image, -1, kernel = laplacian_kernel)

# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(sharpened_laplacian_kernel[:, :, ::-1]); plt.title('Output Image ( Laplacian_filter)')
```
#### Output
<img width="1310" height="526" alt="image" src="https://github.com/user-attachments/assets/2893b53d-bb35-49cd-b384-4206ffaf31a2" />


#### ii) Using Laplacian Operator
```
# ii) Using Laplacian Operator (OpenCV built-in)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
laplacian_operator = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_operator = np.uint8(np.absolute(laplacian_operator))

# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(131); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(132); plt.imshow(gray_image, cmap='gray'); plt.title('Gray_image')
plt.subplot(133); plt.imshow(laplacian_operator,cmap='gray'); plt.title('Output Image ( Laplacian_filter)')
```
#### Output
<img width="1345" height="443" alt="image" src="https://github.com/user-attachments/assets/a10ef955-10ea-466c-bf49-94e1bd3cdbd3" />


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
