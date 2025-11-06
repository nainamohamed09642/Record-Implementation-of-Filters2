
###Developed By : 
###Register Number: 
### Smoothing Filters
# In[1]:Using Averaging Filter


import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("")
image2=cv2.cvtColor(image1,cv2.COLOR_)
kernel=np.ones
image3=cv2.filter2D(image2,,kernel)
plt.figure(figsize=())
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot()


# In[2]:Using Weighted Averaging Filter


kernel1=
plt.figure(figsize=(8,8))
plt.subplot(1,)

plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()


# In[3]:Using Gaussian Filter


gaussian_blur=cv2.GaussianBlur()
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()




# In[4]:Using Median Filter


median=
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,)






