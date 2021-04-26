
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def readImage(filename):

    img = cv2.imread(filename, 0)
    if img.ndim == 3:
        img = img[:,:,0] 
    return img

def kernel():
    size = 5
    sigma = 5
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = (-x/sigma**2)*np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
  
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass   
    return Z

def findCorners(img, window_size, k, threshold):

    Iy, Ix = np.gradient(img)
    Ixx = Ix**2
    Ixy = Iy*Ix
    Iyy = Iy**2
    theta = np.arctan2(Ix, Iy)
    height = img.shape[0]
    width = img.shape[1]
    offset = window_size//2


    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            if r < threshold:
                img[y][x] = 0
           
    return img, theta

def main():

    window_size = int(sys.argv[1])
    k = float(sys.argv[2])
    threshold = int(sys.argv[3])
    file_name = str(sys.argv[4])


    img = readImage(file_name)   
    img = cv2.filter2D(img,-1,kernel())
    img,theta = findCorners(img, window_size, k, threshold)
    img = non_max_suppression(img,theta)
        
    plt.imshow(img,cmap ='gray')         
    plt.show()




if __name__ == "__main__":
    main()