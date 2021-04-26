import numpy as np
import matplotlib.pyplot as plt 
import math
import cv2
import sys

def readImage(filename):

    img = plt.imread(filename, 0)
    if img.ndim == 3:
        img = img[:,:,0] 
    return img

def Hx(size, s):
    sigma = s
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = (-x/sigma**2)*np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g

def Hy(size, s):
    sigma = s
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = (-y/sigma**2)*np.exp(-((x**2 + y**2) / (2.0*sigma**2)))
    return g

def magnitude(matrix1,matrix2):
    new_matrix = [[0 for i in range(len(matrix1[0]))] for j in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):          
            new_matrix[i][j] = math.sqrt(matrix1[i][j]**2 + matrix2[i][j]**2)
    return new_matrix

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

def threshold(img,low,high):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        highThreshold = img.max() * high;
        lowThreshold = highThreshold * low;

        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > highThreshold):
                    Z[i][j] = 255
                elif (img[i][j] < lowThreshold):
                    Z[i][j] = 0
                else:
                    Z[i][j] = img[i][j]

        return Z

def hysteresis(img):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] != 255 and img[i,j] != 0):
                try:
                    if ((img[i+1, j-1] == 255) or (img[i+1, j] == 255) or (img[i+1, j+1] == 255)
                        or (img[i, j-1] == 255) or (img[i, j+1] == 255)
                        or (img[i-1, j-1] == 255) or (img[i-1, j] == 255) or (img[i-1, j+1] == 255)):
                        img[i, j] = 255
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img




def findEdge(img,sigma,lowThreshold,highThreshold):

    Ix = cv2.filter2D(img,-1,Hx(3,sigma))
    Iy = cv2.filter2D(img,-1,Hy(3,sigma))
    M = np.asarray(magnitude(Ix,Iy))
    theta = np.arctan2(Ix, Iy)
    M = non_max_suppression(M, theta)
    M = threshold(M,lowThreshold,highThreshold)
    M = hysteresis(M)

    return M


def main():

    file_name = str(sys.argv[1])
    sigma = int(sys.argv[2])
    low = float(sys.argv[3])
    high = float(sys.argv[4])

    img = readImage(file_name)
    img = findEdge(img, sigma, low, high)

    plt.imshow(img,cmap ='gray')  
    plt.show()


if __name__ == "__main__":
    main()