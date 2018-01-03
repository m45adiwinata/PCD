import numpy
import cv2
import math
from matplotlib import pyplot as plt

#generating histogram
print("Generating Image Histogram")
image = cv2.imread('image1.jpg')
color = ('red', 'green', 'blue')
for i, col in enumerate(color):
    histogram = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histogram,color = col)
    plt.xlim([0,256])
plt.show()

#grayscale
print("Creating Grayscale Image")
image = cv2.imread('image1.jpg')
hasil = image
rows = image.shape[0]
cols = image.shape[1]
for i in range(rows):
    for j in range(cols):
        red = int(image[i,j,0])
        green = int(image[i,j,1])
        blue = int(image[i,j,2])
        mean = float(red+green+blue)/3
        hasil[i,j] = mean
cv2.imwrite('grayscale.jpg',hasil)

#biner
print("Creating Binary Image")
image = hasil
for i in range(rows):
    for j in range(cols):
        px = int(image[i,j,0])
        if(px<128):
            hasil[i,j] = 0
        else:
            hasil[i,j] = 255
cv2.imwrite('binary.jpg',hasil)

#mean konvolusi
print("Mean Convolution On Progress")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel = numpy.ones((3,3), dtype="float32")
kernelW = 3
kernelH = 3
faktor = kernelW * kernelH
pad = (kernelW - 1) / 2

for i in range(pad, rows-pad):
    for j in range(pad, cols-pad):
        value_red = 0
        value_green = 0
        value_blue = 0
        for k in range(kernelH):
            for l in range(kernelW):
                temp = float(kernel[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                value_red = value_red + float(temp*pxR)
                value_green = value_green + float(temp*pxG)
                value_blue = value_blue + float(temp*pxB)
        hasil[i,j,0] = float(value_red) / faktor
        hasil[i,j,1] = float(value_green) / faktor
        hasil[i,j,2] = float(value_blue) / faktor
cv2.imwrite('Mean Convolution.jpg',hasil)

#median konvolusi
print("Median Convolution On Progress")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols, 3), dtype="int32")
pad = (kernelH - 1) / 2
medianR = []
medianG = []
medianB = []
value = 0
z = 0
for i in range(pad, rows-pad):
    for j in range(pad, cols-pad):
        for k in range(kernelH):
            for l in range(kernelW):
                medianR.append(float(image[i-pad+k, j-pad+l, 0]))
                medianG.append(float(image[i-pad+k, j-pad+l, 1]))
                medianB.append(float(image[i-pad+k, j-pad+l, 2]))
                value = value + 1
        for m in range(z, value-1):
            for n in range(m+1, value):
                if medianR[m] > medianR[n]:
                    temp = medianR[m]
                    medianR[m] = medianR[n]
                    medianR[n] = temp
                if medianG[m] > medianG[n]:
                    temp = medianR[m]
                    medianG[m] = medianG[n]
                    medianG[n] = temp
                if medianB[m] > medianB[n]:
                    temp = medianB[m]
                    medianB[m] = medianB[n]
                    medianB[n] = temp
        hasil[i,j,0] = medianR[(z+value-1)/2]
        hasil[i,j,1] = medianG[(z+value-1)/2]
        hasil[i,j,2] = medianB[(z+value-1)/2]
        z = value
cv2.imwrite('Median Convolution.jpg', hasil)

#gaussian blur
print("Gaussian Blur On Progress")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel = numpy.array((
    [1,2,1],
    [2,4,2],
    [1,2,1]), dtype="float32")
faktor = 16
pad = (kernelW - 1) / 2
for i in range(pad, rows-pad):
    for j in range(pad, cols-pad):
        valueR = 0
        valueG = 0
        valueB = 0
        for k in range(kernelH):
            for l in range(kernelW):
                temp = float(kernel[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                valueR += float(temp*pxR)
                valueG += float(temp*pxG)
                valueB += float(temp*pxB)
        hasil[i,j,0] = valueR / faktor
        hasil[i,j,1] = valueG / faktor
        hasil[i,j,2] = valueB / faktor
cv2.imwrite('Gaussian Blur.jpg',hasil)

#robert edge detection
print("Performing Robert Edge-Detection")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel_A = numpy.array((
    [0,0,0],
    [0,0,-1],
    [0,1,0]), dtype="float32")
kernel_B = numpy.array((
    [0,0,0],
    [0,0,-1],
    [0,1,0]), dtype="float32")
pad = (kernelH - 1) /2
for i in range(pad,rows-pad):
    for j in range(pad, cols-pad):
        valueXR = 0
        valueXG = 0
        valueXB = 0
        valueYR = 0
        valueYG = 0
        valueYB = 0
        for k in range(kernelH):
            for l in range(kernelW):
                tempX = float(kernel_A[k,l])
                tempY = float(kernel_B[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                valueXR += float(tempX*pxR)
                valueXG += float(tempX*pxG)
                valueXB += float(tempX*pxB)
                valueYR += float(tempY*pxR)
                valueYG += float(tempY*pxG)
                valueYB += float(tempY*pxB)
        xR = valueXR * valueXR
        xG = valueXG * valueXG
        xB = valueXB * valueXB
        yR = valueYR * valueYR
        yG = valueYG * valueYG
        yB = valueYB * valueYB
        zR = xR + yR
        zG = xG + yG
        zB = xB + yB
        hasil[i,j,0] = math.sqrt(zR)
        hasil[i,j,1] = math.sqrt(zG)
        hasil[i,j,2] = math.sqrt(zB)
cv2.imwrite('Robert.jpg', hasil)

#sobel edge detection
print("Performing Sobel Edge-Detection")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel_A = numpy.array((
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]), dtype="float32")
kernel_B = numpy.array((
    [1,2,1],
    [0,0,-1],
    [-1,-2,-1]), dtype="float32")
pad = (kernelH - 1) /2
for i in range(pad,rows-pad):
    for j in range(pad, cols-pad):
        valueXR = 0
        valueXG = 0
        valueXB = 0
        valueYR = 0
        valueYG = 0
        valueYB = 0
        for k in range(kernelH):
            for l in range(kernelW):
                tempX = float(kernel_A[k,l])
                tempY = float(kernel_B[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                valueXR += float(tempX*pxR)
                valueXG += float(tempX*pxG)
                valueXB += float(tempX*pxB)
                valueYR += float(tempY*pxR)
                valueYG += float(tempY*pxG)
                valueYB += float(tempY*pxB)
        xR = valueXR * valueXR
        xG = valueXG * valueXG
        xB = valueXB * valueXB
        yR = valueYR * valueYR
        yG = valueYG * valueYG
        yB = valueYB * valueYB
        zR = xR + yR
        zG = xG + yG
        zB = xB + yB
        hasil[i,j,0] = math.sqrt(zR)
        hasil[i,j,1] = math.sqrt(zG)
        hasil[i,j,2] = math.sqrt(zB)
cv2.imwrite('Sobel.jpg', hasil)

#prewitt negative image
print("Performing Prewitt for Negative Image")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel_A = numpy.array((
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]), dtype="float32")
kernel_B = numpy.array((
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]), dtype="float32")
pad = (kernelH - 1) /2
for i in range(pad,rows-pad):
    for j in range(pad, cols-pad):
        valueXR = 0
        valueXG = 0
        valueXB = 0
        valueYR = 0
        valueYG = 0
        valueYB = 0
        for k in range(kernelH):
            for l in range(kernelW):
                tempX = float(kernel_A[k,l])
                tempY = float(kernel_B[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                valueXR += float(tempX*pxR)
                valueXG += float(tempX*pxG)
                valueXB += float(tempX*pxB)
                valueYR += float(tempY*pxR)
                valueYG += float(tempY*pxG)
                valueYB += float(tempY*pxB)
        xR = valueXR * valueXR
        xG = valueXG * valueXG
        xB = valueXB * valueXB
        yR = valueYR * valueYR
        yG = valueYG * valueYG
        yB = valueYB * valueYB
        zR = xR + yR
        zG = xG + yG
        zB = xB + yB
        hasil[i,j,0] = math.sqrt(zR)
        hasil[i,j,1] = math.sqrt(zG)
        hasil[i,j,2] = math.sqrt(zB)
cv2.imwrite('Prewitt.jpg', hasil)

#laplacian for
print("Performing Laplacian for Negative Image")
image = cv2.imread('image1.jpg')
hasil = numpy.zeros((rows, cols,3), dtype="int32")
kernel = numpy.array((
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]), dtype="float32")
pad = (kernelW - 1) /2
for i in range(pad,rows-pad):
    for j in range(pad, cols-pad):
        valueR = 0
        valueG = 0
        valueB = 0
        for k in range(kernelH):
            for l in range(kernelW):
                temp = float(kernel[k,l])
                pxR = float(image[i-pad+k, j-pad+l, 0])
                pxG = float(image[i-pad+k, j-pad+l, 1])
                pxB = float(image[i-pad+k, j-pad+l, 2])
                valueR += float(temp*pxR)
                valueG += float(temp*pxG)
                valueB += float(temp*pxB)
        hasil[i,j,0] = valueR
        hasil[i,j,1] = valueG
        hasil[i,j,2] = valueB
cv2.imwrite('Laplacian.jpg', hasil)
