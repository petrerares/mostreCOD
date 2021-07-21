import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('im_03.jpg')

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


#############
#Subsection 1
#############

#plt.figure()
#plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title('Original Image');
#plt.subplot(122);plt.imshow(gray);plt.title('Grayscale Image');
#plt.show()
imageB, imageG, imageR = cv2.split(image)
# It converts the BGR color space of image to HSV color space
imageNew=cv2.medianBlur(image,5)
hsv = cv2.cvtColor(imageNew, cv2.COLOR_BGR2HSV)
#cv2.imshow('xkx',hsv)
filterHSV=cv2.medianBlur(hsv,3)
#cv2.imshow('klkl',filterHSV)
#cv2.imshow('vc',filterHSV)
# Threshold of yellow in HSV space
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# preparing the mask to overlay
#mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask = cv2.inRange(filterHSV, lower_yellow, upper_yellow)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(image, image, mask=mask)

#cv2.imshow('klklk',result)
#list(i)---->>>>> RED GREEN BLUE.....

#cv2.imshow('frame', image)
#cv2.imshow('mask', mask)
#cv2.imshow('result', result)

#maskBGR=cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)

#imageB, imageG, imageR = cv2.split(maskBGR)

#plt.figure()

#plt.subplot(141);plt.imshow(image[:,:,::-1]);plt.title('Original Image');
#plt.subplot(142);plt.imshow(imageB);plt.title('Blue Channel');
#plt.subplot(143);plt.imshow(imageG);plt.title('Green Channel');
#plt.subplot(144);plt.imshow(imageR);plt.title('Red Channel');
#plt.show()

thresh=150
maxValue=255

#th, dst_bin = cv2.threshold(imageR, thresh,maxValue, cv2.THRESH_BINARY)
th, dst_bin = cv2.threshold(imageR, thresh,maxValue, cv2.THRESH_BINARY)

#plt.figure()
#plt.imshow(dst_bin); plt.title('Threshold Image');
#plt.show()

kSize= (8,8)
kernel1= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

imageDilated=cv2.dilate(dst_bin, kernel1)
imageEroded=cv2.erode(imageDilated, kernel1)

#plt.figure()
#.subplot(121);plt.imshow(imageDilated);plt.title("Dilated image - bigger kernel")
#plt.subplot(122);plt.imshow(imageEroded);plt.title("Eroded Image")
#plt.show()

kSize = (1,1)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

imageDilated = cv2.dilate(dst_bin, kernel2, iterations=1)
imageDilated2 = cv2.dilate(dst_bin, kernel2, iterations=15)

#plt.figure()
#plt.subplot(121);plt.imshow(imageDilated2);plt.title("Dilated image iteration 5 - small kernel")

imageEroded2 = cv2.erode(imageDilated2, kernel2, iterations=40)

#plt.subplot(122);plt.imshow(imageEroded2);plt.title("Eroded Image iterations 6")
#plt.show()

#plt.figure()
#plt.imshow(imageEroded2,cmap='gray');plt.title('Grayscale');
result2 = cv2.bitwise_and(imageEroded2,imageEroded2, mask=mask)

#cv2.imshow('resulF',result2)


thresh = cv2.threshold(result2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
opening = 255 - cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
opening1=cv2.medianBlur(opening,3)
cv2.imwrite('segmented_03.jpg', opening)
cv2.imwrite('segmented_improved_03.jpg', opening1)
cv2.imshow('opening', opening1)



#############
#Subsection 2
#############
thresh=110
th1, dst_bin12 = cv2.threshold(imageR, thresh,maxValue, cv2.THRESH_BINARY)

_, imLabels1 = cv2.connectedComponents(dst_bin12)
plt.figure()
plt.title("ImLabeled")
plt.imshow(imLabels1)
cv2.imwrite('blobs_03.jpg', imLabels1)

opening=cv2.medianBlur(opening,3)



imThresh = opening
_, imLabels = cv2.connectedComponents(imThresh)

# Display the labels
nComponents = imLabels.max()
cv2.putText(imLabels, "Numar total: %d" %(nComponents),(650,100),cv2.FONT_HERSHEY_SIMPLEX,1,(1,1,1),2)
print("Numarul de elemente: ",nComponents)

plt.imshow(imLabels)
plt.title('Numarul de elemente total:')
#plt.title("Toate elementele afisate")
plt.show()

for i in range(nComponents+1):
    if i == 0:
        plt.title("Lamaile afisate: {}".format(i))
    else:
        plt.imshow(imLabels == i)
        plt.title("Elementul numarul : {}".format(i))
        plt.show()

cv2.imwrite('valid_blobs_03.jpg', imLabels)

#############
#Subsection 3
#############


img1 = cv2.imread('books.jpg', 0)          # queryImage
img2 = cv2.imread('anna.jpg', 0) # trainImage
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    #am pus distanta de 0.35
    if m.distance < 0.35*n.distance:
        good.append([m])
print(len(good))
#in good avem matchurile gasite, dar sunt de forma lista in lista [ [0], [1], [2] ] si e greu de sortat

distante = []
for i in range (len(good)):
    #am facut o alta lista in care adaug primele 20 matchuri ca asa zice in cerinta
    #aici am facut din lista in lista o lista simpla pentru a o putea sorta
    distante.append(good[i][0])
print(distante)

#sortare
distanteSortate = sorted(distante, key = lambda x:x.distance)
catTrebuie = []

# selectez primele 20 elemente pentru ca asa zice cerinta (15-20 match-uri)
for i in range(20):
    catTrebuie.append(distanteSortate[i])
    #drawmatchesknn imi dadea o eroare, asa ca acum transform lista de mai sus sortata intr-o lista cu mai multe liste de genul:
    # [ [0], [1], [2] ] si astfel nu mai am eroare cand desenez liniile
#aici introduc din nou elementele sub forma de lista in lista
listaProasta = [ [i] for i in catTrebuie]
print(len(listaProasta))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,listaProasta,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

cv2.imwrite('matched.jpg', img3)

#plt.show()
cv2.waitKey()