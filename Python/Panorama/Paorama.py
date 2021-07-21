import cv2
import matplotlib.pyplot as plt
import numpy as np






############
#Subsection 1
############
##SET 08###







img1=cv2.imread('shanghai-11.png');
img2=cv2.imread('shanghai-12.png');
img3=cv2.imread('shanghai-13.png');

img1HSV=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
img1HSV[:,:,2]+=17;
img1HSV[:,1,:]+=1;
img1HSV[1,:,:]+=10;
img1=cv2.cvtColor(img1HSV,cv2.COLOR_HSV2BGR)
#functia pentru incadrarea imaginilor dupa alipire/suprapunere(incadrarea panoramica)
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
#endregion



####################################################################################
#determinarea similaritatilor si alipirea imaginilor,daca imaginile au aceeasi incaltime.
####################################################################################


def matchCreatorForSameHight(img1, img2):
    orb = cv2.ORB_create()
    kp0, des0 = orb.detectAndCompute(img1, None)
    kp1, des1 = orb.detectAndCompute(img2, None)

    # Determinarea potrivirilor
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches1 = matcher.match(des0, des1, None)

    # Sortalrea potrivirilor
    matches1.sort(key=lambda x: x.distance, reverse=False)

    # Eliminarea potrivirilor neimportante(Nu avem nevoie decat de primele 15% din toate)
    GOOD_MATCH_PERCENT = 0.15

    numGoodMatches1 = int(len(matches1) * GOOD_MATCH_PERCENT)
    matches1 = matches1[:numGoodMatches1]
    points1 = np.zeros((len(matches1), 2), dtype=np.float32)
    points2 = np.zeros((len(matches1), 2), dtype=np.float32)
    for i, match in enumerate(matches1):
        points1[i, :] = kp0[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    # gasirea homografiei
    h1, mask1 = cv2.findHomography(points2, points1, cv2.RANSAC)


    im0Height, im0Width, channels = img1.shape
    im1Height, im1Width, channels = img2.shape
    #Stabilirea regulii de interpretat in functie de dimeniunile imaginilor de alipit
    im1Aligned = cv2.warpPerspective(img2, h1,(im0Width+im1Width, im1Height))

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)


    img3 = cv2.drawMatches(img1, kp0, img2, kp1, matches1, None, **draw_params)
    cv2.imshow("original_image_drawMatches", img3)

    # Alinierea celor 2 imagini
    stitchedImage = np.copy(im1Aligned)
    stitchedImage[0:im0Height,0:im0Width] = img1
    stitchedImage = trim(stitchedImage)


    plt.figure()
    plt.imshow(stitchedImage)
    plt.show()
    return stitchedImage





####################################################################################
#determinarea similaritatilor si alipirea imaginilor,daca imaginile au aceeasi latime
####################################################################################




def matchCreatorForSameWidth(img1, img2):
    orb = cv2.ORB_create()
    kp0, des0 = orb.detectAndCompute(img1, None)
    kp1, des1 = orb.detectAndCompute(img2, None)

    # Determinarea potrivirilor
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches1 = matcher.match(des0, des1, None)


    matches1.sort(key=lambda x: x.distance, reverse=False)

    # Eliminarea potrivirilor neimportante(Nu avem nevoie decat de primele 15% din toate)
    GOOD_MATCH_PERCENT = 0.15

    numGoodMatches1 = int(len(matches1) * GOOD_MATCH_PERCENT)
    matches1 = matches1[:numGoodMatches1]
    points1 = np.zeros((len(matches1), 2), dtype=np.float32)
    points2 = np.zeros((len(matches1), 2), dtype=np.float32)
    for i, match in enumerate(matches1):
        points1[i, :] = kp0[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    #  gasirea homografiei
    h1, mask1 = cv2.findHomography(points2, points1, cv2.RANSAC)


    im0Height, im0Width, channels = img1.shape
    im1Height, im1Width, channels = img2.shape
    # Stabilirea regulii de interpretat in functie de dimeniunile imaginilor de alipit
    im1Aligned = cv2.warpPerspective(img2, h1,(im0Width, im1Height+im0Height))

    # Alinierea celor 2 imagini
    stitchedImage = np.copy(im1Aligned)
    stitchedImage[0:im0Height,0:im0Width] = img1
    stitchedImage = trim(stitchedImage)

    plt.figure()
    plt.imshow(stitchedImage)
    plt.show()
    return stitchedImage


####################################################################################
#determinarea similaritatilor si alipirea imaginilor,daca dimeniunile lor difera
####################################################################################


def matchCreatorForDifferentHeightAndWidth(imag1, imag2):
    orb = cv2.ORB_create()
    kp0, des0 = orb.detectAndCompute(imag1, None)
    kp1, des1 = orb.detectAndCompute(imag2, None)

    # Determinarea potrivirilor
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches1 = matcher.match(des0, des1, None)

    #Sortarea lor
    matches1.sort(key=lambda x: x.distance, reverse=False)

    # Eliminarea potrivirilor neimportante(Nu avem nevoie decat de primele 15% din toate)
    GOOD_MATCH_PERCENT = 0.15

    numGoodMatches1 = int(len(matches1) * GOOD_MATCH_PERCENT)
    matches1 = matches1[:numGoodMatches1]
    points1 = np.zeros((len(matches1), 2), dtype=np.float32)
    points2 = np.zeros((len(matches1), 2), dtype=np.float32)
    for i, match in enumerate(matches1):
        points1[i, :] = kp0[match.queryIdx].pt
        points2[i, :] = kp1[match.trainIdx].pt

    #  gasirea homografiei
    h1, mask1 = cv2.findHomography(points2, points1, cv2.RANSAC)


    im0Height, im0Width, channels = imag1.shape
    im1Height, im1Width, channels = imag2.shape
    # Stabilirea regulii de interpretat in functie de dimeniunile imaginilor de alipit
    im1Aligned = cv2.warpPerspective(imag2, h1,(im1Width+im0Width, im1Height+im0Height))
	#cv2.imshow('alin',im1Aligned)
    # Alinierea celor 2 imagini
    stitchedImage = np.copy(im1Aligned)
    stitchedImage[0:im0Height,0:im0Width] = imag1
    stitchedImage = trim(stitchedImage)

    plt.figure()
    plt.imshow(stitchedImage)
    plt.show()

    return stitchedImage


Img_1st = matchCreatorForSameHight(img1, img2)
Img_2nd = matchCreatorForSameHight(Img_1st , img3)
#trdImg=matchCreatorForSameHight(Img_2nd, img3)

cv2.imshow('res2',Img_1st)
cv2.imshow('res3',Img_2nd)
#cv2.imshow('res4',trdImg)
#cv2.imshow('res5',t4Img)

cv2.waitKey()
#firstHalf = matchCreatorForSameHight(img2, img3)
#secondHalf = matchCreatorForSameHight(img1, Img_1st)
#trdImg=matchCreatorForSameHight(secondHalf, img3)

#res2=result
#cv2.imshow('res',res2)
#cv2.imshow('res12',Img_1st)
#cv2.imshow('res13',Img_2nd)
#cv2.imshow('res14',trdImg)
#cv2.imshow('res5',t4Img)



#Subsection 2
#SET 03


#########################################
#########################################
#################SIFT####################
#########################################
#########################################


def siftDetectAndDrawBox(object, background):
   #initializarea imaginilor si crearea obiectului sift
   # img1 = object
   # img2 = background
    sift = cv2.SIFT_create()

    # determinarea punctelor cheie/keypoints (kp1/2) si a descriptorilor(des1/2)
    kp1, des1 = sift.detectAndCompute(object, None)
    kp2, des2 = sift.detectAndCompute(background, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Salvam toate rezultatele bune(potrivirile in good)
    good = []
    #Verificam numarul de potriviri daca este mai mare ca cel specificat in cerinta
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good.append(m)
    print("Numarul de potriviri: ", len(good))
    if len(good) > MIN_GOOD_MATCHES:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, c = object.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        background = cv2.polylines(background, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        imagFin = cv2.drawMatches(object, kp1, background, kp2, good, None, flags=2)
        plt.imshow(imagFin)
        plt.show()



    else:
        print("Nu exista suficiente puncte de interes similare")


#########################################
#########################################
#################ORB#####################
#########################################
#########################################
def orbDetectAndDrawBox(object, background):
    # initializarea imaginilor si crearea obiectului sift
	orb = cv2.ORB_create()

	# determinarea punctelor cheie/keypoints (kp1/2) si a descriptorilor(des1/2)
	kp1, des1 = orb.detectAndCompute(object, None)
	kp2, des2 = orb.detectAndCompute(background, None)

	# Determinarea potrivirilor
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# potrivirile
	matches = bf.match(des1, des2)

	# Sortam potrivirile in functie de distanta lor
	matches = sorted(matches, key=lambda x: 0.7)
	if (len(matches) > MIN_GOOD_MATCHES):
		good_matches = matches[:10]

		src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		h, w, c = object.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, M)
		background = cv2.polylines(background, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
		imagFin = cv2.drawMatches(object, kp1, background, kp2, good_matches, None, flags=2)

		plt.imshow(imagFin)
		plt.show()
	else:
		print("Nu exista suficiente puncte de interes similare")
	cv2.waitKey()



clutter01 = cv2.imread('clutter01.jpg')  #cv2.IMREAD_GRAYSCALE)
clutter02 = cv2.imread('clutter02.jpg')  #cv2.IMREAD_GRAYSCALE)
clutter03 = cv2.imread('clutter03.jpg')#, cv2.IMREAD_GRAYSCALE)
clutter04 = cv2.imread('clutter04.jpg')#, cv2.IMREAD_GRAYSCALE)
objToFind = cv2.imread('object.jpg')#, cv2.IMREAD_GRAYSCALE)

im_h = cv2.hconcat([clutter01, clutter02 ])
im_h2 = cv2.hconcat([im_h, clutter03 ])
im_h3 = cv2.hconcat([im_h2, clutter04 ])


obj_rgb=objToFind
clutter01_rgb=clutter01
clutter02_rgb=clutter02
clutter03_rgb=clutter03
clutter04_rgb=clutter04


MIN_GOOD_MATCHES = 10

#########################################
#########################################
#################SIFT####################
#########################################
#########################################




siftDetectAndDrawBox(obj_rgb, clutter01_rgb)
siftDetectAndDrawBox(obj_rgb, clutter02_rgb)
siftDetectAndDrawBox(obj_rgb, clutter03_rgb)
siftDetectAndDrawBox(obj_rgb, clutter04_rgb)
siftDetectAndDrawBox(obj_rgb, im_h3)

#########################################
#########################################
#################ORB####################
#########################################
#########################################



orbDetectAndDrawBox(obj_rgb, clutter01_rgb)
orbDetectAndDrawBox(obj_rgb, clutter02_rgb)
orbDetectAndDrawBox(obj_rgb, clutter03_rgb)
orbDetectAndDrawBox(obj_rgb, clutter04_rgb)
orbDetectAndDrawBox(obj_rgb, im_h3)