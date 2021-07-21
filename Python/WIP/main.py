import cv2
import numpy as np
#import utilitis as utlis

curveList = []
avgVal = 10

def getHist(img,minPer=0.1,disp=False,region=1):
	if region == 1:
		histVal = np.sum(img, axis=0)
	else:
		histVal = np.sum(img[img.shape[0] // region:, :], axis=0)

	# print(histValues)
	maxVal = np.max(histVal)
	minVal = minPer * maxVal

	indexArray = np.where(histVal >= minVal)
	basePoint = int(np.average(indexArray))
	# print(basePoint)

	if disp:
		imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
		for x, intensity in enumerate(histVal):
			cv2.line(imgHist, (x, img.shape[0]), (x, img.shape[0] - intensity // 255 // region), (255, 0, 255), 1)
			cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
		return basePoint, imgHist

	return basePoint

def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
	cv2.namedWindow("Trackbars")
	cv2.resizeWindow("Trackbars", 360, 240)
	cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0], wT // 2, nothing)
	cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
	cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], wT // 2,nothing)
	cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT,nothing)

def nothing(a):
	pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def warpImg(img, points, w, h,inv=False):


	pts1 = np.float32(points)
	pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
	if inv:
		matrix = cv2.getPerspectiveTransform(pts2, pts1)
	else:
		matrix = cv2.getPerspectiveTransform(pts1, pts2)
	imgWarp = cv2.warpPerspective(img, matrix, (w, h))
	return imgWarp

def getHistogram(img, display=False, minVal=0.1, region=4):
	histValues = np.sum(img, axis=0)
	print(histValues)

def drawPoints(img, points):
	for x in range(4):
		cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
	return img

def getLaneCurve(img,display=2):
	imgCopy = img.copy()
	imgResult = img.copy()
	#### STEP 1
	# converted = convert_hls(img)
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.uint8([70, 0, 0])
	upper = np.uint8([255, 160, 255])

	mask = cv2.inRange(image, lower, upper)
    #cv2.imshow('t',mask)
	# Am identificat drumul pe care trebuie sa il urmeze masina
	# regiunea de interes
	result1 = img.copy()
	#cv2.imshow('TEST',mask)


	hT, wT, c = img.shape

	widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
	heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
	widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
	heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
	puntcte = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop),
	                      (widthBottom, heightBottom), (wT - widthBottom, heightBottom)])

	points = puntcte

	imgWarp = warpImg(mask, points, wT, hT)
	imgWarpPoints = drawPoints(imgCopy, points)
	#cv2.imshow('test', imgWarp)
	midPoint,HistImg=getHist(imgWarp,disp=True,minPer=0.5,region=4)
	curveAvgPoint, HistImg = getHist(imgWarp, disp=True, minPer=0.9, region=1)
	curveRaw=curveAvgPoint-midPoint
	#cv2.imshow('HistImg',HistImg)
	curveList.append(curveRaw)
	if len(curveList)>avgVal:
		curveList.pop(0)
	curve=int(sum(curveList)/len(curveList))


	#print(curveAvgPoint-midPoint)
	if display != 0:
		imgInvWarp = warpImg(imgWarp, points, wT, hT, inv=True)
		imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
		imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
		imgLaneColor = np.zeros_like(img)
		imgLaneColor[:] = 0, 255, 0
		imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
		imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
		midY = 450
		cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
		cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 5, 255), 5)
		cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
		for x in range(-30, 30):
			w = wT // 20
			cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
			         (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
		#fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
		#cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
	if display == 2:
		imgStacked = stackImages(0.7, ([img, imgWarpPoints, imgWarp],
		                                     [HistImg, imgLaneColor, imgResult]))
		cv2.imshow('ImageStack', imgStacked)
	elif display == 1:
		cv2.imshow('Resutlt', imgResult)
###############
	#getHistogram(imgWarp2)
################

	#imgWarped=None;

	##NORMALIZARE##
	curve=curve/100
	if curve>1: curve == 1
	if curve<-1: curve == -1
	return curve

if __name__ == '__main__':
	cap = cv2.VideoCapture('vid2.mp4')
	intialTrackBarVals = [102, 80, 20, 214]
	initializeTrackbars(intialTrackBarVals)
	frameCounter = 0
	while True:
		frameCounter += 1
		if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			frameCounter = 0

		success, img = cap.read()
		img = cv2.resize(img, (480, 240))
		curve = getLaneCurve(img,display=2)
		#print(curve)
		# cv2.imshow('Vid',img)
		cv2.waitKey(1)