import cv2

camera=cv2.VideoCapture(0)

def Inregistrare(afisare= False,rez=[480,240]):
	_,img=camera.read()
	img=cv2.resize(img,(rez[0],rez[1]))
	if afisare==1:
		cv2.imshow('inregistrare',img)
	return img

if __name__== '__main__':
	while True:
		img=Inregistrare(True)
		key = cv2.waitKey(30)
		if key == ord('q'):
			camera.release()
			cv2.destroyAllWindows()
			break