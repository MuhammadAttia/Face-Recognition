import cv2,sys,glob,math,string,os.path
from math import exp,sqrt
from numpy.linalg import inv
from numpy.linalg import det
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import FastICA



f1 = open('pca_result_glass.txt','a')
f2 = open('lda_result_glass.txt','a')
image_resolution = 92 * 112
eigenface  = 500       # n component
training = 410       # total number of images
kne = 30
print 'Component = ',str(eigenface)
# f.write('n='+str(eigenface)+', dataset='+str(training))

folders   = glob.glob('dataset/*') # loading dataset
testFaces = glob.glob('test/*')  # loading test image (one image)

# get the folder name (also the person name)
def getName(filename):
    fullname = string.split(filename, '/')
    return fullname[1].replace("s", "")

# convert images for process
def preprocess(filename):
    imageColor = cv2.imread(filename)
    imageGray  = cv2.cvtColor(imageColor, cv2.cv.CV_RGB2GRAY)
    imageGray  = cv2.equalizeHist(imageGray)
    return imageGray.flat

# find face function
def find():    
    X = np.zeros([training, image_resolution], dtype='int8')
    y = []

    # read the training data
    z = 0
    for x, folder in enumerate(folders):
        trainFaces = glob.glob(folder + '/*')
        for i, face in enumerate(trainFaces):
            X[z,:] = preprocess(face)
            # print X
            y.append(getName(face))
            z = z + 1

    # component analysis (3 ways)
    pca = PCA(n_components=eigenface, whiten=True).fit(X)
    lda = LinearDiscriminantAnalysis(n_components=eigenface).fit(X,y)
    # ica = FastICA(n_components=eigenface,whiten=True).fit(X)
    X_pca = pca.transform(X)
    X_lda = lda.transform(X)
    # X_ica = ica.transform(X)

    # images X
    X = np.zeros([len(testFaces), image_resolution], dtype='int8')

    # preprocessing
    for i,face in enumerate(testFaces):
        X[i,:] = preprocess(face)
        # print X
    test1 = pca.transform(X)
    test2 = lda.transform(X)
    # test3 = ica.transform(X)

    neigh = KNeighborsClassifier(n_neighbors=kne)
    gnb = GaussianNB()
    clf1 = svm.SVC()
    clf2 = svm.LinearSVC()
    # clf3 = svm.NuSVC()
    
    # PCA
    trainingimage,label = [],[]
    for i, testPca in enumerate(X_pca):
        trainingimage.append(testPca)
        label.append(i/10)
    # different classifer
    y1 = gnb.fit(trainingimage,label).predict(test1)
    y2 = neigh.fit(trainingimage,label).predict(test1)
    y3 = clf1.fit(trainingimage,label).predict(test1)
    y4 = clf2.fit(trainingimage,label).predict(test1)
    # y5 = clf3.fit(trainingimage,label).predict(test1)
    print y1,y2,y3,y4

    # LDA
    trainingimage,label = [],[]
    for i, testLda in enumerate(X_lda):
        trainingimage.append(testLda)
        label.append(i/10)
    # different classifer
    y1l = gnb.fit(trainingimage,label).predict(test2)
    y2l = neigh.fit(trainingimage,label).predict(test2)
    y3l = clf1.fit(trainingimage,label).predict(test2)
    y4l = clf2.fit(trainingimage,label).predict(test2)
    # y5l = clf3.fit(trainingimage,label).predict(test2)
    print y1l,y2l,y3l,y4l
    f1.write(str(y1[0])+','+str(y2[0])+','+str(y3[0])+','+str(y4[0])+'\n')
    f2.write(str(y1l[0])+','+str(y2l[0])+','+str(y3l[0])+','+str(y4l[0])+'\n')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
cam = cv2.VideoCapture(0)
cam_image = "screenshot.png"

# crop the test image
def crop(faces):
	img = cv2.imread(cam_image)
	for (x, y, w, h) in faces:
		# print x, y, w, h
		cropImg = img[y: y + h, x: x + w] # Crop from x, y, w, h 
		cropImg = cv2.resize(cropImg,(92,112))
         	cv2.imshow("cropped", cropImg)
        	cv2.imwrite('test/face.png', cropImg)

def run():
    stop = False
    if cam.isOpened(): # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False
        print "Camera is not found"

    while rval or not stop:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
        gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	    )

	# draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        key = cv2.waitKey(20)
        cv2.imshow('Face Detection', frame)
        if key & 0xFF in [ord('S'), ord('s')]: # Screenshot
            cv2.imwrite(cam_image, gray)
	    crop(faces)
	    find()
            # print "Screenshot taken!"
        elif key & 0xFF in [ord('C'), ord('c')]: # Cropped
            crop(faces)
            print "Cropped"
        elif key & 0xFF in [ord('Q'), ord('q')]: # Exit
            print "Exiting"
            stop = True
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()

f1.close()
f2.close()
