import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt

class sift_feature():
    def __init__(self,siftIniSigma = 0.5,siftMaxIntreStepS = 5,contrThr = 0.04,r  = 10):
        self.siftIniSigma = siftIniSigma
        self.siftMaxIntreStepS = siftMaxIntreStepS
        self.contrThr = contrThr
        self.r = r

    def add_good_ori_feature(self,smoothOriHist,magThr,numBin = 36):
        hist = smoothOriHist
        newRoiHist = np.zeros(shape=[numBin])
        for i in  range(numBin):
            l = (i+numBin-1)%numBin
            r = (i+numBin+1)%numBin
            c = i
            if hist[c]>hist[l] and hist[c]>hist[r] and hist[c]>magThr:
                bin = c + self.interpHistPeak(l = l,c =c ,r = r)
                if bin<0:
                    bin = (abs(numBin+bin))%numBin
                elif bin>=numBin:
                    bin = (bin-numBin)%numBin
                else:
                    bin = bin
                bin = np.round(bin)
                bin = bin.astype(np.int)
                newRoiHist[bin] = hist[c]+self.interpHistPeak(l = hist[l],c = hist[c],r = hist[r])
        return newRoiHist

    def cal_oriHist(self,gaussPyr,keypoint,sigma,numBin = 36):
        rad = np.round(3*1.5*sigma)
        rad = rad.astype(np.int)
        o = keypoint[1]
        s = keypoint[2]
        r = keypoint[3]
        c = keypoint[4]
        L = gaussPyr[o][s,:]
        imWidth = L.shape[1]
        imHeight = L.shape[0]
        hist = np.zeros(shape=[numBin])
        expDen = 2.0*sigma*sigma
        for i in range(-1*rad,rad+1):
            for j in range(-1*rad,rad+1):
                rIdxs = r+i
                cIdxs = c+j
                if rIdxs<=0 or cIdxs<=0 or rIdxs>=imHeight-1 or cIdxs>=imWidth-1:
                    continue
                mag,theta = self.cal_mag_ori(rIdxs,cIdxs,L)
                weight = np.exp(-(i*i+j*j)/expDen)
                bin = np.round(numBin*(theta+np.pi)/(np.pi*2))
                bin = bin.astype(np.int)
                if bin>=numBin:
                    bin = 0
                hist[bin] = hist[bin]+weight*mag
        return hist
		
    def cal_smoothOriHist(self,oriHist):
        numBin = len(oriHist)
        smoothOriHist = np.zeros(shape=[numBin])
        for i in range(numBin):
            smoothOriHist[i] = 0.25*oriHist[(i+numBin-1)%numBin]+\
                                 0.5*oriHist[i]+\
                                 0.25*oriHist[(i+numBin+1)%numBin]
        return smoothOriHist		
		
    def interpHistPeak(self,l,c,r):
        if abs(l-2.0*c+r)<10**(-6):
            return 0
        return 0.5*(l-r)/(l-2.0*c+r)
		
    def cal_featureOris(self,gaussPyr,keypoints,numBin = 36):
        keyptsSigma = sift_featrue.keypoints_sigma(gaussPyr=gaussPyr, keypoints=keypoints)
        featureOris = []
        for i in range(len(keypoints)):
            keypt = keypoints[i]
            sigma = keyptsSigma[i]
            oriHist = self.cal_oriHist(gaussPyr=gaussPyr,keypoint=keypt,sigma=sigma,numBin=numBin)
            smoothOriHist = self.cal_smoothOriHist(oriHist=oriHist)
            maxMag = np.max(smoothOriHist)
            newOriHist = self.add_good_ori_feature(smoothOriHist = smoothOriHist,magThr=maxMag*0.8)
            featureOris.append(newOriHist)
        return np.array(featureOris)

    def cal_mag_ori(self,i,j,L):
        mag = np.sqrt(np.square(L[i+1,j]-L[i-1,j])+\
                  np.square(L[i,j+1]-L[i,j-1]))
        theta = np.arctan((L[i,j+1]-L[i,j-1])/(L[i+1,j]-L[i-1,j]))
        return mag,theta

    def keypoints_sigma(self,gaussPyr,keypoints,sigma = 0.5):
        keyptsSigma = []
        sNum = gaussPyr[0].shape[0]-3
        for keypt in keypoints:
            o = keypt[1]
            s = keypt[2]
            keyptSigma = sigma*pow(2,o+s/sNum)
            keyptsSigma.append(keyptSigma)
        return keyptsSigma

    def Hessian_2D(self,dog,i,j):
        point = dog[i, j]
        dxx = (dog[i, j + 1] + dog[i, j - 1] - 2 * point)
        dyy = (dog[i + 1, j] + dog[i - 1, j] - 2 * point)
        dxy = (dog[i + 1, j + 1] - dog[i + 1, j - 1] - dog[i - 1, j + 1] + dog[i - 1, j - 1]) / 4.0
        return np.array([[dxx, dxy],[dxy, dyy]])

    def edgeLine(self,dogPyr,o,s,i,j):
        dog = dogPyr[o][s,:]
        dD2D = self.Hessian_2D(dog = dog,i = i,j = j)
        dxx = dD2D[0,0]
        dxy = dD2D[0,1]
        dyy = dD2D[1,1]
        tr = dxx + dyy
        det = dxx*dyy - dxy*dxy
        if det<=0:
            return True
        r = self.r
        if tr*tr/det<(r+1.0)*(r+1.0)/r:
            return False
        return True

    def interpExtrem(self,dogPyr,o,s,i,j):
        point = dogPyr[o][s,i,j]
        sNum = dogPyr[o].shape[0]
        iNum = dogPyr[o].shape[1]
        jNum = dogPyr[o].shape[2]
        for step in range(self.siftMaxIntreStepS):
            if s<=0 or i<=0 or j<=0 or s>=sNum-1 or i>=iNum-1 or j>=jNum-1:
                return None
            X = self.offset(dogPyr = dogPyr, o = o, s = s, i = i, j = j)
            xi = X[0];xj = X[1];xs = X[2];
            if abs(xi)<0.5 and abs(xj)<0.5 and abs(xs)<0.5:
                break
            s = int(np.round(xs) + s)
            i = int(np.round(xi) + i)
            j = int(np.round(xj) + j)
            if s<0 or i<0 or j<0 or s>=sNum or i>=iNum or j>=jNum:
                return None
        if step>=self.siftMaxIntreStepS-1:
            return None
        dD = self.deriv_3D(dogPyr,o = o, s = s,i = i,j = j)
        D = point+0.5*np.dot(dD,X)
        if D<self.contrThr/(sNum-2):
            return None
        return [D,o,s,i,j]

    def offset(self,dogPyr,o,s,i,j):
        dD = self.deriv_3D(dogPyr=dogPyr,o=o,s = s, i = i, j = j)
        H = self.hessian_3D(dogPyr=dogPyr,o = o,s = s, i = i, j = j)
        HInv = cv2.invert(src=H,flags=cv2.DECOMP_SVD)[1]
        X = np.dot(-1*HInv,dD)
        return X

    def hessian_3D(self,dogPyr,o,s,i,j):
        dogs = dogPyr[o]
        point = dogs[s,i,j]
        dxx = (dogs[s,i,j+1]+dogs[s,i,j-1]-2*point)
        dyy = (dogs[s,i+1,j]+dogs[s,i-1,j]-2*point)
        dss = (dogs[s+1,i,j]+dogs[s-1,i,j]-2*point)
        dxy = (dogs[s,i+1,j+1]-dogs[s,i+1,j-1]-
               dogs[s,i-1,j+1]+dogs[s,i-1,j-1])/4.0
        dxs = (dogs[s+1,i,j+1]-dogs[s+1,i,j-1]-
               dogs[s-1,i,j+1]+dogs[s-1,i,j-1])/4.0
        dys = (dogs[s+1,i+1,j]-dogs[s+1,i-1,j]-
               dogs[s-1,i+1,j]+dogs[s-1,i-1,j])/4.0
        return np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])

    def deriv_3D(self,dogPyr,o,s,i,j):
        dogs = dogPyr[o]
        point = dogs[s,i,j]
        dx = dogs[s,i,j+1] - point
        dy = dogs[s,i+1,j] - point
        ds = dogs[s+1,i,j] - point
        return np.array([dx,dy,ds])

    def extremePoint(self,dogPyr,o,s,i,j):
        dogs = dogPyr[o]
        point = dogs[s,i,j]
        for sIdxs in range(s-1,s+2):
            for iIdxs in range(i-1,i+2):
                for jIdxs in range(j-1,j+2):
                    if sIdxs == s and iIdxs == i and jIdxs == j:
                        continue
                    if point < 0 and point > dogs[sIdxs,iIdxs,jIdxs]:
                        return False
                    if point >= 0 and point < dogs[sIdxs,iIdxs,jIdxs]:
                        return False
        return True

    def key_point_search(self,dogPyr):
        keypoints = []
        for o in range(len(dogPyr)):
            for s in range(1,dogPyr[o].shape[0]-1):
                for i in range(1,dogPyr[o][s].shape[0]-1):
                    for j in range(1,dogPyr[o][s].shape[1]-1):
                        if abs(dogPyr[o][s,i,j])<0.03:
                            continue
                        if self.extremePoint(dogPyr=dogPyr,o = o,s = s,i = i,j = j):
                            keypoint = self.interpExtrem(dogPyr = dogPyr,o = o, s = s,i = i, j = j)
                            if keypoint is not None:
                                if self.edgeLine(dogPyr = dogPyr,o = keypoint[1],s = keypoint[2],
								                 i = keypoint[3],j = keypoint[4]) == False:
                                    keypoints.append(keypoint)
        return keypoints

    def create_dogPyr(self,gaussPyr):
        dogPyr = []
        for octaves in gaussPyr:
            Dogs = []
            for i in range(1,octaves.shape[0]):
                dog = octaves[i] - octaves[i-1]
                Dogs.append(dog)
            Dogs = np.array(Dogs)
            dogPyr.append(Dogs)
        return dogPyr

    def create_sigmas(self,octaves,S,sigma):
        storeies = S+3
        sigmas = np.zeros(shape=[octaves, storeies])
        k = pow(2, 1 / S)
        for i in range(octaves):
            sigma_i_origin = sigma * pow(2, i)
            for j in range(storeies):
                sigmas[i, j] = sigma_i_origin * pow(k, j)
        return sigmas

    def create_guassPyr(self,initImg,octaves,S = 3,sigma = None):
        storeies = S+3
        # originImg = initImg[0:initImg.shape[0]:2,0:initImg.shape[1]:2]
        # cv2.imshow('img',img)
        if sigma is None:
            sigma = self.siftIniSigma
        guassPyr = []

        sigmas = self.create_sigmas(octaves,S,sigma)
        # print(sigmas)
        for i in range(octaves):
            imgs = []
            originImg = initImg[0:initImg.shape[0]:int(pow(2,i+1)),
                         0:initImg.shape[1]:int(pow(2,i+1))]
            for j in range(storeies):
                if i == 0 and j == 0:
                    imgs.append(originImg)
                    # cv2.imshow('origin',originImg)
                    continue
                else:
                    sig = sigmas[i,j]
                    size = int(sig*6+1)
                    if size%2 == 0:
                        size = size+1
                    img = cv2.GaussianBlur(src=originImg,ksize=(size,size),sigmaX=sig,sigmaY=sig)
                    imgs.append(img)
                    # cv2.imshow('img'+str(i)+str(j),img)
            imgs = np.array(imgs)
            # imgs = imgs.astype(np.float32)
            guassPyr.append(np.array(imgs))
        return guassPyr

    def create_minus_one_img(self,img,sigma = 1.6):
        siftIniSigma = self.siftIniSigma
        sig = np.sqrt(sigma*sigma-siftIniSigma*siftIniSigma*4)
        initImg = cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
        initImg = cv2.resize(src=initImg,dsize=(img.shape[1]*2,img.shape[0]*2),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        initImg = cv2.GaussianBlur(src=initImg,ksize=(0,0),sigmaX=sig,sigmaY=sig)
        return initImg

    def draw_sift_features(self,image,keypoints,featureRois):
        for i in range(len(keypoints)):
            keypt = keypoints[i]
            featureRoi = featureRois[i]
            o = keypt[1]
            x1 = keypt[3]*pow(2,o)
            y1 = keypt[4]*pow(2,o)
            x1 = int(x1)
            y1 = int(y1)
            if x1>=image.shape[1] or y1>=image.shape[0]:
                continue
            cv2.circle(image,center=(y1,x1),radius=2,color=(0,255,0))
            thetaPer = np.pi*2/len(featureRoi)
            for k in range(len(featureRoi)):
                mag = featureRoi[k]
                if mag>10**(-6):
                    x2 = np.round(x1+mag*np.cos(k*thetaPer+thetaPer/2)).astype(np.int)
                    y2 = np.round(y1+mag*np.sin(k*thetaPer+thetaPer/2)).astype(np.int)
                    cv2.arrowedLine(img=image,pt1=(y1,x1),pt2=(y2,x2),color=(255,0,0))
        cv2.imshow('sift',image)
        cv2.waitKey()
        return image

if __name__ == '__main__':
    image = cv2.imread('white_cat.jpg')
    image = image.astype(np.float32)
    sift_featrue = sift_feature()
    initImg = sift_featrue.create_minus_one_img(image)

    gaussPyr = sift_featrue.create_guassPyr(initImg=initImg,octaves=4)
    dogsPyr = sift_featrue.create_dogPyr(gaussPyr=gaussPyr)
    keypoints = sift_featrue.key_point_search(dogPyr=dogsPyr)
    print(len(keypoints))
    print(type(keypoints))
    featureRois = sift_featrue.cal_featureOris(gaussPyr=gaussPyr,keypoints=keypoints)
    print(featureRois.shape)
    siftImage = sift_featrue.draw_sift_features(image=image.astype(np.uint8),keypoints=keypoints,featureRois=featureRois)
    cv2.imwrite('sift_white_cat.jpg',siftImage)