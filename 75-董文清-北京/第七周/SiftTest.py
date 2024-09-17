import numpy as np
import cv2

def showKeyPoints(src):

    img, imgGray = cv2.imread(src), cv2.imread(src, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(imgGray, None)
    image = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                              color=(51, 163, 236))
    cv2.imshow("SIFT KeyPoints", image)
    k = cv2.waitKey(0)
    if k == 27 :
        cv2.destroyAllWindows()

def KeyPointsMatch(s1, s2):

    img1 = cv2.imread(s1)
    img2 = cv2.imread(s2)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    goodMatch = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            goodMatch.append(m)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    image[:h1,:w1] = img1
    image[:h2, w1:w1 + w2] = img2
    sourcePoints = [kp.queryIdx for kp in goodMatch]
    destPoints = [kp.trainIdx for kp in goodMatch]
    sourcePoints = np.int32([kp1[p].pt for p in sourcePoints])
    destPoints = np.int32([kp2[p].pt for p in destPoints]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(sourcePoints, destPoints):
        cv2.line(image, (x1, y1), (x2, y2), color=(0,0,255))
    cv2.imshow("SIFT match", image)
    k = cv2.waitKey(0)
    if k == 27 :
        cv2.destroyAllWindows()

if __name__ == '__main__':

    #showKeyPoints('.\source\lenna.png')
    KeyPointsMatch('iphone1.png', 'iphone2.png')