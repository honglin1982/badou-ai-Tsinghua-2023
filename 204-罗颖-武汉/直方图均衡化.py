import cv2

img=cv2.imread('2.bmp',1)
(b,g,r)=cv2.split(img)
bh=cv2.equalizeHist(b)
gh=cv2.equalizeHist(g)
rh=cv2.equalizeHist(r)

result=cv2.merge((bh,gh,rh))
cv2.imshow('123',result)
cv2.waitKey(0)