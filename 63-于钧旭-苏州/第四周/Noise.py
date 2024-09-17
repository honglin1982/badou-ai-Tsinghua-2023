import cv2 as cv
from skimage import util



img=cv.imread('img/lenna.png')
noise_gs_img=util.random_noise(img,mode='gaussian')
noise_sp_img=util.random_noise(img,mode='s&p')
noise_ps_img=util.random_noise(img,mode='poisson')
cv.imshow('source', img)
cv.imshow('gaussian',noise_gs_img)
cv.imshow('s&p',noise_sp_img)
cv.imshow('poisson',noise_sp_img)
#cv.imwrite('lenna_noise.png',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()