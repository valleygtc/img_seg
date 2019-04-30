import cv2 as cv
img = cv.imread('img/lena_std.tif', cv.IMREAD_GRAYSCALE)
cv.imshow('lena', img)
cv.waitKey(0)
cv.destroyAllWindows()
