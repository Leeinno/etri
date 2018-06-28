import cv2
import numpy as np
from skimage.exposure import rescale_intensity

def convolve(image, kernel, pad, stride):
    (iH, iW, ch) = image.shape
    (kH, kW) = kernel.shape[:2]
    (fH, fW) = (int((iH+2*pad-kH)/stride+1),int((iW+2*pad-kW)/stride+1))
    
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((fH,fW,ch), dtype="float32")
    
    for y in np.arange(0, fH):
        for x in np.arange(0, fW):
            for z in np.arange(0, ch):
                result = float(0)
                for i in np.arange(0, kH):
                    for j in np.arange(0, kW):
                        h = y*stride+i
                        w = x*stride+j
                        image_val = image[h, w, z]
                        kernel_val = kernel[i, j]
                        result += image_val*kernel_val
                output[y, x, z] = result
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    
    return output

def run():
    img = cv2.imread('suji.jpg')
    kernel = np.array([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]])
    dst = convolve(img, kernel, 2, 4)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    run()
