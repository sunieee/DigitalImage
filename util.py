import numpy as np
import cv2
import os
import time

cache_folder = os.path.join(os.path.expanduser("~"), 'AppData/Roaming/DigitalImage')
os.makedirs(cache_folder, exist_ok=True)

def generate_name(suffix='.png'):
    if suffix and suffix[0]!= '.':
        suffix = '.' + suffix
    name = time.strftime("%Y%m%d%H%M%S", time.localtime())
    path = f"{cache_folder}/{name}{suffix}"
    i = 1
    while os.path.exists(path):
        path = f"{cache_folder}/{name}_{i}{suffix}"
        i += 1
    print(f'target file saved at {path}')
    if suffix == '':
        os.makedirs(path)
    return path


def expand(arr, width, height):
    # (1680, 1080, 3) 1118 1680     np.array.shape = (h, w, c)
    print(arr.shape, width, height)
    h, w, c = arr.shape
    # 向右，扩展width
    z = np.zeros((h, width - w, c), dtype=arr.dtype)
    output = np.concatenate((arr,z), axis=1)
    # 向下，扩展height
    z = np.zeros((height - h, width, c), dtype=arr.dtype)
    return np.concatenate((output,z), axis=0)


def histeq(imarr):
    hist, bins = np.histogram(imarr, 255)
    cdf = np.cumsum(hist)
    cdf = 255 * (cdf/cdf[-1])
    res = np.interp(imarr.flatten(), bins[:-1], cdf)
    res = res.reshape(imarr.shape)
    return res, hist

def fft2(img):
    return np.fft.fft2(img)

def fftshift(img):
    return np.fft.fftshift(img)

def ifftshift(img):
    return np.fft.ifftshift(img)

def ifft2(img):
    return np.fft.ifft2(img)

# 不调用库
def RobertsOperator(roi):
    operator_first = np.array([[-1,0],[0,1]])
    operator_second = np.array([[0,-1],[1,0]])
    return np.abs(np.sum(roi[1:,1:]*operator_first))+np.abs(np.sum(roi[1:,1:]*operator_second))
 

def RobertsAlogrithm(img):
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,img.shape[0]):
        for j in range(1,img.shape[1]):
            img[i,j] = RobertsOperator(img[i-1:i+2,j-1:j+2])
    return img[1:img.shape[0],1:img.shape[1]]


# 调用库
def Roberts(img):
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    #转uint8
    absX = cv2.convertScaleAbs(x)    
    absY = cv2.convertScaleAbs(y)  
    return cv2.addWeighted(absX,0.5,absY,0.5,0)


def Prewitt(img):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    #转uint8
    absX = cv2.convertScaleAbs(x)    
    absY = cv2.convertScaleAbs(y)  
    return cv2.addWeighted(absX,0.5,absY,0.5,0)


def Sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    
    return cv2.addWeighted(absX,0.5,absY,0.5,0)


def Scharr(img):
    x = cv2.Scharr(img,cv2.CV_16S,1,0)
    y = cv2.Scharr(img,cv2.CV_16S,0,1)
    
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    
    return cv2.addWeighted(absX,0.5,absY,0.5,0)


def Laplacian(img):
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize = 3)
    return cv2.convertScaleAbs(dst)

# 不使用库函数的做法
def LaplaceOperator(roi, operator_type):
    if operator_type == "fourfields":
        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif operator_type == "eightfields":
        laplace_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    else:
        raise ("type Error")
    result = np.abs(np.sum(roi * laplace_operator))
    return result
 
 
def LaplaceAlogrithm(img, operator_type):
    new_image = np.zeros(img.shape)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            new_image[i - 1, j - 1] = LaplaceOperator(img[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(img))
    return new_image.astype(np.uint8)
 
def noisy(noise_typ,img):
    if noise_typ == "gauss":
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(img)
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = img.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = img + img * gauss
        return noisy


def Log(img):
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    # 再通过拉普拉斯算子做边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(dst)


def Canny(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edge_output = cv2.Canny(blurred, 50, 150)
    return cv2.bitwise_and(img, img, mask=edge_output)


def smoothing(img, num, high=True):
    f = fft2(img)
    fshift = fftshift(f)

    rows, cols = img.shape
    crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
    
    if high:
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-num:crow+num, ccol-num:ccol+num] = 0
    else:
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow-num:crow+num, ccol-num:ccol+num] = 1
    fshift = fshift*mask

    ishift = ifftshift(fshift)
    iimg = ifft2(ishift)
    return np.abs(iimg)
