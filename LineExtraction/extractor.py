import enum
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate
import pylab as pl
import sys
sys.path.append('..')

from util import generate_name

class Extractor:
    def __init__(self, path, xs=0, xt=100, ys=0, yt=100):
        img = Image.open(path)
        img = img.convert('RGB')   # 修改颜色通道为RGB
        x, y = img.size   # 获得长和宽

        points = []
        for i in tqdm(range(x)):
            for k in range(y):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50:
                    if not points or points and k >= points[-1][1]:
                        points.append((i, k))
                        break
        # print(points)

        cordinate = []
        for ix, p in enumerate(points):
            xx = round(p[0]/x*(xt-xs) + xs, 3)
            yy = round((1-p[1]/y)*(yt-ys) + ys, 3)
            if ix and xx - cordinate[-1][0] < 1:
                continue
            cordinate.append((xx, yy))


        y_cor = [p[1] for p in points]
        points = []
        for k in tqdm(range(y-1, 0, -1)):
            for i in range(x-1, 0, -1):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50 and k not in y_cor:
                    if not points or points and points[-1][0] >= i:
                        points.append((i, k))
                        break
        # print(points)

        for ix, p in enumerate(points):
            xx = round(p[0]/x*(xt-xs) + xs, 3)
            yy = round((1-p[1]/y)*(yt-ys) + ys, 3)
            if ix and cordinate[-1][1] - yy > 1:
                continue
            cordinate.append((xx, yy))
        
        def get_x(p):
            return (p[0] * 100000 - p[1]) / 100000
        def distance(x, y):
            return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
        
        cordinate.sort(key=get_x)

        self.cordinate = []
        for c in cordinate:
            if self.cordinate and distance(self.cordinate[-1], c) > 10:
                self.cordinate.append(c)
            
        csv = generate_name('.csv')
        print(f"曲线坐标点已保存在：{csv}")
        with open(csv, 'w') as f:
            f.write("x,y\n")
            for p in self.cordinate:
                f.write(f"{p[0]},{p[1]}\n")

        self.x = [get_x(p) for p in cordinate]
        self.y = [p[1] for p in cordinate]

        # print("cor", cordinate)
        # print(self.x)
        # print(self.y)
    
    def fit(self):
        '''拟合'''
        def function(x, a_3, a_2, a_1, a0, a1, a2, a3):
            return a_3 * x ** (-3) + a_2 * x ** (-2) + a_1 * x ** (-1) + a0 \
                + a1 * x + a2 * x ** 2 + a3 * x ** 3

        p_est, _ = curve_fit(function, self.x, self.y)
        print(p_est)
        plt.figure(figsize=(12,9))
        plt.plot(self.x, self.y, "rx")
        plt.plot(self.x, [function(i, *p_est) for i in self.x], "k--")
        plt.show()

    
    def interpolate(self):
        '''差值'''
        pl.figure(figsize=(7, 7))
        pl.plot(self.x, self.y,'ro')
        xnew = np.linspace(min(self.x), max(self.x), 1000)
        for kind in ['nearest', 'zero','linear']: # ,'quadratic', 'cubic'
            #根据kind创建插值对象interp1d
            f = interpolate.interp1d(self.x, self.y, kind = kind)
            ynew = f(xnew)      #计算插值结果
            pl.plot(xnew, ynew, label = str(kind))

        pl.xticks(fontsize=20)
        pl.yticks(fontsize=20)

        pl.legend(loc = 'lower right')
        path = generate_name()
        pl.savefig(path)
        return path


if __name__ == "__main__":
    ex = Extractor('out/21.jpg')
    ex.fit()
