import enum
from math import sqrt
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
    def __init__(self, path, xs=0, xt=100, ys=0, yt=100, z=4):
        img = Image.open(path)
        img = img.convert('RGB')   # 修改颜色通道为RGB
        x, y = img.size   # 获得长和宽
        max_dis = sqrt(z * 100) / (xt-xs)

        # 一定包含最上面的点
        points = []
        for k in range(y):
            for i in range(x):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50 and i < x/3:
                    points.append((i, k))
                    break
            if points:
                break
        
        for i in tqdm(range(points[-1][0], x)):
            for k in range(points[-1][1], y):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50:
                    if k - points[-1][1] <= max_dis * y:
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
        for i in range(x-1, 0, -1):
            for k in range(y-1, 0, -1):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50 and k > y/3:
                    points.append((i, k))
                    break
            if points:
                break

        for k in tqdm(range(points[-1][1], 0, -1)):
            for i in range(points[-1][0], 0, -1):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50 and k not in y_cor:
                    if points[-1][0] - i <= max_dis * x:
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
            if not self.cordinate or self.cordinate and z < distance(self.cordinate[-1], c) < 100 * z:
                self.cordinate.append(c)
        
        remove_lis = []
        for c in self.cordinate:
            valid = False
            for cc in self.cordinate:
                if cc != c and distance(c, cc) < 25 * z:
                    valid = True
                    break
            if not valid:
                remove_lis.append(c)
        print(remove_lis)
        for t in remove_lis:
            self.cordinate.remove(t)
            
        csv = generate_name('.csv')
        print(f"曲线坐标点已保存在：{csv}")
        with open(csv, 'w') as f:
            f.write("x,y\n")
            for p in self.cordinate:
                f.write(f"{p[0]},{p[1]}\n")

        self.x = [get_x(p) for p in self.cordinate]
        self.y = [p[1] for p in self.cordinate]

        # print("cor", self.cordinate)
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
