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

        self.points = []
        for i in tqdm(range(x)):
            for k in range(y):
                pixel = img.getpixel((i, k))
                if np.sum(pixel) > 50:
                    if self.points:
                        last = self.points[-1]
                        offset = i - last[0]
                        # 单调递减
                        if k - last[1] >= 0:
                            self.points.append((i, k))
                            break
                    else:
                        self.points.append((i, k))
                        break

        # print("points", self.points)
        # self.color_img = Image.new('RGB', (x, y), 'black')
        # for p in self.points:
        #     self.color_img.putpixel(p, img.getpixel(p))
        # self.color_img.show()

        self.cordinate = []
        for p in self.points:
            self.cordinate.append((p[0]/x*(xt-xs) + xs, (1-p[1]/y)*(yt-ys) + ys))

        print("曲线坐标点如下，格式为(x, y)") 
        print(self.cordinate)

        self.x = [p[0] for p in self.cordinate]
        self.y = [p[1] for p in self.cordinate]

    
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
        pl.figure(figsize=(10,8))
        pl.plot(self.x, self.y,'ro')
        xnew = np.linspace(min(self.x), max(self.x), 1000)
        for kind in ['nearest', 'zero','linear','quadratic', 'cubic']:
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
