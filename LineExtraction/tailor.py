from PIL import Image
import numpy as np

class Tailor:
    def __init__(self, path):
        img = Image.open(path)
        imgData = np.array(img, dtype='float')

        self.shape = imgData.shape
        print(self.shape)
        assert len(self.shape) == 3
        assert self.shape[2] == 3

        self.a0m = np.mean(np.mean(imgData, axis=0), axis=1)
        self.a1m = np.mean(np.mean(imgData, axis=1), axis=1)
        gray = np.mean(imgData, axis=2)
        self.mid = np.mean(self.a0m)
        # print(a0m.shape, a0m)
        # print(a1m.shape, a1m)

        self.thresh = 0.5
        self.corner_pixels = 10

        xAxis, yAxis = self.get_axis()
        x = int((xAxis[0] + xAxis[1]) / 2)
        y = int((yAxis[0] + yAxis[1]) / 2)
        # print(yAxis, y, gray[:, y])
        # print(xAxis, x, gray[x, :])

        yFrag = self.get_fragment(gray[:, y])
        xFrag = self.get_fragment(gray[x, :])

        print("坐标轴位置：", yAxis, xAxis)
        print("坐标轴线段：", yFrag, xFrag)

        # print(len(self.a0m), len(self.a1m))

        ys = yFrag[0] + 1
        yt = xAxis[0] - 1
        xs = yAxis[1] + 1
        xt = xFrag[1] - 1
        # 避免黑色方框！
        black_thresh = 100
        while self.a0m[xs] < black_thresh:
            xs += 1
        while self.a0m[xt] < black_thresh:
            xt -= 1
        while self.a1m[ys] < black_thresh:
            ys += 1
        while self.a1m[yt] < black_thresh:
            yt -= 1

        print(f'分割：({ys},{yt}) * ({xs},{xt})')
        self.tailor = imgData[ys:yt, xs:xt]


    def get_axis(self):
        while True:
            d0 = self.get_dark_line(self.a0m)
            d1 = self.get_dark_line(self.a1m)

            print(d0, d1, self.thresh)
            if len(d0) and len(d1):
                yAxis = None
                xAxis = None
                for line in d0:
                    if line[0] < self.shape[0] / 2:
                        yAxis = line
                for line in d1:
                    if line[0] > self.shape[1] / 2:
                        xAxis = line
                if xAxis and yAxis:
                    return xAxis, yAxis
            self.thresh *= 1.1
            
            if self.thresh > 1:
                raise Exception("can not found axis")


    def get_dark_line(self, am):
        '''顺序检测线条，不限线条粗细
        第一个左侧为空，最后一个右侧为空的不要
        '''
        dark_index = []
        for i in range(self.corner_pixels, len(am) - self.corner_pixels):
            light = am[i]
            if light < self.mid * self.thresh:
                dark_index.append(i)
        d = self.get_dark_frag(dark_index)

        if len(d):
            valid = False
            for i in range(self.corner_pixels, d[0][0] - self.corner_pixels):
                if am[i] < 250:
                    valid = True
            if not valid:
                d.remove(d[0])

        if len(d):
            valid = False
            for i in range(d[-1][1] + self.corner_pixels, len(am) - self.corner_pixels):
                if am[i] < 250:
                    valid = True
            if not valid:
                d.remove(d[-1])
        
        return d


    def get_dark_frag(self, dark_index):
        line = []

        def add_line(i):
            j = i + 1
            while j in dark_index or j + 1 in dark_index or j + 2 in dark_index or j + 3 in dark_index \
                or j + 4 in dark_index or j + 5 in dark_index:
                j += 1
            line.append((i, j-1))

        for i in dark_index:
            if  i-1 not in dark_index:
                add_line(i)
        return line


    def get_fragment(self, pixels):
        '''提取线段，返回起终点
        前提：长度超过一半，不是在边缘
        '''
        dark_index = []
        for i in range(self.corner_pixels, len(pixels) - self.corner_pixels):
            if pixels[i] < self.mid * self.thresh:
                dark_index.append(i)

        dark_line = self.get_dark_frag(dark_index)
        print("该线段dark_line:", dark_line)
        for line in dark_line:
            if line[1] - line[0] > len(pixels) / 2:
                return line


if __name__ == "__main__":
    # Extractor('data/37- Ce-BDC-1.jpg')

    Tailor('data/69-UiO-66（Ce）.jpg')