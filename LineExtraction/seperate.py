from collections import defaultdict
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from tqdm import tqdm
import cv2

class Color:
    max_distance = 0.2
    black = 50
    white = 235

    def __init__(self, data):
        self.r = data[0]
        self.g = data[1]
        self.b = data[2]

    def __str__(self):
        return f"({self.r},{self.g},{self.b})"

    def __add__(self, other):
        return Color([self.r + other.r, self.g + other.g, self.b + other.b])

    def __sub__(self, other):
        return Color([self.r - other.r, self.g - other.g, self.b - other.b])

    def length(self):
        return abs(self.r) + abs(self.g) + abs(self.g)

    def distance(self, other):
        return (self - other).length()

    def __hash__(self) -> int:
        return hash((self.r, self.g, self.b))

    def __eq__(self, other) -> bool:
        return self.r == other.r and self.g == other.g and self.b == other.b

    @property
    def rgb(self):
        return (self.r, self.g, self.b)
    
    @property
    def amplified_rgb(self):
        def amplify(x):
            t = Color.black * 2
            return t + int(x * (256 - t) / 256)
        return (amplify(self.r), amplify(self.g), amplify(self.b))

    @property
    def hsv(self):
        return rgb_to_hsv(self.rgb)
    
    @property
    def gray(self):
        # Luminosity算法计算灰度值
        return (self.r + self.g + self.b) / 3
        # return self.r * 0.299 + self.g * 0.587 + self.b * 0.114

    def hue_distance(self, other):
        t = abs(self.hsv[0] - other.hsv[0])
        return min(t, 1-t)

    def in_lis(self, lis):
        # 白色
        if self.foreground():
            return Color((255,255,255))

        # 相似度评分
        mark = 10
        color = None

        for c in lis:
            if c == Color((255,255,255)):
                continue
            dis = self.distance(c)
            if dis <= 120:
                hue = self.hue_distance(c)
                if hue <= 0.02:
                    return c
                if hue <= 0.15:
                    m = hue * hue * dis
                    if m < mark:
                        mark = m
                        color = c
        if mark < 0.4:
            return color

        return None


    
    # def in_lis_hue(self, lis):
    #     for c in lis:
    #         if abs(self.hsv[0] - c.hsv[0]) < 0.01:
    #             return c
    #     return None

    def valid(self):
        return self.gray > Color.black


    def foreground(self):
        return self.gray > Color.white or self.r == self.g == self.b and self.gray > Color.black


def hue_split(img_path):
    image = cv2.imread(img_path)
    cv2.imshow('img', image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv[:,:,0])
    (thresh, im_bw) = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('otsu', im_bw)


class Seperator:
    def __init__(self, path, line_num=6, b=50, w=235):
        Color.black = b
        Color.white = w
        print(f'采用的黑白色彩区间：{b}~{w}')
        img = Image.open(path)
        img = img.convert('RGB')   # 修改颜色通道为RGB
        x, y = img.size   # 获得长和宽
        
        # a0m = np.mean(np.mean(img, axis=0), axis=1)
        # print("平均深度", np.mean(a0m))

        dic = defaultdict(int)
        print('提取颜色，按照近似色过滤...')
        for i in tqdm(range(x)):
            for k in range(y):
                c = Color(img.getpixel((i, k)))
                if c.valid():
                    color = c.in_lis(dic.keys()) # or c.in_lis_hue(dic.keys())
                    if color is None:
                        color = c
                    dic[color] += 1


        print("所有颜色数目", len(dic))
        print("最大颜色数目", max(dic.values()))
        print("白色数目", dic[Color((255,255,255))])

        thresh = x / 2
        print('自动调整有效点数阈值:')
        while True:
            colors = [c for c, times in dic.items() if times > thresh and c.valid()]
            print(f"{thresh}({len(colors)})", end="  ")
            if len(colors) <= line_num:
                break
            thresh  += 1

        colors.sort(key = lambda x: dic[x], reverse=True)
        l = len(colors)
        print("总共颜色数：", len(colors))
        for ix, c in enumerate(colors):
            print(f"第{ix+1}组线条，点数为{dic[c]}， RGB颜色为{str(c)}， 色相为{c.hsv[0]}")


        # 生成l个图片存储颜色
        print('生成分离图片...')
        self.color_img = [] 
        for i in range(l):
            self.color_img.append(Image.new('RGB', (x, y), 'black'))

        for i in tqdm(range(x)):
            for k in range(y):
                c = Color(img.getpixel((i, k)))
                if c.valid():
                    color = c.in_lis(colors)  # or c.in_lis_hue(colors)
                    if color and color in colors:
                        index = colors.index(color)
                        self.color_img[index].putpixel((i, k), c.amplified_rgb)
        
        # for i in range(l):
        #     self.color_img[i].show()
    

if __name__ == "__main__":
    # Extractor('data/37- Ce-BDC-1.jpg')
    t = Seperator('out/0.png')
    # print(t.child)

    # print(Color((163,159,160)).distance(Color((129,118,116))))
