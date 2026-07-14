from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('..')
from util import generate_name
import os



class Split:

    def __init__(self, path, thresh=60, distance=0.015, output=''):
        self.path = path
        img = Image.open(path).convert('RGB')
        imgData = np.array(img, dtype='float')

        self.shape = imgData.shape
        height, width, _ = imgData.shape
        distance = distance * self.shape[0]
        print(self.shape)
        assert len(self.shape) == 3
        assert self.shape[2] == 3

        # 亮色则反色
        if np.mean(imgData[:]) > 200:
            imgData = np.ones(imgData.shape, dtype=np.uint8) * 255 - imgData
            imgData = imgData.astype('uint8')
            
            # Calculate the most frequent color (background color)
            unique_colors, counts = np.unique(imgData.reshape(-1, imgData.shape[2]), axis=0, return_counts=True)
            background_color = unique_colors[np.argmax(counts)]
            print("background_color:", background_color)

            # Replace the background color with black
            mask = np.all(imgData == background_color, axis=-1)
            imgData[mask] = [0, 0, 0]
            print("Mask applied to background:", np.sum(mask))

            imgData = imgData.astype('uint8')
            img = Image.fromarray(imgData)

        left = int(width * 0.1)  # 左边界
        right = int(width * 0.9)  # 右边界
        cropped_imgData = imgData[:, left:right]
        self.horizontal = np.mean(np.mean(cropped_imgData, axis=1), axis=1)

        valid = defaultdict(list)
        g = 0
        for i, gray in enumerate(self.horizontal):
            if gray > thresh:   
                # 保证每个group至少有5个点，否则继续添加
                # 保证每个group内部的点的距离大于5，否则继续添加
                # 两个group之间的距离大于distance   
                if len(valid[g]) >= 5 \
                    and valid[g][-1] - valid[g][0] > distance \
                    and i > valid[g][-1] + distance: 
                    g += 1
                valid[g].append(i)
        del_keys = []
        for k, v in valid.items():
            if len(v) < 5:
                del_keys.append(k)
        for k in del_keys:
            del valid[k]

        print("valid:", valid)
        valid = list(valid.values())       

        self.computed = []
        index = 0 
        for group in valid:
            for i in range(index, group[0]):
                self.computed.append(0)
            for i in range(group[0], group[-1]):
                self.computed.append(200)
            index = group[-1]
        for i in range(index, len(self.horizontal)):
            self.computed.append(0)

        plt.figure(figsize=(8, 6))
        plt.plot(self.horizontal)
        plt.plot(self.computed)
        self.visualize = generate_name()
        plt.savefig(self.visualize)

        group = []
        dis = []
        i = 0
        while i < len(valid):
            if len(group):
                dis.append(valid[i][0] - group[-1][-1])

            if i + 1 == len(valid): # 只剩最后一个group
                group.append((valid[i][0], valid[i][-1]))
                break

            if len(group) and valid[i+1][0] - valid[i][0] > distance * 10:   # 两个group之间的距离太大，需要分开保存
                group.append((valid[i][0], valid[i][-1]))
                i += 1
                continue
            
            group.append((valid[i][0], valid[i+1][-1]))
            i += 2

        print(group, dis)
        self.output = output if output else generate_name('')
        ix = 0
        for name in os.listdir(self.output):
            try:
                ix = max(ix, int(name.split('.')[0]))
            except:
                pass

        if len(dis) == 0:
            print('No need to split')
            img.save(f'{self.output}/{ix}.png')
            return

        for i, g in enumerate(group):
            ix += 1
            # print(g, dis[i])
            A = g[0] - np.mean(dis)/2 if i == 0 else g[0] - dis[i-1] / 2
            B = g[1] + np.mean(dis)/2 if i >= len(dis) else g[1] + dis[i] / 2
            img.crop((0, A, self.shape[1], B)).save(f'{self.output}/{ix}.png')
            

if __name__ == "__main__":
    s = Split('../data/1/0.png')