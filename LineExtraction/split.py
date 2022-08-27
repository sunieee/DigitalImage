from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('..')
from util import generate_name
import os



class Split:

    def __init__(self, path, thresh=100, distance=0.02, output=''):
        self.path = path
        img = Image.open(path).convert('RGB')
        imgData = np.array(img, dtype='float')

        self.shape = imgData.shape
        distance = distance * self.shape[0]
        print(self.shape)
        assert len(self.shape) == 3
        assert self.shape[2] == 3

        # 白色则反色
        if np.mean(imgData[:]) > 200:
            imgData = np.ones(imgData.shape, dtype=np.uint8) * 255 - imgData
            imgData = imgData.astype('uint8')
            img = Image.fromarray(imgData)

        self.horizontal = np.mean(np.mean(imgData, axis=1), axis=1)

        valid = defaultdict(list)
        g = 0
        for i, gray in enumerate(self.horizontal):
            if gray > thresh:
                if len(valid[g]) and i > valid[g][-1] + distance:
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
        for i in range(int(len(valid)/2)):
            if len(group):
                dis.append(valid[2*i][0] - group[-1][-1])
            group.append((valid[2*i][0], valid[2*i+1][-1]))

        print(group, dis)

        self.output = output if output else generate_name('')
        ix = 0
        for name in os.listdir(self.output):
            try:
                ix = max(ix, int(name.split('.')[0]))
            except:
                pass

        for i, g in enumerate(group):
            ix += 1
            # print(g, dis[i])
            A = g[0] - np.mean(dis)/2 if i == 0 else g[0] - dis[i-1] / 2
            B = g[1] + np.mean(dis)/2 if i >= len(dis) else g[1] + dis[i] / 2
            img.crop((0, A, self.shape[1], B)).save(f'{self.output}/{ix}.png')
            

if __name__ == "__main__":
    s = Split('../data/1/0.png')