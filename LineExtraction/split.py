from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
sys.path.append('..')
from util import generate_name

class Split:

    def __init__(self, path, thresh=80, distance=30):
        img = Image.open(path).convert('RGB')
        imgData = np.array(img, dtype='float')

        self.shape = imgData.shape
        print(self.shape)
        assert len(self.shape) == 3
        assert self.shape[2] == 3

        if np.mean(imgData[:]) > 200:
            imgData = np.ones(imgData.shape, dtype=np.uint8) * 255 - imgData

        self.horizontal = np.mean(np.mean(imgData, axis=1), axis=1)

        self.valid = defaultdict(list)
        g = 0
        for i, gray in enumerate(self.horizontal):
            if gray > thresh:
                if len(self.valid[g]) and i > self.valid[g][-1] + distance:
                    g += 1
                self.valid[g].append(i)

        print(self.valid)

        self.computed = []
        index = 0 
        for group in self.valid.values():
            for i in range(index, group[0]):
                self.computed.append(0)
            for i in range(group[0], group[-1]):
                self.computed.append(200)
            index = group[-1]
        for i in range(index, len(self.horizontal)):
            self.computed.append(0)


        plt.plot(self.horizontal)
        plt.plot(self.computed)
        self.visualize = generate_name()
        plt.savefig(self.visualize)

        group = []
        dis = []
        for i in range(int(len(self.valid)/2)):
            group.append((self.valid[2*i][0], self.valid[2*i+1][-1]))
            if len(group):
                dis.append(self.valid[2*i][0] - group[-1][-1])

        print(group, dis)


if __name__ == "__main__":
    s = Split('../data/1/0.png')
    print(s.child)