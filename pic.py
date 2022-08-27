from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QLabel, QWidget, QInputDialog
from Ui_main import QtCore
import os
import numpy as np
from util import *
from PIL import Image
from helper import *


font1 = QFont()
font1.setFamily(u"Times New Roman")
font1.setPointSize(12)
base_folder = os.path.join(os.path.expanduser("~"), 'Desktop')
if os.path.exists(base_folder + '/file'):
    base_folder += '/file'
if os.path.exists(base_folder + '/photo/src/LineExtraction/data'):
    base_folder += '/photo/src/LineExtraction/data'


class Pic:
    MAX_WIDTH = 700
    

    def __init__(self, path, ui) -> None:
        print(f'opening {path}')
        assert os.path.exists(path)
        self.src = path
        self.ui = ui
        self.name = path.split('/')[-1]
        self.lis = None
        self.open()
        self.open_pix()
        self.dic = {0: (self.pix, self.img, self.w, self.h)}
    
    def open_pix(self):
        # if not self.img:
        #     self.ui.warning(f'打开文件{self.src}失败')

        if self.img.width > self.img.height:
            self.w = Pic.MAX_WIDTH
            self.h = int(self.img.height / self.img.width * Pic.MAX_WIDTH)
        else:
            self.w = int(self.img.width / self.img.height * Pic.MAX_WIDTH)
            self.h = Pic.MAX_WIDTH

        self.pix = QPixmap(self.path).scaled(self.w, self.h)
        print(self.pix)
        assert not self.pix.isNull()


    def open(self):
        if self.src.endswith('.raw') or self.src.endswith('.data'):
            print('You are opening a raw file!')
            imgData = np.fromfile(self.src, dtype=np.uint8)
            length = imgData.shape[0]
            h, ok = QInputDialog.getInt(self.ui, '图片尺寸', '请输入图片height')
            if not h or not ok:
                h = 512
            if length % h != 0:
                return False

            c, ok = QInputDialog.getInt(self.ui, '图片尺寸', '请输入图片channel数')
            if not h or not ok:
                h = 3
            if (length / h) % c != 0:
                return False
            imgData = imgData.reshape(h, int(length/h/c), c)
            # imgData = imgData.astype('uint8')
            self.path = generate_name()
            Image.fromarray(imgData).save(self.path)
        elif self.src.endswith('.pdf'):
            print('You are opening a pdf file! Converting to images!')
            h, ok = QInputDialog.getInt(self.ui, '正在打开pdf', '请输入pdf转图片dpi（默认144，即两倍大小')
            if not h or not ok:
                h = 144
            self.folder = pdf2pic(self.src, dpi=h)
            self.lis = get_folder(self.folder)
            self.index = 0
            self.path = self.lis[0]
        else:
            self.path = self.src
        self.img = Image.open(self.path)
        

    def label_text(self):
        path = ''
        if self.lis:
            path += f'\n{self.index + 1}/{len(self.lis)}'
        return f'{self.img.width} × {self.img.height}\n{self.name}{path}'
    
    def create_widget(self):
        self.widget = QWidget()
        self.label = QLabel(self.widget)
        self.label.setObjectName(self.name)
        # self.label.setGeometry(QRect(0, 0, self.w, self.h))
        # self.label.setFixedSize(self.w, self.h)
        self.label.setPixmap(self.pix)
        # self.label.setScaledContents(True)

        self.text = QLabel(self.widget)
        self.text.setGeometry(QtCore.QRect(550, 630, 150, 70))
        self.text.setFont(font1)
        self.text.setText(self.label_text())
        # self.widget.
        return self.widget

    def to_resize(self):
        self.img = self.img.resize((self.w, self.h))
        self.text.setText(self.label_text())
    
    def to_grey(self):
        self.img = self.img.convert('L')
        self.path = generate_name()
        self.img.save(self.path)
        self.pix = QPixmap(self.path).scaled(self.w, self.h)
        self.label.setPixmap(self.pix)

    def renew_label(self):
        self.path = self.lis[self.index]
        if self.index in self.dic:
            # 直接加载缓存
            self.pix, self.img, self.w, self.h = self.dic[self.index]
        else:
            self.img = Image.open(self.path)
            self.open_pix()
            self.dic[self.index] = (self.pix, self.img, self.w, self.h)

        self.label.setPixmap(self.pix)
        self.text.setText(self.label_text())
        

    def left(self):
        if self.lis:
            self.index -= 1
            if self.index < 0:
                self.index = len(self.lis) - 1
            self.renew_label()
            
    def right(self):
        if self.lis:
            self.index = (self.index + 1) % len(self.lis)
            self.renew_label()
    
    def base(self, func_name: str, *args):
        direct = {
            'grey': lambda img: img.convert('L'),
            'resize': lambda img: img.resize((self.w, self.h)),
            'rotate': lambda img: img.rotate(args[0]),
            'scale': lambda img: img.resize((int(img.width * args[0]), int(img.height * args[0]))),
            'enlarge': lambda img: img.resize((int(img.width * 2), int(img.height * 2))),
            'shrink': lambda img: img.resize((int(img.width / 2), int(img.height / 2))),
        }
        use_cv = {
            'Roberts': lambda f: Roberts(f),
            'Sobel': lambda f: Sobel(f),
            'Prewitt': lambda f: Prewitt(f),
            'Scharr': lambda f: Scharr(f),
            'Laplacian': lambda f: Laplacian(f),
            'Log': lambda f: Log(f),
            'Canny': lambda f: Canny(f),
        }
        indirect = {
            'reverse': lambda data: np.ones(data.shape, dtype=np.uint8) * 255 - data,
            'histogram': lambda data: histeq(data)[0],
            'high': lambda data: smoothing(np.mean(data, axis=2), args[0]),
            'low': lambda data: smoothing(np.mean(data, axis=2), args[0], False),
            'colorHigh': lambda data: np.dstack([smoothing(data[:,:,i], args[0]) for i in range(3)]),
            'colorLow': lambda data: np.dstack([smoothing(data[:,:,i], args[0]) for i in range(3)])
        }
        if func_name in direct:
            func = lambda f: direct[func_name](Image.open(f))
        elif func_name in indirect:
            func = lambda f: Image.fromarray(
                indirect[func_name](
                    np.array(Image.open(f))
                ).astype('uint8')
            )
        elif func_name in use_cv:
            func = lambda f: Image.fromarray(
                use_cv[func_name](cv2.imread(f, 0))
            )
        else:
            self.ui.warning(f"No such function: {func_name}")
            return


        if self.lis:
            folder = generate_name('')
            for f in tqdm(os.listdir(self.folder)):
                func(os.path.join(self.folder, f)).save(os.path.join(folder, f))
            path = pic2pdf(folder)
        else:
            suffix = self.name.split('.')[-1]
            path = generate_name(suffix)
            func(self.path).save(path)
        return path


    @property
    def imgData(self):
        try:
            return np.array(self.img, dtype='float')
        except:
            return np.array(self.img)

    def save(self, path=None):
        if path is None:
            suffix = self.name.split('.')[-1]
            path = generate_name(suffix)
        if path.endswith('.raw') or path.endswith('.data'):
            self.imgData.tofile(path)
        elif path.endswith('.pdf'):
            print('You are saving a pdf!')
            pic2pdf(self.folder, path=path)
        else:
            if self.imgData.dtype == np.complex128:
                self.ui.warning('You are saving a complex data. File types must be raw/data')
            self.img.save(path)
        

if __name__ == "__main__":
    p = Pic('data/1/0.png', None) #.grey()
    p.base('grey')