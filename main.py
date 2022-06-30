from re import L
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog, QLabel, QWidget
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QInputDialog
from cv2 import drawContours
from Ui_main import Ui_MainWindow, QtCore
import os
from PIL import Image, ImageQt
import threading
import numpy as np
from util import *
import cv2


font1 = QFont()
font1.setFamily(u"Times New Roman")
font1.setPointSize(12)
base_folder = os.path.join(os.path.expanduser("~"), 'Desktop')
if os.path.exists(base_folder + '/file'):
    base_folder += '/file'


class Pic:
    def __init__(self, path, ui) -> None:
        print(f'opening {path}')
        assert os.path.exists(path)
        self.path = path
        self.ui = ui
        self.name = path.split('/')[-1]
        self.open()
            # self.pix = ImageQt.toqpixmap(self.img)
        if not self.img:
            self.ui.warning(f'打开文件{path}失败')
        self.pix = QPixmap(self.path).scaled(self.w, self.h)
        print(self.pix)
        assert not self.pix.isNull()


    def open(self):
        if self.path.endswith('raw') or self.path.endswith('data'):
            imgData = np.fromfile(self.path, dtype=np.uint8)
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
            Image.fromarray(imgData).save('result.png')
            self.path = 'result.png'
        self.img = Image.open(self.path)

        if self.img.width > self.img.height:
            self.w = 600
            self.h = int(self.img.height / self.img.width * 600)
        else:
            self.w = int(self.img.width / self.img.height * 600)
            self.h = 600
    
    def create_widget(self):
        self.widget = QWidget()
        self.label = QLabel(self.widget)
        self.label.setObjectName(self.name)
        # self.label.setGeometry(QRect(0, 0, self.w, self.h))
        # self.label.setFixedSize(self.w, self.h)
        self.label.setPixmap(self.pix)
        # self.label.setScaledContents(True)

        self.text = QLabel(self.widget)
        self.text.setGeometry(QtCore.QRect(500, 0, 100, 30))
        self.text.setFont(font1)
        self.text.setText(f'{self.img.width} × {self.img.height}')
        
        # self.widget.
        return self.widget

    def resize(self):
        self.img = self.img.resize((self.w, self.h))
        print(f'{self.img.width} × {self.img.height}')
        self.text.setText(f'{self.img.width} × {self.img.height}')

    
    def grey(self):
        self.img = self.img.convert('L')
        self.path = 'result.png'
        self.img.save(self.path)
        self.pix = QPixmap(self.path).scaled(self.w, self.h)
        self.label.setPixmap(self.pix)


    @property
    def imgData(self):
        try:
            return np.array(self.img, dtype='float')
        except:
            return np.array(self.img)

    def save(self, path):
        if path.endswith('raw') or path.endswith('data'):
            # self.img.save('result.png')
            # t = cv2.imread('result.png', cv2.IMREAD_UNCHANGED)
            # t.tofile(path)
            self.imgData.tofile(path)
        else:
            if self.imgData.dtype == np.complex128:
                self.ui.warning('You are saving a complex data. File types must be raw/data')
            self.img.save(path)
        

# 注意 这里选择的父类 要和你UI文件窗体一样的类型
# 主窗口是 QMainWindow， 表单是 QWidget， 对话框是 QDialog
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = Ui_MainWindow()
        # 初始化界面
        self.ui.setupUi(self)

        # 使用界面定义的控件，也是从ui里面访问
        # self.ui.webview.load('http://www.baidu.com')
        self.opened = []
        self.output = []
        self.bind()

    def bind(self):
        # self.ui.open.activate.connect(self.open)
        for t in self.ui.__dict__.keys():
            if t in dir(self):
                # def wrap():
                #     threading.Thread(target=getattr(self, t), args=(self)).start()
                self.ui.__dict__[t].triggered.connect(getattr(self, t))


    def open(self):
        filePaths, _  = QFileDialog.getOpenFileNames(
            self,             # 父窗口对象
            "选择你要上传的图片", # 标题
            base_folder,        # 起始目录
            "图片类型 (*.png *.jpg *.bmp *.raw *.data);;jpg类型 (*.jpg);;png类型 (*.png);;bmp类型 (*.bmp);;raw类型 (*.raw *.data)" # 选择类型过滤项，过滤内容在括号中
        )
        for p in filePaths:
            self.openTarget(p)

    def openTarget(self, path):
        ix = len(self.opened) + 1
        pic = Pic(path, self)
        self.opened.append(pic)
        print(pic)

        self.ui.tabWidget.addTab(pic.create_widget(), str(ix))
        self.ui.tabWidget.setCurrentIndex(ix)
        print('finish')
        # Pyside2用不了！！！

    def warning(self, text):
        QMessageBox.warning(self, "Warning", text, QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)


    def infomation(self, text):
        QMessageBox.information(self, "Infomation", text, QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)


    def save_pic(self, t):
        filePath, _  = QFileDialog.getSaveFileName(
            self,             # 父窗口对象
            "保存文件", # 标题
            base_folder + '/output',        # 起始目录
            "jpg类型 (*.jpg);;png类型 (*.png);;bmp类型 (*.bmp);;raw类型 (*.raw *.data)" # 选择类型过滤项，过滤内容在括号中
        )
        if filePath:
            print(filePath)
            t.save(filePath)


    def save(self):
        t = self.get_current_pic()
        if not t:
            self.warning('输入窗口尚未打开任何图片！')
            return
        self.save_pic(t)


    def saveR(self):
        t = self.get_current_picR()
        if not t:
            self.warning('输出窗口尚未打开任何图片！')
            return
        self.save_pic(t)


    def get_current_pic(self):
        # print(self.opened)
        w = self.ui.tabWidget.currentWidget()
        for pic in self.opened:
            if pic.widget == w:
                return pic
        self.warning('输入窗口尚未打开任何图片！')


    def get_current_picR(self):
        # print(self.output)
        w = self.ui.tabWidgetR.currentWidget()
        for pic in self.output:
            if pic.widget == w:
                return pic
        self.warning('输出窗口尚未打开任何图片！')


    def close(self):
        self.opened.remove(self.get_current_pic())
        t = self.ui.tabWidget.currentIndex()
        self.ui.tabWidget.removeTab(t)
        

    def closeR(self):
        self.output.remove(self.get_current_picR())
        t = self.ui.tabWidgetR.currentIndex()
        self.ui.tabWidgetR.removeTab(t)
        

    def resize(self):
        self.get_current_pic().resize()

    def grey(self):
        self.get_current_pic().grey()

    def resizeAll(self):
        for pic in self.opened:
            pic.resize()
    

    def add(self):
        if len(self.opened) == 0:
            return

        w = 0
        h = 0
        for pic in self.opened:
            w = max(w, pic.img.width)
            h = max(h, pic.img.height)

        result = np.zeros((h, w, 3), np.float64)
        for pic in self.opened:
            t = np.array(pic.img).astype('float')
            # print(t)
            # print(expand(t, w, h).shape)
            result += expand(t, w, h)

        result = result / len(self.opened)
        result = result.astype('uint8')
        resultImage = Image.fromarray(result)
        resultImage.save('result.png')
        self.show_output()
    
    def show_output(self):
        output = Pic('result.png', self)
        ix = len(self.output) + 1
        self.output.append(output)
        self.ui.tabWidgetR.addTab(output.create_widget(), str(ix))
        self.ui.tabWidgetR.setCurrentIndex(ix)


    def reverse(self):
        t = self.get_current_pic().imgData
        result = np.ones(t.shape, dtype=np.uint8) * 255 - t
        self.show_array(result)


    def rotate(self):
        num, ok = QInputDialog.getInt(self, 'Rotate', '将当前图片逆时针旋转 X°')
        if ok and num:
            t = self.get_current_pic().img
            t.rotate(num).save('result.png')
            self.show_output()


    def scale(self):
        num, ok = QInputDialog.getDouble(self, 'Scale', '将当前图片放大/缩小到 X 倍')
        if ok and num:
            t = self.get_current_pic().img
            t.resize((int(t.width * num), int(t.height * num))).save('result.png')
            self.show_output()


    def enlarge(self):
        t = self.get_current_pic().img
        t.resize((int(t.width * 2), int(t.height * 2))).save('result.png')
        self.show_output()


    def shrink(self):
        t = self.get_current_pic().img
        t.resize((int(t.width / 2), int(t.height / 2))).save('result.png')
        self.show_output()


    def show_array(self, result):
        result = result.astype('uint8')
        resultImage = Image.fromarray(result)
        resultImage.save('result.png')
        self.show_output()


    def histogram(self):
        result, _ = histeq(self.get_current_pic().imgData)
        self.show_array(result)


    def FFT(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)

        f = fft2(img)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift))
        # print(magnitude_spectrum.max())
        magnitude_spectrum = 255 / magnitude_spectrum.max() * magnitude_spectrum
        self.show_array(fshift)
        self.show_array(magnitude_spectrum)

        ishift = ifftshift(fshift)
        iimg = ifft2(ishift)
        iimg = np.abs(iimg)
        self.show_array(iimg)


    def high(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)

        f = fft2(img)
        fshift = fftshift(f)

        rows, cols = img.shape
        crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
        mask = np.ones((rows, cols), np.uint8)
        num, _ = QInputDialog.getInt(self, 'high', '滤波中心半径（默认50）')
        if not num:
            num = 50
        mask[crow-num:crow+num, ccol-num:ccol+num] = 0
        fshift = fshift*mask

        ishift = ifftshift(fshift)
        iimg = ifft2(ishift)
        iimg = np.abs(iimg)
        self.show_array(iimg)

    def low(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)

        f = fft2(img)
        fshift = fftshift(f)

        rows, cols = img.shape
        crow, ccol = int(rows/2) , int(cols/2)     # 中心位置
        mask = np.zeros((rows, cols), np.uint8)
        num, _ = QInputDialog.getInt(self, 'low', '滤波中心半径（默认50）')
        if not num:
            num = 50
        mask[crow-num:crow+num, ccol-num:ccol+num] = 1
        fshift = fshift*mask

        ishift = ifftshift(fshift)
        iimg = ifft2(ishift)
        iimg = np.abs(iimg)
        self.show_array(iimg)


    def reconstruct(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)

        # img = cv2.GaussianBlur(img, (15, 15), 0)
        # _, img = cv2.threshold(img, 30, 255, 0)
        # 边缘检测
        img = Log(img)
        kernel = np.ones((5,5),np.uint8) 
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        self.show_array(img)


        # 选择最大轮廓，即目标轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # self.show_array(contours)
        # out = cv2.drawContours(img, contours,-1,(0,0,255),3,lineType=cv2.LINE_AA)

        ix = 0
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > cv2.contourArea(contours[ix]):
                ix = i

        print(img.shape, ix, cv2.contourArea(contours[ix]))
        result = np.zeros((*img.shape, 3))
        out = drawContours(result, contours, ix, (255,255,255), 5, lineType=cv2.LINE_AA)
        # print(out)
        self.show_array(out)

        # 计算轮廓的傅里叶描述子
        contour = contours[ix]
        des = []
        print("contour size", contour.size)
        s = len(contour)
        for i in range(s):
            xsum = 0
            ysum = 0
            for j in range(s):
                p = contour[j]
                assert len(p) == 1
                p = p[0]
                # print(p)
                cof = 2 * np.pi * i * j / s
                x = p[0]
                y = p[1]
                xsum += x * np.cos(cof) + y * np.sin(cof)
                ysum += y * np.cos(cof) - x * np.sin(cof)
            des.append((xsum, ysum))

        def formatter(t):
            l = np.sqrt(t[0] * t[0] + t[1] * t[1])
            return f'{round(l, 3)}: ({round(t[0], 3)}, {round(t[1], 3)})'

        lis = [f'前32个描述子（总共{s}个）--> 描述子长度: (x, y)']
        lis += [formatter(des[i]) for i in range(32)]
        self.infomation('\n'.join(lis))

        num, ok = QInputDialog.getInt(self, 'Reconstruct', f'请输入重构项数（总共{s}项）')

        if num and ok:
            for t in des[:num]:
                print(t)
            out = drawContours(result, contour, -1, (255,255,255), 5, lineType=cv2.LINE_AA)
            # print(out)
            self.show_array(out)
        

    def Roberts(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Roberts(img)
        self.show_array(out)


    def Sobel(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Sobel(img)
        self.show_array(out)


    def Prewitt(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Prewitt(img)
        self.show_array(out)


    def Scharr(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Scharr(img)
        self.show_array(out)


    def Laplacian(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Laplacian(img)
        self.show_array(out)

    def Log(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Log(img)
        self.show_array(out)

    def Canny(self):
        t = self.get_current_pic()
        t.save('result.png')
        img = cv2.imread('result.png', 0)
        out = Canny(img)
        self.show_array(out)


app = QApplication([])
mainw = MainWindow()
mainw.show()
app.exec_()