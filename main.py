from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog, QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QInputDialog
from cv2 import drawContours
from LineExtraction.extractor import Extractor
from Ui_main import Ui_MainWindow
from PIL import Image
import numpy as np
from util import *
from LineExtraction.tailor import Tailor
from LineExtraction.seperate import Seperator
import cv2
from pic import Pic, base_folder
from helper import *


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

    
    def wrap(self, x):
        # print(x)
        self.__dict__[x] = lambda _: self.show_output(
            self.get_current_pic().base(x)
        )

    def bind(self):
        for func in ['grey','resize','enlarge','shrink',
            'Roberts','Sobel','Prewitt','Scharr','Laplacian','Log','Canny',
            'reverse','histogram']:
            self.wrap(func)

        # self.ui.open.activate.connect(self.open)
        # def wrap():
        #     threading.Thread(target=getattr(self, t), args=(self)).start()
        for t in self.ui.__dict__.keys():
            if t in dir(self):
                # print(self.ui.__dict__[t], type(self.ui.__dict__[t]))
                if type(self.ui.__dict__[t]) == QPushButton:
                    self.ui.__dict__[t].clicked.connect(getattr(self, t))
                else: 
                    self.ui.__dict__[t].triggered.connect(getattr(self, t))
    
        self.ui.close1.clicked.connect(self.close)
        self.ui.close2.clicked.connect(self.closeR)


    def open(self):
        filePaths, _  = QFileDialog.getOpenFileNames(
            self,             # 父窗口对象
            "选择你要上传的图片", # 标题
            base_folder,        # 起始目录
            "支持类型 (*.png *.jpg *.bmp *.raw *.data *.pdf);;图片类型 (*.png *.jpg *.bmp *.raw *.data);;jpg类型 (*.jpg);;png类型 (*.png);;bmp类型 (*.bmp);;raw类型 (*.raw *.data);;pdf类型 (*.pdf)" # 选择类型过滤项，过滤内容在括号中
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

    def current_pic(self):
        w = self.ui.tabWidget.currentWidget()
        for pic in self.opened:
            if pic.widget == w:
                return pic

    def get_current_pic(self):
        w = self.ui.tabWidget.currentWidget()
        for pic in self.opened:
            if pic.widget == w:
                return pic
        self.warning('输入窗口尚未打开任何图片！')

    def current_picR(self):
        w = self.ui.tabWidgetR.currentWidget()
        for pic in self.output:
            if pic.widget == w:
                return pic

    def get_current_picR(self):
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


    def toIn(self):
        ix = len(self.opened) + 1
        t = self.get_current_picR()
        self.output.remove(t)
        self.opened.append(t)
        self.ui.tabWidgetR.removeTab(self.ui.tabWidgetR.currentIndex())
        self.ui.tabWidget.addTab(t.widget, str(ix))
        self.ui.tabWidget.setCurrentIndex(ix)

    def toOut(self):
        ix = len(self.output) + 1
        t = self.get_current_pic()
        self.opened.remove(t)
        self.output.append(t)
        self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex())
        self.ui.tabWidgetR.addTab(t.widget, str(ix))
        self.ui.tabWidgetR.setCurrentIndex(ix)

    def resizeAll(self):
        for pic in self.opened:
            path = pic.base('grey')
            self.show_output(path)

    def add(self):
        if len(self.opened) == 0:
            self.warning('No pictures opened. Abort!')
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
        path = generate_name()
        resultImage.save(path)
        self.show_output(path)


    def show_input(self, path):
        pic = Pic(path, self)
        ix = len(self.opened) + 1
        self.opened.append(pic)
        self.ui.tabWidget.addTab(pic.create_widget(), str(ix))
        self.ui.tabWidget.setCurrentIndex(ix)
    

    def show_output(self, path):
        pic = Pic(path, self)
        ix = len(self.output) + 1
        self.output.append(pic)
        self.ui.tabWidgetR.addTab(pic.create_widget(), str(ix))
        self.ui.tabWidgetR.setCurrentIndex(ix)


    def rotate(self):
        num, ok = QInputDialog.getInt(self, 'Rotate', '将当前图片逆时针旋转 X°')
        if ok and num:
            path = self.get_current_pic().base('rotate', num)
            self.show_output(path)


    def scale(self):
        num, ok = QInputDialog.getDouble(self, 'Scale', '将当前图片放大/缩小到 X 倍')
        if ok and num:
            path = self.get_current_pic().base('scale', num)
            self.show_output(path)


    def show_array(self, result):
        result = result.astype('uint8')
        resultImage = Image.fromarray(result)
        path = generate_name()
        resultImage.save(path)
        self.show_output(path)

    def tailor(self):
        t = self.get_current_pic()
        ex = Tailor(t.path)
        self.show_array(ex.tailor)


    def colorSeperate(self):
        t = self.get_current_pic()
        path = t.path
        if np.mean(t.imgData[:]) > 200:
            path = t.base('reverse')
        s = Seperator(path)
        for i in range(len(s.color_img)):
            arr = np.array(s.color_img[i])
            self.show_array(arr)


    def singleLine(self):
        t = self.get_current_pic()
        ex = Extractor(t.path)
        path = ex.interpolate()
        self.show_output(path)


    def allLine(self):
        pass


    def FFT(self):
        t = self.get_current_pic()
        t.save()
        img = cv2.imread(t.path, 0)

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
        num, _ = QInputDialog.getInt(self, 'high', '滤波中心半径（默认50）')
        if not num:
            num = 50
        path = self.get_current_pic().base('high', num)
        self.show_output(path)


    def low(self):
        num, _ = QInputDialog.getInt(self, 'low', '滤波中心半径（默认50）')
        if not num:
            num = 50
        path = self.get_current_pic().base('low', num)
        self.show_output(path)


    def colorHigh(self):
        num, _ = QInputDialog.getInt(self, 'high', '滤波中心半径（默认50）')
        if not num:
            num = 50
        path = self.get_current_pic().base('colorHigh', num)
        self.show_output(path)
        

    def colorLow(self):
        num, _ = QInputDialog.getInt(self, 'low', '滤波中心半径（默认50）')
        if not num:
            num = 50
        path = self.get_current_pic().base('colorLow', num)
        self.show_output(path)


    def reconstruct(self):
        t = self.get_current_pic()
        t.save()
        img = cv2.imread(t.path, 0)

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

    def toPDF(self):
        folder = [t.src for t in self.opened if t.endswith('.png') or t.endswith('.jpg')]
        print('valid pictures are:', folder)
        path = pic2pdf(folder)
        self.show_output(path)

    def toGIF(self):
        t = self.get_current_pic()
        h, ok = QInputDialog.getInt(self, '正在生成gif', '请输入gif图片的fps（默认为1）')
        if not h or not ok:
            h = 1
        if t.lis:
            print('converting pdf to gif...')
            folder = t.lis
        else:
            print('converting pictures to gif...')
            folder = [t.src for t in self.opened if t.endswith('.png') or t.endswith('.jpg')]
        print('valid pictures are:', folder)
        path = pic2gif(folder)
        self.show_output(path)

    def remove_gray(self):
        t = self.get_current_pic()
        h, ok = QInputDialog.getInt(self, '正在去除水印', '请输入灰色阈值（默认为150，高于此都会被置于白色）')
        if not h or not ok:
            h = 150
        if t.lis:
            folder = generate_name('')
            print(f'共{len(t.lis)}张图片，请耐心等待...')
            for f in os.listdir(t.folder):
                remove_gray(os.path.join(t.folder, f), h, os.path.join(folder, f))
            
            path = pic2pdf(folder)
            self.show_output(path)
        else:
            path = remove_gray(t.path)
            self.show_output(path)

    def compose(self):
        t = self.get_current_pic()
        h, ok = QInputDialog.getInt(self, '正在合成大图', '请输入大图片每行的图片数量（默认为2）')
        if not h or not ok:
            h = 2
        if t.lis:
            print('converting pdf to gif...')
            folder = t.lis
        else:
            print('converting pictures to gif...')
            folder = [t.src for t in self.opened if t.endswith('.png') or t.endswith('.jpg')]
        print('valid pictures are:', folder)
        path = concat(folder, h)
        self.show_output(path)

    def left1(self):
        t = self.current_pic()
        if t and t.lis:
            t.left()
            return

        t = self.ui.tabWidget.currentIndex()
        if t == 0:
            t = self.ui.tabWidget.count()
        self.ui.tabWidget.setCurrentIndex(t - 1)


    def right1(self):
        t = self.current_pic()
        if t and t.lis:
            t.right()
            return

        t = self.ui.tabWidget.currentIndex()
        if t == self.ui.tabWidget.count()-1:
            t = -1
        if t == 0:
            t = 0 if self.ui.tabWidget.count()>0 else -1
        self.ui.tabWidget.setCurrentIndex(t + 1)


    def left2(self):
        t = self.current_picR()
        if t and t.lis:
            t.left()
            return

        t = self.ui.tabWidgetR.currentIndex()
        if t == 0:
            t = self.ui.tabWidgetR.count()
        self.ui.tabWidgetR.setCurrentIndex(t - 1)


    def right2(self):
        t = self.current_picR()
        if t and t.lis:
            t.right()
            return

        t = self.ui.tabWidgetR.currentIndex()
        if t == self.ui.tabWidgetR.count()-1:
            t = -1
        if t == 0:
            t = 0 if self.ui.tabWidgetR.count()>0 else -1
        self.ui.tabWidgetR.setCurrentIndex(t + 1)

app = QApplication([])
mainw = MainWindow()
mainw.show()
app.exec_()