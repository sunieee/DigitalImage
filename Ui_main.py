# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\sy650\Desktop\file\photo\src\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(1524, 882)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(17, 3, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        spacerItem2 = QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem5 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(700, 725))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setAcceptDrops(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tab.setFont(font)
        self.tab.setObjectName("tab")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(-10, -10, 721, 741))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("c:\\Users\\sy650\\Desktop\\file\\photo\\src\\test.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(240, 310, 241, 61))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.label_3.setFont(font)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setObjectName("label_3")
        self.tabWidget.addTab(self.tab, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setMinimumSize(QtCore.QSize(80, 0))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.toIn = QtWidgets.QPushButton(self.frame_2)
        self.toIn.setGeometry(QtCore.QRect(10, 580, 50, 50))
        self.toIn.setMinimumSize(QtCore.QSize(50, 50))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(24)
        self.toIn.setFont(font)
        self.toIn.setObjectName("toIn")
        self.toOut = QtWidgets.QPushButton(self.frame_2)
        self.toOut.setGeometry(QtCore.QRect(10, 490, 50, 50))
        self.toOut.setMinimumSize(QtCore.QSize(50, 50))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(24)
        self.toOut.setFont(font)
        self.toOut.setObjectName("toOut")
        self.horizonInput = QtWidgets.QTextEdit(self.frame_2)
        self.horizonInput.setGeometry(QtCore.QRect(10, 60, 80, 31))
        self.horizonInput.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setFamily("明黑")
        font.setPointSize(10)
        self.horizonInput.setFont(font)
        self.horizonInput.setToolTip("")
        self.horizonInput.setObjectName("horizonInput")
        self.verticalInput = QtWidgets.QTextEdit(self.frame_2)
        self.verticalInput.setGeometry(QtCore.QRect(10, 140, 80, 31))
        self.verticalInput.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setFamily("明黑")
        font.setPointSize(10)
        self.verticalInput.setFont(font)
        self.verticalInput.setObjectName("verticalInput")
        self.label_6 = QtWidgets.QLabel(self.frame_2)
        self.label_6.setGeometry(QtCore.QRect(10, 30, 71, 21))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame_2)
        self.label_7.setGeometry(QtCore.QRect(10, 110, 71, 21))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.colorInput = QtWidgets.QTextEdit(self.frame_2)
        self.colorInput.setGeometry(QtCore.QRect(10, 230, 80, 31))
        self.colorInput.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setFamily("明黑")
        font.setPointSize(10)
        self.colorInput.setFont(font)
        self.colorInput.setObjectName("colorInput")
        self.label_8 = QtWidgets.QLabel(self.frame_2)
        self.label_8.setGeometry(QtCore.QRect(10, 200, 71, 21))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.frame_2)
        self.label_9.setGeometry(QtCore.QRect(10, 290, 71, 21))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.disInput = QtWidgets.QTextEdit(self.frame_2)
        self.disInput.setGeometry(QtCore.QRect(10, 320, 80, 31))
        self.disInput.setMinimumSize(QtCore.QSize(80, 0))
        font = QtGui.QFont()
        font.setFamily("明黑")
        font.setPointSize(10)
        self.disInput.setFont(font)
        self.disInput.setObjectName("disInput")
        self.horizontalLayout.addWidget(self.frame_2)
        self.tabWidgetR = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidgetR.sizePolicy().hasHeightForWidth())
        self.tabWidgetR.setSizePolicy(sizePolicy)
        self.tabWidgetR.setMinimumSize(QtCore.QSize(700, 725))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tabWidgetR.setFont(font)
        self.tabWidgetR.setTabsClosable(False)
        self.tabWidgetR.setMovable(True)
        self.tabWidgetR.setTabBarAutoHide(True)
        self.tabWidgetR.setObjectName("tabWidgetR")
        self.tab_2 = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tab_2.setFont(font)
        self.tab_2.setObjectName("tab_2")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(-20, -10, 731, 741))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("c:\\Users\\sy650\\Desktop\\file\\photo\\src\\test1.png"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.tabWidgetR.addTab(self.tab_2, "")
        self.horizontalLayout.addWidget(self.tabWidgetR)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMinimumSize(QtCore.QSize(0, 32))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 1461, 34))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem8)
        self.left1 = QtWidgets.QPushButton(self.layoutWidget)
        self.left1.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.left1.setFont(font)
        self.left1.setObjectName("left1")
        self.horizontalLayout_3.addWidget(self.left1)
        self.close1 = QtWidgets.QPushButton(self.layoutWidget)
        self.close1.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.close1.setFont(font)
        self.close1.setObjectName("close1")
        self.horizontalLayout_3.addWidget(self.close1)
        self.right1 = QtWidgets.QPushButton(self.layoutWidget)
        self.right1.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.right1.setFont(font)
        self.right1.setObjectName("right1")
        self.horizontalLayout_3.addWidget(self.right1)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem9)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem10)
        self.left2 = QtWidgets.QPushButton(self.layoutWidget)
        self.left2.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.left2.setFont(font)
        self.left2.setObjectName("left2")
        self.horizontalLayout_3.addWidget(self.left2)
        self.close2 = QtWidgets.QPushButton(self.layoutWidget)
        self.close2.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.close2.setFont(font)
        self.close2.setObjectName("close2")
        self.horizontalLayout_3.addWidget(self.close2)
        self.right2 = QtWidgets.QPushButton(self.layoutWidget)
        self.right2.setMinimumSize(QtCore.QSize(30, 30))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.right2.setFont(font)
        self.right2.setObjectName("right2")
        self.horizontalLayout_3.addWidget(self.right2)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem11)
        self.verticalLayout.addWidget(self.frame)
        spacerItem12 = QtWidgets.QSpacerItem(17, 3, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem12)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setEnabled(True)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1524, 27))
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(14)
        self.menubar.setFont(font)
        self.menubar.setFocusPolicy(QtCore.Qt.TabFocus)
        self.menubar.setAutoFillBackground(False)
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menu.setFont(font)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menu_2.setFont(font)
        self.menu_2.setObjectName("menu_2")
        self.menuFFT = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menuFFT.setFont(font)
        self.menuFFT.setObjectName("menuFFT")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menu_3.setFont(font)
        self.menu_3.setObjectName("menu_3")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menu_5.setFont(font)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("华文细黑")
        font.setPointSize(12)
        self.menu_6.setFont(font)
        self.menu_6.setObjectName("menu_6")
        MainWindow.setMenuBar(self.menubar)
        self.open = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.open.setFont(font)
        self.open.setObjectName("open")
        self.save = QtWidgets.QAction(MainWindow)
        self.save.setObjectName("save")
        self.saveR = QtWidgets.QAction(MainWindow)
        self.saveR.setObjectName("saveR")
        self.close = QtWidgets.QAction(MainWindow)
        self.close.setObjectName("close")
        self.add = QtWidgets.QAction(MainWindow)
        self.add.setObjectName("add")
        self.reverse = QtWidgets.QAction(MainWindow)
        self.reverse.setObjectName("reverse")
        self.rotate = QtWidgets.QAction(MainWindow)
        self.rotate.setObjectName("rotate")
        self.histogram = QtWidgets.QAction(MainWindow)
        self.histogram.setObjectName("histogram")
        self.FFT = QtWidgets.QAction(MainWindow)
        self.FFT.setObjectName("FFT")
        self.a1D_FFT_center = QtWidgets.QAction(MainWindow)
        self.a1D_FFT_center.setObjectName("a1D_FFT_center")
        self.high = QtWidgets.QAction(MainWindow)
        self.high.setObjectName("high")
        self.enlarge = QtWidgets.QAction(MainWindow)
        self.enlarge.setObjectName("enlarge")
        self.shrink = QtWidgets.QAction(MainWindow)
        self.shrink.setObjectName("shrink")
        self.scale = QtWidgets.QAction(MainWindow)
        self.scale.setObjectName("scale")
        self.fouriour = QtWidgets.QAction(MainWindow)
        self.fouriour.setObjectName("fouriour")
        self.reconstruct = QtWidgets.QAction(MainWindow)
        self.reconstruct.setObjectName("reconstruct")
        self.Roberts = QtWidgets.QAction(MainWindow)
        self.Roberts.setObjectName("Roberts")
        self.Sobel = QtWidgets.QAction(MainWindow)
        self.Sobel.setObjectName("Sobel")
        self.Prewitt = QtWidgets.QAction(MainWindow)
        self.Prewitt.setObjectName("Prewitt")
        self.Laplacian = QtWidgets.QAction(MainWindow)
        self.Laplacian.setObjectName("Laplacian")
        self.resize = QtWidgets.QAction(MainWindow)
        self.resize.setObjectName("resize")
        self.resizeAll = QtWidgets.QAction(MainWindow)
        self.resizeAll.setObjectName("resizeAll")
        self.blend = QtWidgets.QAction(MainWindow)
        self.blend.setObjectName("blend")
        self.move = QtWidgets.QAction(MainWindow)
        self.move.setObjectName("move")
        self.closeR = QtWidgets.QAction(MainWindow)
        self.closeR.setObjectName("closeR")
        self.r1D_FFT = QtWidgets.QAction(MainWindow)
        self.r1D_FFT.setObjectName("r1D_FFT")
        self.r1D_FFT_center = QtWidgets.QAction(MainWindow)
        self.r1D_FFT_center.setObjectName("r1D_FFT_center")
        self.gray = QtWidgets.QAction(MainWindow)
        self.gray.setObjectName("gray")
        self.grey = QtWidgets.QAction(MainWindow)
        self.grey.setObjectName("grey")
        self.low = QtWidgets.QAction(MainWindow)
        self.low.setObjectName("low")
        self.Scharr = QtWidgets.QAction(MainWindow)
        self.Scharr.setObjectName("Scharr")
        self.Log = QtWidgets.QAction(MainWindow)
        self.Log.setObjectName("Log")
        self.Canny = QtWidgets.QAction(MainWindow)
        self.Canny.setObjectName("Canny")
        self.tailor = QtWidgets.QAction(MainWindow)
        self.tailor.setObjectName("tailor")
        self.out2in = QtWidgets.QAction(MainWindow)
        self.out2in.setObjectName("out2in")
        self.colorSeperate = QtWidgets.QAction(MainWindow)
        self.colorSeperate.setObjectName("colorSeperate")
        self.singleLine = QtWidgets.QAction(MainWindow)
        self.singleLine.setObjectName("singleLine")
        self.allLine = QtWidgets.QAction(MainWindow)
        self.allLine.setObjectName("allLine")
        self.colorFFT = QtWidgets.QAction(MainWindow)
        self.colorFFT.setObjectName("colorFFT")
        self.colorHigh = QtWidgets.QAction(MainWindow)
        self.colorHigh.setObjectName("colorHigh")
        self.colorLow = QtWidgets.QAction(MainWindow)
        self.colorLow.setObjectName("colorLow")
        self.toPDF = QtWidgets.QAction(MainWindow)
        self.toPDF.setObjectName("toPDF")
        self.toPIC = QtWidgets.QAction(MainWindow)
        self.toPIC.setObjectName("toPIC")
        self.remove_gray = QtWidgets.QAction(MainWindow)
        self.remove_gray.setObjectName("remove_gray")
        self.reverseAll = QtWidgets.QAction(MainWindow)
        self.reverseAll.setObjectName("reverseAll")
        self.composeAll = QtWidgets.QAction(MainWindow)
        self.composeAll.setObjectName("composeAll")
        self.toGIF = QtWidgets.QAction(MainWindow)
        self.toGIF.setObjectName("toGIF")
        self.horizontal_split = QtWidgets.QAction(MainWindow)
        self.horizontal_split.setObjectName("horizontal_split")
        self.menu.addAction(self.open)
        self.menu.addAction(self.save)
        self.menu.addAction(self.saveR)
        self.menu.addAction(self.close)
        self.menu.addAction(self.closeR)
        self.menu_2.addAction(self.add)
        self.menu_2.addAction(self.reverse)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.grey)
        self.menu_2.addAction(self.resize)
        self.menu_2.addAction(self.resizeAll)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.rotate)
        self.menu_2.addAction(self.enlarge)
        self.menu_2.addAction(self.shrink)
        self.menu_2.addAction(self.scale)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.histogram)
        self.menuFFT.addAction(self.FFT)
        self.menuFFT.addSeparator()
        self.menuFFT.addAction(self.high)
        self.menuFFT.addAction(self.low)
        self.menuFFT.addSeparator()
        self.menuFFT.addAction(self.colorHigh)
        self.menuFFT.addAction(self.colorLow)
        self.menuFFT.addSeparator()
        self.menuFFT.addAction(self.reconstruct)
        self.menu_3.addAction(self.Roberts)
        self.menu_3.addAction(self.Sobel)
        self.menu_3.addAction(self.Prewitt)
        self.menu_3.addAction(self.Scharr)
        self.menu_3.addAction(self.Laplacian)
        self.menu_3.addAction(self.Log)
        self.menu_3.addAction(self.Canny)
        self.menu_5.addAction(self.tailor)
        self.menu_5.addAction(self.colorSeperate)
        self.menu_5.addAction(self.singleLine)
        self.menu_5.addAction(self.allLine)
        self.menu_6.addAction(self.toPDF)
        self.menu_6.addAction(self.toGIF)
        self.menu_6.addAction(self.remove_gray)
        self.menu_6.addAction(self.composeAll)
        self.menu_6.addAction(self.horizontal_split)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menuFFT.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "输入"))
        self.label.setText(_translate("MainWindow", "输出"))
        self.label_3.setText(_translate("MainWindow", "请打开一张图片/PDF"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Default"))
        self.toIn.setText(_translate("MainWindow", "<"))
        self.toOut.setText(_translate("MainWindow", ">"))
        self.horizonInput.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.verticalInput.setPlaceholderText(_translate("MainWindow", "0-100"))
        self.label_6.setText(_translate("MainWindow", "水平坐标轴"))
        self.label_7.setText(_translate("MainWindow", "垂直坐标轴"))
        self.colorInput.setPlaceholderText(_translate("MainWindow", "6"))
        self.label_8.setText(_translate("MainWindow", "色彩分离数"))
        self.label_9.setText(_translate("MainWindow", "取点距区间"))
        self.disInput.setPlaceholderText(_translate("MainWindow", "5-20"))
        self.tabWidgetR.setTabText(self.tabWidgetR.indexOf(self.tab_2), _translate("MainWindow", "Default"))
        self.left1.setText(_translate("MainWindow", "<"))
        self.close1.setText(_translate("MainWindow", "×"))
        self.right1.setText(_translate("MainWindow", ">"))
        self.left2.setText(_translate("MainWindow", "<"))
        self.close2.setText(_translate("MainWindow", "×"))
        self.right2.setText(_translate("MainWindow", ">"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "基本操作"))
        self.menuFFT.setTitle(_translate("MainWindow", "FFT"))
        self.menu_3.setTitle(_translate("MainWindow", "边缘检测"))
        self.menu_5.setTitle(_translate("MainWindow", "提取"))
        self.menu_6.setTitle(_translate("MainWindow", "实用工具"))
        self.open.setText(_translate("MainWindow", "打开"))
        self.open.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.save.setText(_translate("MainWindow", "导出当前输入"))
        self.save.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.saveR.setText(_translate("MainWindow", "导出当前输入"))
        self.saveR.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.close.setText(_translate("MainWindow", "关闭当前输入"))
        self.close.setShortcut(_translate("MainWindow", "Ctrl+W"))
        self.add.setText(_translate("MainWindow", "加（所有输入图像）"))
        self.reverse.setText(_translate("MainWindow", "求反"))
        self.rotate.setText(_translate("MainWindow", "逆时针旋转"))
        self.histogram.setText(_translate("MainWindow", "直方图均衡化"))
        self.histogram.setShortcut(_translate("MainWindow", "Ctrl+Shift+G"))
        self.FFT.setText(_translate("MainWindow", "傅里叶变换(输出DFT/增强/反变换图)"))
        self.FFT.setShortcut(_translate("MainWindow", "Ctrl+F"))
        self.a1D_FFT_center.setText(_translate("MainWindow", "1D-FFT（低频中心）"))
        self.high.setText(_translate("MainWindow", "高通滤波"))
        self.high.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.enlarge.setText(_translate("MainWindow", "放大2倍"))
        self.enlarge.setShortcut(_translate("MainWindow", "Ctrl+Shift+="))
        self.shrink.setText(_translate("MainWindow", "缩小2倍"))
        self.shrink.setShortcut(_translate("MainWindow", "Ctrl+Shift+-"))
        self.scale.setText(_translate("MainWindow", "等比缩放到..."))
        self.fouriour.setText(_translate("MainWindow", "傅里叶描述子"))
        self.reconstruct.setText(_translate("MainWindow", "傅里叶描述子M项数重构（输出描述子/重构轮廓）"))
        self.reconstruct.setShortcut(_translate("MainWindow", "Ctrl+Shift+M"))
        self.Roberts.setText(_translate("MainWindow", "Roberts"))
        self.Sobel.setText(_translate("MainWindow", "Sobel"))
        self.Prewitt.setText(_translate("MainWindow", "Prewitt"))
        self.Laplacian.setText(_translate("MainWindow", "Laplacian"))
        self.resize.setText(_translate("MainWindow", "等比缩放到适应"))
        self.resize.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.resizeAll.setText(_translate("MainWindow", "全部输入图像等比缩放到适应"))
        self.resizeAll.setShortcut(_translate("MainWindow", "Ctrl+Shift+R"))
        self.blend.setText(_translate("MainWindow", "混合（所有打开的图像）"))
        self.move.setText(_translate("MainWindow", "平移"))
        self.closeR.setText(_translate("MainWindow", "关闭当前输出"))
        self.closeR.setShortcut(_translate("MainWindow", "Ctrl+Shift+W"))
        self.r1D_FFT.setText(_translate("MainWindow", "逆傅里叶变换"))
        self.r1D_FFT.setShortcut(_translate("MainWindow", "Ctrl+Shift+F"))
        self.r1D_FFT_center.setText(_translate("MainWindow", "逆1D-FFT（低频中心）"))
        self.gray.setText(_translate("MainWindow", "转为灰度图"))
        self.grey.setText(_translate("MainWindow", "转为灰度图"))
        self.grey.setShortcut(_translate("MainWindow", "Ctrl+G"))
        self.low.setText(_translate("MainWindow", "低通滤波"))
        self.low.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.Scharr.setText(_translate("MainWindow", "Scharr"))
        self.Log.setText(_translate("MainWindow", "Log"))
        self.Canny.setText(_translate("MainWindow", "Canny"))
        self.tailor.setText(_translate("MainWindow", "裁剪坐标轴"))
        self.tailor.setShortcut(_translate("MainWindow", "Ctrl+T"))
        self.out2in.setText(_translate("MainWindow", "将当前输出移到输入"))
        self.out2in.setShortcut(_translate("MainWindow", "Ctrl+M"))
        self.colorSeperate.setText(_translate("MainWindow", "颜色通道分离"))
        self.colorSeperate.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.singleLine.setText(_translate("MainWindow", "提取单一曲线数据点"))
        self.singleLine.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.allLine.setText(_translate("MainWindow", "提取曲线数据点全流程"))
        self.colorFFT.setText(_translate("MainWindow", "彩色傅里叶变换"))
        self.colorFFT.setShortcut(_translate("MainWindow", "Ctrl+Shift+F"))
        self.colorHigh.setText(_translate("MainWindow", "彩色高通滤波"))
        self.colorHigh.setShortcut(_translate("MainWindow", "Ctrl+Shift+H"))
        self.colorLow.setText(_translate("MainWindow", "彩色低通滤波"))
        self.colorLow.setShortcut(_translate("MainWindow", "Ctrl+Shift+L"))
        self.toPDF.setText(_translate("MainWindow", "全部输入图像转PDF"))
        self.toPIC.setText(_translate("MainWindow", "PDF转图片"))
        self.remove_gray.setText(_translate("MainWindow", "去除水印"))
        self.reverseAll.setText(_translate("MainWindow", "反色"))
        self.composeAll.setText(_translate("MainWindow", "合成大图"))
        self.toGIF.setText(_translate("MainWindow", "全部输入图像/PDF转GIF"))
        self.horizontal_split.setText(_translate("MainWindow", "水平分割"))
