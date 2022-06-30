# 数字图像处理

## 程序说明

### 目录结构

- main.py：程序主入口，可通过`python main.py`启动PyQt5界面
- main.ui：使用Qt Designer进行的界面设计
- Ui_main.py：通过编译.ui文件生成的界面代码，供主程序调用
- util.py：各种工具函数，包含直方图、傅里叶变换、各种边缘检测算法等
- dist：此文件夹中包含打包的各种库和依赖，及可执行文件
  - 直接运行UI程序，请点击`dist/main/main.exe`

打包命令：

```sh
pyinstaller -D -i test.ico main.py
```

### 程序界面

程序打开后如下图所示，包含输入输出两个窗格，都是固定600*600大小，窗口可放大缩小，当窗格不可伸缩。

![image-20220630223352486](https://n.sunie.top:9000/gallery/2022summer/202206302233866.png)

菜单栏包含5个主菜单，每个菜单中都有一些具体的功能，一一对应了下面大作业的所有需求，部分功能都相应的快捷键。

- 文件：打开、关闭和导出图片，注意打开之后会显示在输入窗格
- 输入：对输入窗格内的图片进行处理，直接在输入窗口中修改（inplace操作）
- 输出：使用输入窗格内激活的图片作为输入，进行图像基础操作，在输出窗格出显示图片
- FFT：使用输入窗格内激活的图片作为输入，自动转为灰度图，进行FFT相关操作，在输出窗格出显示图片
  - “傅里叶变换”功能一共输出

- 边缘检测：使用输入窗格内激活的图片作为输入，自动转为灰度图，进行边缘检测相关操作，在输出窗格出显示图片

![image-20220630223744228](https://n.sunie.top:9000/gallery/2022summer/202206302316409.png)

![image-20220630225123867](https://n.sunie.top:9000/gallery/2022summer/202206302251998.png)

在运行程序时：

- 控制台输出的是程序运行的输出，如果执行异常，程序会退出，退出原因间控制台输出
- 打开的图片会被压缩显示，即长、宽中较大的一个被设置成600，另一个维度等比缩放。但是图片的实际尺寸不变，并通过数字显示在右上角
- 输入输出均有多个窗格，可以在窗格之间进行切换，但是尚不支持点击'x'关闭，想要关闭窗格需点击菜单>文件>关闭
- 每个窗格中当前展示的内容是正在激活的tab

## 作业文档

### 大作业内容（1）

- 基于VC的多文档界面（MDI ）方式，设计数字图像处理程序框架
- 软件中编程实现BMP格式图像文件的读取、显示
- 选择实现JPG、 RAW格式文件的读取、显示，以及与BMP格式的转换
- 完成图像的基本操作：加、求反、几何变换
- 完成图像的直方图均衡化处理

#### 文件的读写

文件的格式种类多样，打开和关闭都支持四种类型的格式。通过这种方式实现了各种图片格式之间的转换。

![image-20220630225657487](https://n.sunie.top:9000/gallery/2022summer/202206302256827.png)

#### 图像的基本操作

![image-20220630230720121](https://n.sunie.top:9000/gallery/2022summer/202206302307886.png)

加法会使用所有打开的输入进行加法，最终会使用所有图中最大宽和高最为新图的尺寸。

**注意**：请不要使用通道不一致的图片进行加分操作，会造成程序崩溃（可预先都转换为灰度图）

![image-20220630230834085](https://n.sunie.top:9000/gallery/2022summer/202206302308093.png)

进行缩放呈现的效果不会改变，但实际的大小已经变了，见右上角的尺寸。

#### 直方图均衡化

色彩明显变得更加鲜艳：

![image-20220628191431704](https://n.sunie.top:9000/gallery/2022summer/202206302316413.png)



## 大作业内容（2）

FFT作业说明

- 实现图像的FFT变换和显示
- 实现FFT反变换

观察典型图像FFT变换后的频谱图

- 首先构造一幅黑白二值测试图像，例如：在128×128的黑色背景中心产生一个4×4的白色方块。然后依次进行以下测试。 

  - DFT

    <img src="https://n.sunie.top:9000/gallery/2022summer/202206281712345.png" alt="image-20220627215141715" style="zoom:67%;" />

  - 平移、缩放

    <img src="https://n.sunie.top:9000/gallery/2022summer/202206281712605.png" alt="image-20220627215247881" style="zoom:67%;" />

#### FFT

通过菜单栏>FFT>傅里叶变换进行，执行过程中输出3张图片：

- 复数域上频谱图（强行转换uint8使得图片失真）
- 动态范围压缩的2DFT图（将值最高的点映射成亮度255）
- FFT反变换的输出图

![image-20220630231153611](https://n.sunie.top:9000/gallery/2022summer/202206302311110.png)

#### 高通/低通滤波

支持自定义滤波半径，通过交互式输入方式选择，高通、低通的结果分别如下：

![image-20220630232002932](https://n.sunie.top:9000/gallery/2022summer/202206302320811.png)

## 大作业内容（3）

对于图1中XY平面上的边界，对其进行傅里叶描述子的表示，用不同的项数重构

![image-20220627215629337](https://n.sunie.top:9000/gallery/2022summer/202206272156275.png)

傅里叶描述子是一种图像特征，用来描述轮廓的特征参数。

傅里叶描述子的基本思想是：首先我们设定物体的**形状轮廓**是一条闭合的曲线，一个点沿边界曲线运动，假设这个点为p(l)，它的复数形式的坐标为x(l)+jy(l)，它的周期是这个闭合曲线的周长，这也表明属于一个周期函数。该以曲线周长作为周期的函数能够通过傅里叶[级数](https://so.csdn.net/so/search?q=级数&spm=1001.2101.3001.7020)表示。在傅里叶级数里面的多个系数z(k)与闭合边界曲线的形状有着直接关系，将其定义为**傅里叶描述子**。当取到足够阶次的系数项z(k)时，傅里叶描述子能够完全提取形状信息，并恢复物体的形状。
也就是说，傅里叶描述子用一个向量表示轮廓，将轮廓数字化，从而能更好的区分不同的轮廓，达到识别物体的目的。傅里叶描述子的特点是简单并且非常高效，是识别物体形状的重要方法之一。

简单来说，傅里叶描述子就是用一个向量代表一个轮廓，将轮廓数字化，从而能更好地区分不同的轮廓，进而达到识别物体的目的。

如上图所示，少数的傅里叶描述子就可以用于捕获边界的大体特征。这一性质很有用，因为这些系数携带有形状信息。

**整个流程如下：**

1. 边缘检测：使用边缘检测算法将边缘提取出来，并执行闭操作，让边缘更明晰，去掉小黑点
2. 选择所有轮廓中最大轮廓，即目标轮廓，绘制出来
3. 计算轮廓的傅里叶描述子，通过窗格方式输出前32个描述子
4. 可选择描述子的项数进行重构。

![image-20220630232156686](https://n.sunie.top:9000/gallery/2022summer/202206302321671.png)



## 大作业内容（4）

边缘检测

- 编程实现基于典型微分算子（不少于Roberts、Sobel、Prewitt、拉普拉斯算子）的图像边缘提取，能够读取图像文件内容，进行检测后输出边缘检测结果
- 分析比较不同算子的特性

![image-20220630233332598](https://n.sunie.top:9000/gallery/2022summer/202206302333920.png)

![image-20220630233039433](https://n.sunie.top:9000/gallery/2022summer/202206302330088.png)

![image-20220630233227582](https://n.sunie.top:9000/gallery/2022summer/202206302332421.png)

| 算子      | 优缺点比较                                                   |
| --------- | ------------------------------------------------------------ |
| Roberts   | 对具有陡峭的低噪声的图像处理效果较好，但利用Roberts算子提取边缘的结果是边缘比较粗，因此边缘定位不是很准确 |
| Sobel     | 对灰度渐变和噪声较多的图像处理效果比较好，Sobel算子对边缘定位比较准确 |
| Scharr    | 与Sobel算子的不同点是在平滑部分，这里所用的平滑算子是 1/16 *[3, 10, 3]，相比于 1/4*[1, 2, 1]，中心元素占的权重更重。假设图像这种随机性较强的信号，领域相关性不大 |
| Prewitt   | 对灰度渐变和噪声较多的图像处理效果较好                       |
| Laplacian | 对图像中的阶跃性边缘点定位准确，对噪声非常敏感，丢失一部分边缘的方向信息，造成一些不连续的检测边缘。 |
| LoG       | LG算子经常出现双边缘像素边界，而且该检测方法对噪声比较敏感，所以很少用LG算子检测边缘，而是用来判断边缘像素是位于图像的明区还是暗区。 |
| Canny     | 此方法不容易受噪声的干扰，能够检测到真正的弱边缘。在edge函数中，最有效的边缘检测方法是Canny方法。该方法的优点在于使用两种不同的阙值分别检测强边缘和弱边缘，并且仅当弱边缘与强边缘相连时，才将弱边缘包含在输出图像 。因此，这种方法不容易被噪声“填充”，更容易检测出真正的弱边缘 |



## 大作业内容（5）

设有信源A＝｛a0，a1，a2， a3，a4，a5， a6，a7｝,信源对应的概率P＝｛0.24, 0.20,0.15, 0.10, 0.07, 0.03, 0.19, 0.02｝。下表已按字符概率进行了排序，请在下表中给出信源字符的霍夫曼编码过程和结果（注意：为保证编码结果的确定性，编码过程中要求大概率的赋码字0，小概率的赋码字1）。

| **字符** | **概率** | **过程** | **结果** |
| -------- | -------- | -------- | -------- |
| a0       | 0.24     |          | 01       |
| a1       | 0.20     |          | 11       |
| a6       | 0.19     |          | 000      |
| a2       | 0.15     |          | 001      |
| a3       | 0.10     |          | 101      |
| a4       | 0.07     |          | 1000     |
| a5       | 0.03     |          | 10011    |
| a7       | 0.02     |          | 10010    |

（1）请计算上述霍夫曼编码的平均码长和编码效率
（2）根据编码结果，对码串“010001011001001”进行解码

![image-20220701000412165](https://n.sunie.top:9000/gallery/2022summer/202207010004227.png)

![image-20220701000419264](https://n.sunie.top:9000/gallery/2022summer/202207010004822.png)