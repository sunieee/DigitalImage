from PIL import Image
import os
from moviepy.editor import ImageSequenceClip
from util import generate_name
from pdf2image import convert_from_path
from tqdm import tqdm
import numpy as np
import re


re_digits = re.compile(r'(\d+)')


def embedded_numbers(s):    
    pieces = re_digits.split(s)                 # 切成数字和非数字
    pieces[1::2] = map(int, pieces[1::2])       # 将数字部分转成整数
    return pieces
 

def get_folder(folder, deep=False):
    files = []
    if type(folder) == list:
        for t in folder:
            if t.endswith('.png') or t.endswith('.jpg'):
                if not os.path.exists(t):
                    print(f"file {t} not exists. Ignore!")
                else:
                    files.append(t)
            elif os.path.isdir(t) and deep:
                files += get_folder(t, True)
        
    elif os.path.isdir(folder):
        for t in os.listdir(folder):
            if deep:
                files += get_folder(os.path.join(folder, t), True)
            else:
                if t.endswith('.png') or t.endswith('.jpg'):
                    files.append(os.path.join(folder, t))

    elif folder.endswith('.png') or folder.endswith('.jpg'):
        files.append(folder)

    return sorted(files, key=embedded_numbers) 


def pic2pdf(folder, deep=False, path=None):
    sources = []
    pngFiles = get_folder(folder, deep)
    output = Image.open( pngFiles[0] )
    pngFiles.pop(0)
    for file in tqdm(pngFiles):
        pngFile = Image.open( file )
        if pngFile.mode == "RGB":
            pngFile = pngFile.convert( "RGB" )
        sources.append(pngFile)
    if path is None:
        path = generate_name('.pdf')
    output.save(path, "pdf", save_all=True, append_images=sources)
    return path


def pic2gif(folder, deep=False, fps=1, path=None):
    names = get_folder(folder, deep=deep)
    clip = ImageSequenceClip(names, fps=fps)
    if path is None:
        path = generate_name('.gif')
    clip.write_gif(path)
    return path


def pdf2pic(pdf, dpi=144):
    folder = generate_name('')
    for i, page in tqdm(enumerate(convert_from_path(pdf, dpi=dpi))):
        page.save(os.path.join(folder, f'{i}.jpg'), 'JPEG')
    return folder


def remove_gray(pic, thresh=150, path=None):
    img = Image.open(pic)
    img = img.convert('RGB')   # 修改颜色通道为RGB
    x, y = img.size   # 获得长和宽

    for i in tqdm(range(x)):
        for k in range(y):
            c = img.getpixel((i, k))
            if np.mean(c) > thresh and c[0] == c[1] == c[2]:
                img.putpixel((i, k), (255,255,255))

    # img.show()
    if path is None:
        path = generate_name()
    img.save(path)
    return path

def get_size(pic):
    """输入路径，返回宽，长"""
    a, b = Image.open(pic).size
    w = max(a,b)
    h = min(a,b)
    return w, h


def concat(folder, line_max=2, scale=0.8, path=None):
    all_path = get_folder(folder)
    N = len(all_path)
    assert N>0
    row_max = int((N-1) / line_max) + 1

    # 固定一个宽度，最后的实际宽度（取均值）
    width = 0
    for i in range(N):
        w, _ = get_size(all_path[i])
        width += w
    width = int(width * scale/N)

    # 同行的高度以第一个为准
    height_total = 0
    for i in range(0, N, line_max):
        w, h = get_size(all_path[i])
        h = int(width * h/w)
        print(f"the {i}th height is {h}")
        height_total += h

    toImage = Image.new('RGBA',(width * line_max, height_total))
    height = 0
    height_row = 0

    i=-1
    num = 0
    while True:
        i+=1
        for j in range(line_max):
            # 每次打开图片绝对路路径列表的第一张图片
            pic_fole_head = Image.open(all_path[num])
            w, h = get_size(all_path[num])

            # 获取行首图片的高度
            if j % line_max==0:
                height += height_row
                height_row = int(width * h/w)

            # 按照指定的尺寸，给图片重新赋值，<PIL.Image.Image image mode=RGB size=200x200 at 0x127B7978>
            tmppic = pic_fole_head.resize((width, height_row))

            # 计算每个图片的左上角的坐标点(0, 0)，(0, 200)，(0, 400)，(200, 0)，(200, 200)。。。。(400, 400)
            loc = (int(j * width), int(height))
            print("第{}/{}张图的存放位置".format(num+1, N),loc)
            toImage.paste(tmppic, loc)
            num = num + 1

            if num >= N:
                break
        if num >= N:
            break

    # print(toImage.size)
    if path is None:
        path = generate_name()
    toImage.save(path)
    return path



if __name__ == "__main__":
    # print(get_folder('..', True))
    # print(get_folder('..', False))
    # pic2gif('gif')
    # concat("C:/Users/sy650/Desktop/file/photo/src/helper/gif")
    img = Image.open('data/2/1.png')
    print(img)
    print(img.size, np.array(img).shape)
    print(np.array(img).astype('uint8'))
    # print(np.ones(np.array(img).shape, dtype=np.uint8) * 255-img)