from pathlib import Path
import re
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from helper import concat, get_folder, pdf2pic, pic2gif, pic2pdf, remove_gray
from LineExtraction.extractor import Extractor
from LineExtraction.seperate import Seperator
from LineExtraction.split import Split
from LineExtraction.tailor import Tailor
from util import (
    Canny,
    Laplacian,
    Log,
    Prewitt,
    Roberts,
    Scharr,
    Sobel,
    cache_folder,
    expand,
    fft2,
    fftshift,
    generate_name,
    histeq,
    ifft2,
    ifftshift,
    smoothing,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WORK_DIR = Path(cache_folder)
for folder in (DATA_DIR, UPLOAD_DIR, WORK_DIR):
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
EDGE_OPERATORS = {
    "Roberts": Roberts,
    "Sobel": Sobel,
    "Prewitt": Prewitt,
    "Scharr": Scharr,
    "Laplacian": Laplacian,
    "LoG": Log,
    "Canny": Canny,
}


def init_state():
    st.session_state.setdefault("inputs", [])
    st.session_state.setdefault("outputs", [])
    st.session_state.setdefault("input_index", 0)
    st.session_state.setdefault("output_index", 0)


def clamp_indices():
    for name, key in [("inputs", "input_index"), ("outputs", "output_index")]:
        items = st.session_state[name]
        if not items:
            st.session_state[key] = 0
        else:
            st.session_state[key] %= len(items)


def safe_name(name):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return stem or "upload"


def unique_upload_path(name):
    return UPLOAD_DIR / f"{time.strftime('%Y%m%d%H%M%S')}_{safe_name(name)}"


def queue_item(path, label):
    return {"path": str(path), "label": label}


def current_item(queue_name):
    items = st.session_state[queue_name]
    if not items:
        return None
    index_key = "input_index" if queue_name == "inputs" else "output_index"
    return items[st.session_state[index_key] % len(items)]


def add_output(path, label):
    st.session_state.outputs.append(queue_item(path, label))
    st.session_state.output_index = len(st.session_state.outputs) - 1


def image_paths(queue_name="inputs"):
    return [
        item["path"]
        for item in st.session_state[queue_name]
        if Path(item["path"]).suffix.lower() in IMAGE_SUFFIXES and Path(item["path"]).exists()
    ]


def save_pil(img, suffix=".png"):
    path = generate_name(suffix)
    if img.mode == "RGBA" and suffix.lower() in {".jpg", ".jpeg"}:
        img = img.convert("RGB")
    img.save(path)
    return path


def save_array(arr, suffix=".png"):
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr)
        max_value = arr.max() if arr.size else 0
        min_value = arr.min() if arr.size else 0
        if max_value > 255 or min_value < 0:
            arr = arr - min_value
            max_value = arr.max() if arr.size else 0
            if max_value:
                arr = arr / max_value * 255
        arr = np.clip(arr, 0, 255).astype("uint8")
    return save_pil(Image.fromarray(arr), suffix)


def read_image(path, mode=None):
    img = Image.open(path)
    return img.convert(mode) if mode else img


def ingest_uploads(files, pdf_dpi):
    for file in files:
        path = unique_upload_path(file.name)
        path.write_bytes(file.getbuffer())
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            folder = pdf2pic(str(path), dpi=pdf_dpi)
            for page in get_folder(folder):
                st.session_state.inputs.append(queue_item(page, f"{file.name} / {Path(page).stem}"))
        elif suffix in IMAGE_SUFFIXES:
            st.session_state.inputs.append(queue_item(path, file.name))
        else:
            st.warning(f"暂不支持 {file.name}，请上传图片或 PDF。")
    st.session_state.input_index = max(0, len(st.session_state.inputs) - 1)
    clamp_indices()


def display_current(item, queue_name):
    if not item:
        st.info("暂无图片")
        return

    path = Path(item["path"])
    if not path.exists():
        st.warning(f"{item['label']} 不存在")
        return

    suffix = path.suffix.lower()
    st.caption(f"{item['label']} | {path.relative_to(ROOT)}")
    if suffix in IMAGE_SUFFIXES or suffix == ".gif":
        st.image(str(path), use_container_width=True)
    elif suffix == ".csv":
        st.dataframe(np.genfromtxt(path, delimiter=",", names=True), use_container_width=True)
    elif suffix == ".pdf":
        st.info("PDF 文件已生成，可下载查看。")
    else:
        st.code(str(path))

    with path.open("rb") as fh:
        st.download_button(
            "下载当前文件",
            fh,
            file_name=path.name,
            key=f"download-{queue_name}-{path}",
        )


def shift(queue_name, delta):
    items = st.session_state[queue_name]
    if not items:
        return
    index_key = "input_index" if queue_name == "inputs" else "output_index"
    st.session_state[index_key] = (st.session_state[index_key] + delta) % len(items)


def close_current(queue_name):
    items = st.session_state[queue_name]
    if not items:
        return
    index_key = "input_index" if queue_name == "inputs" else "output_index"
    items.pop(st.session_state[index_key])
    clamp_indices()


def move_current(source, target):
    items = st.session_state[source]
    if not items:
        return
    source_key = "input_index" if source == "inputs" else "output_index"
    target_key = "input_index" if target == "inputs" else "output_index"
    item = items.pop(st.session_state[source_key])
    st.session_state[target].append(item)
    st.session_state[target_key] = len(st.session_state[target]) - 1
    clamp_indices()


def pane_controls(queue_name):
    left, right, close = st.columns(3)
    if left.button("‹", key=f"{queue_name}-left", use_container_width=True):
        shift(queue_name, -1)
        st.rerun()
    if right.button("›", key=f"{queue_name}-right", use_container_width=True):
        shift(queue_name, 1)
        st.rerun()
    if close.button("×", key=f"{queue_name}-close", use_container_width=True):
        close_current(queue_name)
        st.rerun()


def bridge_controls():
    left, right = st.columns(2)
    if left.button("输入 → 输出", use_container_width=True):
        move_current("inputs", "outputs")
        st.rerun()
    if right.button("输出 → 输入", use_container_width=True):
        move_current("outputs", "inputs")
        st.rerun()


def run_basic_operation(item):
    path = item["path"]
    op = st.sidebar.selectbox(
        "基础操作",
        ["灰度", "反色", "直方图均衡化", "旋转", "缩放", "放大 2 倍", "缩小 1/2"],
    )
    angle = st.sidebar.number_input("旋转角度", value=90, step=1, disabled=op != "旋转")
    scale = st.sidebar.number_input("缩放倍数", value=1.0, min_value=0.05, step=0.1, disabled=op != "缩放")
    if not st.sidebar.button("执行基础操作", type="primary"):
        return

    img = read_image(path)
    if op == "灰度":
        out = img.convert("L")
    elif op == "反色":
        out = Image.fromarray(255 - np.array(img.convert("RGB")))
    elif op == "直方图均衡化":
        out = Image.fromarray(histeq(np.array(img))[0].astype("uint8"))
    elif op == "旋转":
        out = img.rotate(angle)
    elif op == "缩放":
        out = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    elif op == "放大 2 倍":
        out = img.resize((img.width * 2, img.height * 2))
    else:
        out = img.resize((max(1, img.width // 2), max(1, img.height // 2)))
    add_output(save_pil(out), op)
    st.rerun()


def run_add_images():
    paths = image_paths("inputs")
    if len(paths) < 2:
        st.sidebar.info("至少上传两张输入图片后才能相加。")
        return
    if not st.sidebar.button("平均叠加全部输入图片", type="primary"):
        return

    images = [np.array(read_image(path, "RGB")).astype("float") for path in paths]
    width = max(arr.shape[1] for arr in images)
    height = max(arr.shape[0] for arr in images)
    result = np.zeros((height, width, 3), np.float64)
    for arr in images:
        result += expand(arr, width, height)
    add_output(save_array(result / len(images)), "图片平均叠加")
    st.rerun()


def run_edge_operation(item):
    op = st.sidebar.selectbox("边缘检测算子", list(EDGE_OPERATORS))
    if not st.sidebar.button("执行边缘检测", type="primary"):
        return
    add_output(save_array(EDGE_OPERATORS[op](cv2.imread(item["path"], 0))), op)
    st.rerun()


def run_fft_operation(item):
    op = st.sidebar.selectbox("频域操作", ["FFT 三图", "高通滤波", "低通滤波", "彩色高通滤波", "彩色低通滤波"])
    radius = st.sidebar.slider("滤波中心半径", 1, 300, 50, disabled=op == "FFT 三图")
    if not st.sidebar.button("执行频域操作", type="primary"):
        return

    if op == "FFT 三图":
        img = cv2.imread(item["path"], 0)
        shifted = fftshift(fft2(img))
        spectrum = np.log(np.abs(shifted) + 1)
        spectrum = 255 / spectrum.max() * spectrum if spectrum.max() else spectrum
        inverse = np.abs(ifft2(ifftshift(shifted)))
        add_output(save_array(shifted), "FFT 复数频谱")
        add_output(save_array(spectrum), "FFT 动态压缩频谱")
        add_output(save_array(inverse), "FFT 逆变换")
    else:
        data = np.array(read_image(item["path"], "RGB"))
        high = "高通" in op
        if "彩色" in op:
            result = np.dstack([smoothing(data[:, :, i], radius, high) for i in range(3)])
        else:
            result = smoothing(np.mean(data, axis=2), radius, high)
        add_output(save_array(result), op)
    st.rerun()


def run_extraction(item):
    op = st.sidebar.selectbox("曲线提取步骤", ["方框裁剪", "按颜色分离", "顺序选点并插值", "水平分割"])

    if op == "方框裁剪" and st.sidebar.button("执行裁剪", type="primary"):
        add_output(save_array(Tailor(item["path"]).tailor), "方框裁剪")
        st.rerun()

    if op == "按颜色分离":
        line_num = st.sidebar.number_input("最大线条颜色数", value=6, min_value=1, step=1)
        black = st.sidebar.slider("黑色阈值", 0, 255, 50)
        white = st.sidebar.slider("白色阈值", 0, 255, 235)
        if st.sidebar.button("执行颜色分离", type="primary"):
            source = item["path"]
            data = np.array(read_image(source, "RGB"))
            if np.mean(data) > 200:
                source = save_array(255 - data)
            sep = Seperator(source, line_num=int(line_num), b=black, w=white)
            for index, img in enumerate(sep.color_img, start=1):
                add_output(save_array(np.array(img)), f"颜色分离 {index}")
            st.rerun()

    if op == "顺序选点并插值":
        xs, xt = st.sidebar.number_input("X 起点", value=0.0), st.sidebar.number_input("X 终点", value=100.0)
        ys, yt = st.sidebar.number_input("Y 起点", value=0.0), st.sidebar.number_input("Y 终点", value=100.0)
        z = st.sidebar.number_input("最小取点距", value=5.0, min_value=0.0)
        zm = st.sidebar.number_input("离群点距离下限", value=20.0, min_value=0.0)
        if st.sidebar.button("提取数据点", type="primary"):
            ex = Extractor(item["path"], xs, xt, ys, yt, z, zm)
            add_output(ex.interpolate(), "曲线插值图")
            add_output(ex.csv_path, "曲线坐标 CSV")
            st.rerun()

    if op == "水平分割":
        st.sidebar.caption("水平分割会处理输入区全部图片。")
        if st.sidebar.button("执行全部输入的水平分割", type="primary"):
            paths = image_paths("inputs")
            if not paths:
                st.sidebar.warning("输入区没有可分割的图片。")
                return
            for source in paths:
                split = Split(source)
                add_output(split.visualize, f"水平分割可视化 {Path(source).name}")
                for path in get_folder(split.output):
                    add_output(path, f"分割结果 {Path(source).stem}-{Path(path).name}")
            st.rerun()


def run_utilities(item):
    op = st.sidebar.selectbox("实用工具", ["图片转 PDF", "图片转 GIF", "去除灰色水印", "合成大图"])
    paths = image_paths("inputs")

    if op == "图片转 PDF" and st.sidebar.button("生成全部输入 PDF", type="primary"):
        if paths:
            add_output(pic2pdf(paths), "图片转 PDF")
            st.rerun()
        st.sidebar.warning("输入区没有可用图片。")

    if op == "图片转 GIF":
        fps = st.sidebar.number_input("FPS", value=1, min_value=1, step=1)
        if st.sidebar.button("生成全部输入 GIF", type="primary"):
            if paths:
                add_output(pic2gif(paths, fps=fps), "图片转 GIF")
                st.rerun()
            st.sidebar.warning("输入区没有可用图片。")

    if op == "去除灰色水印":
        thresh = st.sidebar.slider("灰色阈值", 0, 255, 150)
        if st.sidebar.button("去除当前输入水印", type="primary"):
            add_output(remove_gray(item["path"], thresh), "去除灰色水印")
            st.rerun()

    if op == "合成大图":
        line_max = st.sidebar.number_input("每行图片数", value=2, min_value=1, step=1)
        if st.sidebar.button("合成全部输入图片", type="primary"):
            if paths:
                add_output(concat(paths, int(line_max)), "合成大图")
                st.rerun()
            st.sidebar.warning("输入区没有可用图片。")


def render_sidebar(item):
    st.sidebar.header("文件")
    pdf_dpi = st.sidebar.number_input("PDF 转图片 DPI", value=144, min_value=72, max_value=400, step=12)
    uploads = st.sidebar.file_uploader(
        "上传图片或 PDF",
        type=["png", "jpg", "jpeg", "bmp", "pdf"],
        accept_multiple_files=True,
    )
    if uploads and st.sidebar.button("加入输入区"):
        ingest_uploads(uploads, pdf_dpi)
        st.rerun()

    if st.sidebar.button("清空输入输出队列"):
        st.session_state.inputs = []
        st.session_state.outputs = []
        st.session_state.input_index = 0
        st.session_state.output_index = 0
        st.rerun()

    st.sidebar.divider()
    if not item:
        st.sidebar.info("先上传一张图片或 PDF。")
        return

    mode = st.sidebar.radio("功能", ["基础操作", "图片相加", "FFT/滤波", "边缘检测", "曲线提取", "实用工具"])
    if mode == "基础操作":
        run_basic_operation(item)
    elif mode == "图片相加":
        run_add_images()
    elif mode == "FFT/滤波":
        run_fft_operation(item)
    elif mode == "边缘检测":
        run_edge_operation(item)
    elif mode == "曲线提取":
        run_extraction(item)
    else:
        run_utilities(item)


def queue_title(queue_name):
    items = st.session_state[queue_name]
    index_key = "input_index" if queue_name == "inputs" else "output_index"
    if not items:
        return "0/0"
    return f"{st.session_state[index_key] + 1}/{len(items)}"


def main():
    st.set_page_config(page_title="DigitalImage", page_icon="DI", layout="wide")
    init_state()
    clamp_indices()

    input_item = current_item("inputs")
    render_sidebar(input_item)

    st.title("DigitalImage")
    st.caption("Streamlit 双窗格版。上传与输出文件均保存在项目 data/ 目录下。")

    left, right = st.columns(2)
    with left:
        st.subheader(f"输入区 {queue_title('inputs')}")
        display_current(current_item("inputs"), "inputs")
        pane_controls("inputs")

    with right:
        st.subheader(f"输出区 {queue_title('outputs')}")
        display_current(current_item("outputs"), "outputs")
        pane_controls("outputs")

    bridge_controls()


if __name__ == "__main__":
    main()
