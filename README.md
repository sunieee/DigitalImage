# DigitalImage

Streamlit Web 版数字图像处理工具。项目已经从 PyQt 桌面程序改为 Web 应用，运行入口是 `streamlit_app.py`。

## 运行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

也可以用：

```bash
make run
```

## 工作目录

所有上传和输出都放在项目内的 `data/` 目录：

- `data/uploads/`：上传的原始图片/PDF
- `data/work/`：处理后的图片、CSV、PDF、GIF、分割结果
- `data/1`、`data/2`：原项目自带示例数据

## 功能

- 基础操作：灰度、反色、直方图均衡化、旋转、缩放、图片叠加
- FFT/滤波：FFT 三图、高通/低通、彩色高通/低通
- 边缘检测：Roberts、Sobel、Prewitt、Scharr、Laplacian、LoG、Canny
- 曲线提取：方框裁剪、按颜色分离、顺序选点并插值、水平分割
- 实用工具：图片转 PDF、图片转 GIF、去除灰色水印、合成大图

交互方式继承原 PyQt 程序：输入区、输出区各只显示当前一张图，通过 `‹`、`›`、`×` 切换和关闭，通过 `输入 -> 输出`、`输出 -> 输入` 在两个区域之间移动当前图片。

水平分割入口：上传图片后，在左侧选择 `功能 -> 曲线提取`，再选择 `曲线提取步骤 -> 水平分割`。点击执行后会处理输入区全部图片。

## MCP 工具

MCP server 放在项目目录中，Codex 或 Claude Code 的项目级/全局配置指向它：

```bash
cd /home/kaititech/code/DigitalImage
python mcp_server.py
```

暴露两个工具：

- `extract_curve_points`：对单张曲线图提取数据点，输出 CSV 和插值图
- `split_horizontal_images`：对多张图片执行水平分割，输出可视化图和切分图片

Claude Code 可读取工作区根目录的 `/home/kaititech/code/.mcp.json`，配置会指向本项目内的 `mcp_server.py`。也可以手动添加：

```bash
claude mcp add digital-image --cwd /home/kaititech/code/DigitalImage -- python mcp_server.py
```

Codex 可使用工作区根目录的 `/home/kaititech/code/.codex/config.toml`，或将同样内容添加到全局 Codex 配置中：

```toml
[mcp_servers.digital_image]
command = "python"
args = ["mcp_server.py"]
cwd = "/home/kaititech/code/DigitalImage"
```
