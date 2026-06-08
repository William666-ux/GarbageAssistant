# 智能垃圾分类助手

## 项目介绍

本项目实现了一个面向日常生活场景的智能垃圾分类辅助系统，最终集成三大功能：垃圾投放分析、垃圾分类问答、垃圾图片识别。

* 垃圾投放分析：用户输入垃圾名称，并选择是否有食物或液体残留、是否潮湿、是否含有有害成分、是否具有回收价值等属性，系统结合本地垃圾分类知识库进行综合判断，输出垃圾类别、判断依据、投放建议和注意事项。
* 垃圾分类问答：系统基于本地 `database` 文件夹中的垃圾分类知识库文档，构建 Chroma 向量数据库，并结合 RAG 检索增强问答机制，实现垃圾分类相关问题的智能回答。
* 垃圾图片识别：用户上传垃圾图片后，系统调用训练好的 YOLO 目标检测模型，对图片中的垃圾目标进行自动识别，输出检测框、垃圾类别、置信度和投放建议。
* 系统交互界面：项目基于 Streamlit 构建 Web 端交互页面，支持首页展示、侧边栏导航、文本输入、图片上传和结果可视化展示。

\---

## 项目结构

```text
Garbage assistant/
├── .streamlit/
│   └── secrets.toml                  # API Key 配置文件（需用户自配）
├── database/
│   └── 垃圾分类知识库.docx             # 本地垃圾分类知识库
├── utils/
│   └── garbage\_yolov8m\_best.pt        # 训练好的 YOLO 垃圾检测模型
├── assets/
│   └── garbage\_classification.png     # 首页展示图片，可选
├── main.py                            # Streamlit 主程序
├── yolo\_predict.py                    # YOLO 图片识别模块
├── requirements.txt                   # 项目依赖
└── README.md                          # 项目说明文档
```

如果不需要重新训练模型，可以不保留训练数据集和训练脚本。若需要复现实验训练过程，可另外保留：

```text
GarbageImageDataset/
train\_yolo.py
```

\---

## 使用方法

### 1. 安装项目依赖

安装依赖：

```bash
pip install -r requirements.txt

```

如果安装后出现 `numpy` 版本冲突，建议执行：

```bash
pip install numpy==1.26.4 --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
```

\---

### 2. 配置 DeepSeek API Key

本项目当前使用 DeepSeek 作为问答模型。请在项目根目录下创建 `.streamlit/secrets.toml` 文件。

文件路径：

```text
Garbage assistant/.streamlit/secrets.toml
```

文件内容：

```toml
DEEPSEEK\_API\_KEY = "你的 DeepSeek API Key"
```

注意：如果不配置 API Key，YOLO 图片识别功能仍可使用，但垃圾投放分析、垃圾分类问答和 RAG 详细建议功能将无法正常调用大语言模型。

\---

### 3. 训练 YOLO 图像识别模型

`python train\_yolo.py`

将训练好的 YOLO 模型文件放入 `utils` 文件夹：

```text
utils/garbage\_yolov8m\_best.pt
```

并确保 `yolo\_predict.py` 中的模型路径与实际文件名一致，例如：

```python
MODEL\_PATH = Path("utils/garbage\_yolov8m\_best.pt")
```

\---

### 4. 运行项目

在项目根目录运行：

```bash
streamlit run main.py
```

或：

```bash
python -m streamlit run main.py
```

浏览器会自动打开本地页面，一般地址为：

```text
http://localhost:8501
```

\---

## 功能说明

### 垃圾投放分析

输入垃圾名称，并根据实际情况选择垃圾属性：

* 是否有食物或液体残留
* 是否潮湿或含水
* 是否可以清洗干净
* 是否含有有害成分
* 是否具有回收价值
* 是否可以重复利用

系统会结合本地知识库给出：

* 垃圾类别
* 判断理由
* 投放建议
* 注意事项

\---

### 垃圾分类问答

用户可以直接输入垃圾分类相关问题，例如：

```text
废电池应该怎么扔？
奶茶杯属于什么垃圾？
废弃电脑属于可回收物吗？
湿纸巾属于厨余垃圾吗？
```

系统会基于本地知识库进行检索，并调用大语言模型生成回答。

\---

### 垃圾图片识别

用户上传 `.jpg`、`.jpeg` 或 `.png` 格式图片后，系统会调用 YOLO 模型进行目标检测。

输出结果包括：

* 原始图片
* YOLO 检测结果图
* 检测类别
* 中文名称
* 垃圾大类
* 置信度
* 投放建议

当前模型支持的主要类别包括：

```text
BIODEGRADABLE
CARDBOARD
GLASS
METAL
PAPER
PLASTIC
```

\---

## YOLO 模型说明

本项目使用 YOLO 目标检测模型完成垃圾图片识别。训练数据集包含可降解垃圾、纸板、玻璃、金属、纸张和塑料等类别。

当前推荐使用训练后的 YOLOv8m 模型：

```text
utils/garbage\_yolov8m\_best.pt
```

模型训练完成后，通常会生成以下结果文件：

```text
weights/best.pt
results.png
confusion\_matrix.png
confusion\_matrix\_normalized.png
BoxF1\_curve.png
BoxPR\_curve.png
```

其中，系统运行时只需要使用：

```text
weights/best.pt
```

并将其复制或重命名到 `utils` 文件夹中。

\---

## 注意事项

1. `secrets.toml` 中的 API Key 不要上传到公开仓库。
2. 如果 RAG 问答无法运行，请检查：

   * `DEEPSEEK\_API\_KEY` 是否正确配置；
   * `database` 文件夹中是否存在 `.docx` 知识库；
   * 网络连接是否正常；
   * 依赖包是否安装完整。
3. 如果图片识别失败，请检查：

   * `utils` 文件夹中是否存在 YOLO 模型文件；
   * `yolo\_predict.py` 中的 `MODEL\_PATH` 是否与模型文件名一致；
   * 是否安装 `ultralytics` 和 `opencv-python`。

\---

## 项目运行流程

```text
启动 Streamlit 页面
        ↓
选择功能模块
        ↓
垃圾投放分析 / 垃圾分类问答 / 垃圾图片识别
        ↓
本地知识库检索 或 YOLO 图像检测
        ↓
输出垃圾类别、判断依据和投放建议
```

\---

## 项目特点

* 支持文本输入、问答查询和图片上传三种交互方式；
* 结合本地知识库进行垃圾分类知识检索；
* 集成 YOLO 目标检测模型，实现垃圾图片自动识别；
* 页面简洁，适合课程设计、项目展示和本地部署运行；
* 模块相对独立，便于后续替换大模型、扩展知识库或更新检测模型。

