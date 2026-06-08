import os
import streamlit as st

from PIL import Image
from yolo_predict import predict_garbage, summarize_detections
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA


# =========================
# 页面基础设置
# =========================
st.set_page_config(page_title="智能垃圾分类助手", page_icon="♻️", layout="wide")
st.header("♻️ 智能垃圾分类助手")

# =========================
# 侧边栏样式优化
# =========================
st.markdown(
    """
    <style>
    /* 侧边栏整体背景 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecfdf5 0%, #f8fafc 100%);
        border-right: 1px solid #d1fae5;
    }

    /* 侧边栏内部间距 */
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    /* 侧边栏标题 */
    section[data-testid="stSidebar"] .sidebar-title {
        font-size: 24px;
        font-weight: 850;
        color: #065f46;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    section[data-testid="stSidebar"] .sidebar-subtitle {
        font-size: 14px;
        color: #047857;
        margin-bottom: 22px;
        line-height: 1.6;
    }

    /* radio 外层 */
    div[role="radiogroup"] {
        gap: 10px;
    }

    /* 每个导航项 */
    div[role="radiogroup"] label {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:hover {
        border-color: #10b981;
        background-color: #f0fdf4;
        transform: translateX(3px);
    }

    /* radio 文字 */
    div[role="radiogroup"] label p {
        font-size: 17px;
        font-weight: 650;
        color: #111827;
    }

    /* 选中项文字颜色 */
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-color: #059669;
        box-shadow: 0 8px 18px rgba(16,185,129,0.25);
    }

    div[role="radiogroup"] label:has(input:checked) p {
        color: white;
    }

    /* 隐藏原始小圆点，让导航更像按钮 */
    div[role="radiogroup"] input[type="radio"] {
        display: none;
    }

    /* 底部提示卡片 */
    .sidebar-tip {
        margin-top: 32px;
        background-color: #ffffff;
        border: 1px solid #d1fae5;
        border-radius: 16px;
        padding: 16px;
        color: #065f46;
        font-size: 14px;
        line-height: 1.7;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# DeepSeek Key 设置
# 优先读取 Streamlit secrets；如果没有，再读取系统环境变量
# =========================
def get_deepseek_key():
    try:
        key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
    return key


DEEPSEEK_API_KEY = get_deepseek_key()
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
else:
    st.warning("未检测到 DEEPSEEK_API_KEY。请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY。")


# =========================
# 垃圾类别映射表
# 后续接入图片分类模型时，也可以复用这个映射表
# =========================
GARBAGE_MAP = {
    "塑料瓶/塑料制品": {
        "category": "可回收物",
        "suggestion": "投放前建议清空内容物，尽量压扁后投放至可回收物垃圾桶。若被严重污染，应按其他垃圾处理。"
    },
    "纸张/纸箱/纸板": {
        "category": "可回收物",
        "suggestion": "请保持干燥、干净后投放至可回收物垃圾桶。被油污污染的纸张通常不适合回收。"
    },
    "玻璃瓶/玻璃制品": {
        "category": "可回收物",
        "suggestion": "请清空内容物后投放。破碎玻璃应包裹后再投放，避免划伤他人。"
    },
    "金属罐/易拉罐": {
        "category": "可回收物",
        "suggestion": "请清空残留物后投放至可回收物垃圾桶，可适当压扁以节省空间。"
    },
    "厨余垃圾": {
        "category": "厨余垃圾",
        "suggestion": "应沥干水分后投放至厨余垃圾桶，避免混入塑料袋、餐盒、纸巾等非厨余物。"
    },
    "废电池/过期药品/灯管": {
        "category": "有害垃圾",
        "suggestion": "请投放至有害垃圾收集容器，不要随意丢弃，避免造成环境污染。"
    },
    "用过的纸巾/烟头/陶瓷碎片": {
        "category": "其他垃圾",
        "suggestion": "通常属于其他垃圾。尖锐或破碎物品建议包裹后投放，避免伤人。"
    },
    "无法判断/其他": {
        "category": "其他垃圾或需进一步判断",
        "suggestion": "建议结合是否可回收、是否有污染、是否含有有害成分等因素进一步判断。"
    }
}


# =========================
# 加载大语言模型
# =========================
def load_llm(temperature=0.3):
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com"
    )


# =========================
# 加载本地知识库，建立 RAG 问答链
# 默认读取 database 文件夹下：垃圾分类知识库*.docx
# 如果找不到，则读取 database 下所有 docx
# =========================
@st.cache_resource(show_spinner="正在加载本地垃圾分类知识库并建立向量数据库...")
def load_garbage_qa():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek 模型。")

    llm = load_llm(temperature=0.3)

    # 优先只读取垃圾分类知识库，避免把原来的健康文档也检索进去
    loader = DirectoryLoader(
        "./database",
        glob="垃圾分类知识库*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    documents = loader.load()

    # 如果没有找到垃圾分类知识库，则兜底读取 database 下所有 docx
    if len(documents) == 0:
        loader = DirectoryLoader(
            "./database",
            glob="*.docx",
            loader_cls=UnstructuredWordDocumentLoader
        )
        documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("database 文件夹中没有找到 docx 知识库文件。")

    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    # 使用本地中文 Embedding，不再依赖 OpenAI Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )
    return qa


# =========================
# 首页
# =========================
def page_home():
    st.markdown(
        """
        <style>
        .app-hero {
            background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 45%, #ffffff 100%);
            border: 1px solid #d1fae5;
            border-radius: 22px;
            padding: 34px 38px;
            margin-bottom: 30px;
        }

        .app-title {
            font-size: 46px;
            font-weight: 850;
            color: #111827;
            margin-bottom: 14px;
        }

        .app-subtitle {
            font-size: 20px;
            color: #374151;
            line-height: 1.8;
            margin-bottom: 22px;
        }

        .app-slogan {
            display: inline-block;
            background-color: #10b981;
            color: white;
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 15px;
            font-weight: 650;
        }

        .feature-card {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 28px 26px;
            min-height: 230px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.07);
        }

        .feature-icon {
            font-size: 42px;
            margin-bottom: 12px;
        }

        .feature-title {
            font-size: 25px;
            font-weight: 800;
            color: #111827;
            margin-bottom: 10px;
        }

        .feature-desc {
            font-size: 16px;
            color: #4b5563;
            line-height: 1.8;
        }

        .section-title {
            font-size: 30px;
            font-weight: 800;
            color: #111827;
            margin-top: 34px;
            margin-bottom: 18px;
        }

        .guide-box {
            background-color: #f9fafb;
            border-radius: 16px;
            padding: 24px 28px;
            border: 1px solid #e5e7eb;
            line-height: 2;
            font-size: 16px;
            color: #374151;
        }

        .notice-box {
            background-color: #eff6ff;
            border-left: 5px solid #3b82f6;
            border-radius: 12px;
            padding: 18px 22px;
            margin-top: 24px;
            color: #1e40af;
            font-size: 15px;
            line-height: 1.8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 首页顶部：左文字，右图片
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            """
            <div class="app-hero">
                <div class="app-title">♻️ 智能垃圾分类助手</div>
                <div class="app-subtitle">
                    不确定垃圾该怎么扔？输入名称、提出问题，或上传图片，
                    系统将帮助你快速判断垃圾类别，并给出清晰的投放建议。
                </div>
                <div class="app-slogan">让垃圾分类更简单、更准确、更方便</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        try:
            st.image(
                "assets/garbage_classification.png",
                caption="垃圾分类示意图",
                width=330
            )
        except Exception:
            st.markdown(
                """
                <div class="app-hero" style="text-align:center;">
                    <div style="font-size:72px;">🗑️♻️</div>
                    <div style="font-size:18px;color:#047857;font-weight:700;">
                        绿色生活，从正确分类开始
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 三个功能入口
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🗑️</div>
                <div class="feature-title">垃圾投放分析</div>
                <div class="feature-desc">
                    输入垃圾名称，并选择是否有残留、是否有害、是否可回收等属性。
                    系统会综合判断垃圾类别，并告诉你应该如何投放。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">垃圾分类问答</div>
                <div class="feature-desc">
                    不知道某个物品属于哪类垃圾？直接输入问题，
                    系统会根据垃圾分类知识库给出解释和建议。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📷</div>
                <div class="feature-title">垃圾图片识别</div>
                <div class="feature-desc">
                    上传垃圾图片，系统会自动识别图片中的垃圾目标，
                    显示识别结果、置信度和对应的投放建议。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 使用指南
    st.markdown('<div class="section-title">你可以这样使用</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="guide-box">
            <b>① 知道垃圾名称：</b> 进入“垃圾投放分析”，输入垃圾名称并选择属性，获取分类结果。<br>
            <b>② 想咨询分类问题：</b> 进入“垃圾分类问答”，直接提问，例如“废电池怎么扔？”“奶茶杯属于什么垃圾？”<br>
            <b>③ 有垃圾图片：</b> 进入“垃圾图片识别”，上传图片后自动识别垃圾类别。<br>
            <b>④ 结果不确定：</b> 可以结合图片识别结果和文字问答结果进行二次确认。
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice-box">
            <b>温馨提示：</b>
            不同城市的垃圾分类标准可能存在细微差异。对于电池、灯管、药品、电子废弃物等特殊物品，
            建议优先按照当地社区或回收点要求进行投放。
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 模块一：垃圾投放分析
# =========================
def page_analysis():
    st.title("垃圾投放分析")
    st.markdown("请输入垃圾物品的基本属性，系统将结合本地知识库分析其分类和投放方式。")

    garbage_name = st.text_input("垃圾名称", placeholder="例如：奶茶杯、废电池、香蕉皮、外卖餐盒")

    col1, col2 = st.columns(2)
    with col1:
        has_residue = st.selectbox("是否有食物或液体残留？", ["不确定", "是", "否"])
        is_wet = st.selectbox("是否潮湿或含水？", ["不确定", "是", "否"])
        can_clean = st.selectbox("是否可以清洗干净？", ["不确定", "是", "否"])
    with col2:
        has_hazard = st.selectbox("是否含有有害成分？", ["不确定", "是", "否"])
        recyclable = st.selectbox("是否具有回收价值？", ["不确定", "是", "否"])
        reusable = st.selectbox("是否可以重复利用？", ["不确定", "是", "否"])

    if st.button("开始分析"):
        if not garbage_name.strip():
            st.warning("请先输入垃圾名称。")
            return

        query = f"""
        请根据本地垃圾分类知识库，对以下垃圾进行分类分析，并输出：
        1. 判断类别：可回收物、厨余垃圾、有害垃圾、其他垃圾之一；
        2. 判断理由；
        3. 投放建议；
        4. 注意事项。

        回答要求：
        - 请使用简洁、准确、适合普通用户理解的语言。
        - 如果涉及电脑、手机、充电宝、家电、电路板等电子废弃物，不要建议用户自行拆解。
        - 对于电子废弃物，应建议保持完整，优先交给正规电子废弃物回收点、社区回收点或可回收物专项回收渠道处理。
        - 如果物品含有电池、灯管、药品、化学品等有害成分，应提醒用户单独按有害垃圾或当地专项回收要求处理。
        - 如果不同城市分类标准可能不同，请提醒以当地规定为准。

        垃圾名称：{garbage_name}
        是否有食物或液体残留：{has_residue}
        是否潮湿或含水：{is_wet}
        是否可以清洗干净：{can_clean}
        是否含有有害成分：{has_hazard}
        是否具有回收价值：{recyclable}
        是否可以重复利用：{reusable}
        """

        try:
            qa = load_garbage_qa()
            result = qa.run(query)
            st.markdown("### 分析结果")
            st.write(result)
        except Exception as e:
            st.error(f"分析失败：{e}")
            st.write("请检查：1）DEEPSEEK_API_KEY 是否配置；2）database 文件夹中是否有垃圾分类知识库 docx；3）网络/VPN 是否正常。")


# =========================
# 模块二：垃圾分类问答
# =========================
def page_query():
    st.title("垃圾分类问答")
    st.markdown("请输入垃圾分类相关问题，系统会基于本地知识库进行 RAG 检索并回答。")

    question = st.text_area(
        "请输入你的问题",
        placeholder="例如：奶茶杯属于什么垃圾？废电池怎么扔？湿纸巾属于厨余垃圾吗？外卖餐盒可以回收吗？",
        height=120
    )

    if st.button("开始查询"):
        if not question.strip():
            st.warning("请输入问题后再查询。")
            return

        try:
            qa = load_garbage_qa()

            safe_question = f"""
            请基于本地垃圾分类知识库回答用户问题。

            回答要求：
            - 请使用简洁、准确、适合普通用户理解的语言。
            - 如果涉及电脑、手机、充电宝、家电、电路板等电子废弃物，不要建议用户自行拆解。
            - 对于电子废弃物，应建议保持完整，优先交给正规电子废弃物回收点、社区回收点或可回收物专项回收渠道处理。
            - 如果物品含有电池、灯管、药品、化学品等有害成分，应提醒用户单独按有害垃圾或当地专项回收要求处理。
            - 如果不同城市分类标准可能不同，请提醒以当地规定为准。

            用户问题：
            {question}
            """

            result = qa.run(safe_question)
            st.markdown("### 回答结果")
            st.write(result)
        except Exception as e:
            st.error(f"查询失败：{e}")
            st.write(
                "请检查：1）DEEPSEEK_API_KEY 是否配置；2）database 文件夹中是否有垃圾分类知识库 docx；3）网络/VPN 是否正常。")


# =========================
# 模块三：垃圾图片识别
# 第一版：先支持上传图片和手动类别选择
# 第二版：后续用 garbage_predict.py 接入训练好的图像分类模型
# =========================
def page_image():
    st.title("垃圾图片识别 / YOLO目标检测")
    st.markdown("上传一张垃圾图片，系统将使用 YOLO 模型自动检测垃圾目标，并给出分类结果和投放建议。")

    uploaded_image = st.file_uploader(
        "请选择垃圾图片",
        type=["jpg", "jpeg", "png"]
    )

    conf_threshold = st.slider(
        "检测置信度阈值",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    if uploaded_image is None:
        st.info("请先上传一张 jpg、jpeg 或 png 格式的图片。")
        return

    image = Image.open(uploaded_image).convert("RGB")

    st.markdown("### 原始图片")
    st.image(image, caption="原始图片", width=500)

    if st.button("开始检测"):
        try:
            with st.spinner("正在使用 YOLO 模型检测垃圾目标..."):
                annotated_image, detections = predict_garbage(
                    image=image,
                    conf_threshold=conf_threshold
                )

            st.markdown("### 检测结果图")
            st.image(annotated_image, caption="YOLO检测结果", width=500)

            st.markdown("### 检测结果")
            if len(detections) == 0:
                st.warning("未检测到明显垃圾目标。可以尝试降低置信度阈值，或换一张更清晰的图片。")
            else:
                for i, det in enumerate(detections, start=1):
                    st.markdown(f"#### 目标 {i}")
                    st.write(f"检测类别：{det['class_name']}")
                    st.write(f"中文名称：{det['cn_name']}")
                    st.write(f"垃圾大类：{det['garbage_type']}")
                    st.write(f"置信度：{det['confidence']}")
                    st.write(f"投放建议：{det['suggestion']}")

                summary = summarize_detections(detections)

                st.markdown("### 结果汇总")
                st.text(summary)

                if st.button("基于本地知识库生成详细投放建议"):
                    try:
                        qa = load_garbage_qa()
                        query = (
                            "请根据垃圾分类知识库，对以下 YOLO 检测结果给出详细投放建议，"
                            "包括垃圾大类、判断依据、投放前处理方式和注意事项。\n\n"
                            "回答要求：\n"
                            "- 请使用简洁、准确、适合普通用户理解的语言。\n"
                            "- 如果涉及电脑、手机、充电宝、家电、电路板等电子废弃物，不要建议用户自行拆解。\n"
                            "- 对于电子废弃物，应建议保持完整，优先交给正规电子废弃物回收点、社区回收点或可回收物专项回收渠道处理。\n"
                            "- 如果物品含有电池、灯管、药品、化学品等有害成分，应提醒用户单独按有害垃圾或当地专项回收要求处理。\n"
                            "- 如果不同城市分类标准可能不同，请提醒以当地规定为准。\n\n"
                            f"YOLO 检测结果：\n{summary}"
                        )
                        result = qa.run(query)
                        st.markdown("### 本地知识库 RAG 建议")
                        st.write(result)
                    except Exception as e:
                        st.error(f"RAG 生成建议失败：{e}")

        except Exception as e:
            st.error(f"检测失败：{e}")


# =========================
# 侧边栏导航
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">♻️ 导航栏</div>
        <div class="sidebar-subtitle">
            选择你需要的功能模块，快速完成垃圾分类查询、分析和图片识别。
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        label="",
        options=[
            "🏠 首页",
            "🗑️ 垃圾投放分析",
            "💬 垃圾分类问答",
            "📷 垃圾图片识别"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-tip">
            <b>使用提示</b><br>
            不确定垃圾类别时，可以先用文字问答；有图片时，再结合图片识别结果进行判断。
        </div>
        """,
        unsafe_allow_html=True
    )

if page == "🏠 首页":
    page_home()
elif page == "🗑️ 垃圾投放分析":
    page_analysis()
elif page == "💬 垃圾分类问答":
    page_query()
elif page == "📷 垃圾图片识别":
    page_image()
