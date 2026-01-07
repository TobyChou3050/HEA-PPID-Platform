# utils1.py
import pandas as pd
import pickle
import base64
import uuid
import re
import streamlit as st



# ======================= 数据检查 =======================
def check_nan(df: pd.DataFrame):
    """检查 DataFrame 是否有 NaN 值"""
    null_columns = df.columns[df.isnull().any()]
    if len(null_columns) > 0:
        st.error(f"Error: NaN in column {list(null_columns)} !")
        st.stop()


# ======================= 下载按钮 =======================
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    创建下载按钮

    object_to_download: 可以是 pd.DataFrame, bytes 或其他对象
    download_filename: 文件名
    button_text: 按钮显示文本
    pickle_it: 是否先 pickle 化对象
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None
    else:
        if isinstance(object_to_download, bytes):
            pass
        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
        else:
            object_to_download = pickle.dumps(object_to_download)

    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    return f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}" target="_blank">{button_text}</a>'


# ======================= 模型加载 =======================
def load_model(model_path: str):
    """从 pickle 文件加载模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ======================= 可选：自定义绘图类 =======================
class customPlot:
    """用于网页显示预测 vs 实际的简单绘图"""

    def pred_vs_actual(self, y_true, y_pred):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_true, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Prediction")
        ax.set_title("Prediction vs Actual")
        st.pyplot(fig)
