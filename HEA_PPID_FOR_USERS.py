'''
Runs the streamlit app
Call this file in the terminal via `streamlit run app.py`
'''
import streamlit as st

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from streamlit_shap import st_shap
from streamlit_card import card

from utils import *

from prettytable import PrettyTable

import sqlite3
import os

import requests
from openai import OpenAI

import pandas as pd


# import os
# import streamlit as st
# from streamlit_option_menu import option_menu
# from streamlit_extras.colored_header import colored_header
# from prettytable import PrettyTable




#         page_title="MLMD",
#         page_icon="🍁",
#         layout="centered",
#         initial_sidebar_state="auto",
#         menu_items={
#         })

# sysmenu = '''
# <style>
# MainMenu {visibility:hidden;}
# footer {visibility:hidden;}
# '''

# # https://icons.bootcss.com/
# st.markdown(sysmenu,unsafe_allow_html=True)
DEFAULT_STORAGE_PATH = "data"


def HEA_PPID():
    with st.sidebar:
        select_option = option_menu("", ["Home Page", "Model Inference", "Chat with Model"],

                                    menu_icon="boxes", default_index=0)
        st.write('''
            **Contact**: 



    ''')
        # 此函数为定义主页面的选择框后续为选择框的各个内容。

    if select_option == "Home Page":
        colored_header(label="Model Inference", description=" ", color_name="violet-90")

        # 项目简介
        st.subheader("项目简介")
        st.write("""
        我们构建了一个高熵合金网络预测平台，旨在为无人工智能基础的材料研究者提供便捷的工具和模型。该平台基于最新的机器学习算法，能够准确预测高熵合金的性能，推动合金材料的研发和应用。
        """)

        # 使用的算法
        st.subheader("使用的算法")
        st.write("""
        - **支持向量机 (SVM)**
        - **TabPFN回归**
        - **XGBoost**
        """)

        # 项目成员
        st.subheader("项目成员")
        st.write("""
        - **项目组长：** 周天毅
        - **组员：** （具体成员待定）
        - **指导老师：** 熊杰
        """)

        # 联系方式
        st.subheader("联系方式")
        st.write("请通过以下方式联系项目团队：")
        st.write("Email: example@example.com")

    elif select_option == "Model Inference":

        colored_header(label="Model Inference", description=" ", color_name="violet-90")

        model_source = st.selectbox('Select Function Block', [

            '[1] Use internal model (Here, select to call internal model)',

            '[2] Upload your own model (Please upload your model here)'

        ])

        if model_source == '[1] Use internal model (Here, select to call internal model)':

            # ======== 先自动检索模型，并始终显示模型选择框 ========
            model_dir = 'model_done'
            model_files = [f for f in os.listdir('model_done') if f.endswith(('.pkl', '.pickle'))]

            st.subheader("Select Internal Model")
            if model_files:
                model_selected = st.selectbox('Select internal model to use', model_files)
                model_path = os.path.join(model_dir, model_selected)

                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    # 检查模型是否有 n_features_in_
                    if hasattr(model, 'n_features_in_'):
                        n_model_features = model.n_features_in_
                        st.info(f"ℹ️ This model expects **{n_model_features} features** as input.")
                    else:
                        st.warning("⚠️ This model does not have `n_features_in_` attribute.")

                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    model = None
                    n_model_features = None


            else:
                st.warning('No models found in model_done/ folder.')
                model_selected = None
                model_path = None

            st.markdown("### Manually Input Metals & Ratios")

            num_metals = st.number_input("Number of metals", min_value=1, max_value=20, value=3, step=1)

            metal_names = []
            metal_ratios = []

            st.markdown("### Enter Metal Names and Their Ratios")

            # 初始化
            num_metals = 10
            metal_names = []
            metal_ratios = []
            ratio_total = 0.0
            input_valid = True

            # 固定金属元素
            fixed_metals = ['Fe', 'Cu', 'Ni', 'Cr', 'Mn', 'Mo', 'Co', 'Zn']
            num_metals = len(fixed_metals)

            metal_ratios = []
            input_valid = True

            # 表格输入，每列一个金属 + 1列显示总和
            cols = st.columns(num_metals + 1)

            # 第一行：展示金属名称（不可编辑）
            for i in range(num_metals):
                with cols[i]:
                    st.markdown(f"**{fixed_metals[i]}**")

            # 第二行：输入比例
            for i in range(num_metals):
                with cols[i]:
                    ratio = st.number_input(
                        "Ratio (%)", min_value=0.0, max_value=100.0, step=0.01, key=f"metal_ratio_{i}"
                    )
                    metal_ratios.append(ratio)

            # 最后一列：显示总比例
            with cols[-1]:
                ratio_total = round(sum(metal_ratios), 2)
                st.markdown("**Total Ratio**")
                st.write(f"{ratio_total} %")

            # 检查输入合法性
            if any(r < 0 or r > 100 for r in metal_ratios) or ratio_total != 100.0:
                input_valid = False
                st.error("Invalid input: All ratios must be between 0 and 100, and the total must equal 100.00%.")
            else:
                input_valid = True
                st.success("Valid input! Total ratio is 100.00%")

            # 表单用于支持禁用按钮
            with st.form(key="manual_input_form"):
                submit_button = st.form_submit_button(
                    label="Generate manual CSV and use for prediction",
                    disabled=not input_valid
                )

            if submit_button:
                df_manual = pd.DataFrame([metal_ratios], columns=fixed_metals)
                st.write("Generated Data:")
                st.write(df_manual)

                tmp_download_link = download_button(df_manual, "manual_input.csv", button_text="Download CSV")
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                df_ready = df_manual
                st.success("Manual input ready to use!")

            # ======== 后续流程（统一用 df_ready，不管是上传 or 手动填写） ========
            if 'df_ready' in locals():
                df = df_ready
                check_string_NaN(df)

                colored_header(label="Data information", description=" ", color_name="violet-70")
                nrow = st.slider("rows", 1, len(df), 5)
                st.write(df.head(nrow))

                colored_header(label="Feature and target", description=" ", color_name="violet-70")

                target_num = st.number_input('target number', min_value=1, max_value=10, value=1)

                col_feature, col_target = st.columns(2)

                features = df.iloc[:, :-target_num]
                targets = df.iloc[:, -target_num:]

                with col_feature:
                    st.write(features.head())

                with col_target:
                    st.write(targets.head())

                colored_header(label="target", description=" ", color_name="violet-70")

                target_selected_option = st.selectbox('target', list(targets)[::-1])
                targets = targets[target_selected_option]

                preprocess = st.selectbox('data preprocess', [None, 'StandardScaler', 'MinMaxScaler'])

                if preprocess == 'StandardScaler':
                    features = StandardScaler().fit_transform(features)
                elif preprocess == 'MinMaxScaler':
                    features = MinMaxScaler().fit_transform(features)

                # ======== 预测部分，点击按钮触发 ========
                st.subheader("Run Prediction")

                if model_selected and st.button("Run Prediction"):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)

                        prediction = model.predict(features)

                        plot = customPlot()
                        plot.pred_vs_actual(targets, prediction)

                        r2 = r2_score(targets, prediction)
                        st.write('R2: {}'.format(r2))

                        result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
                        result_data.columns = ['actual', 'prediction']

                        with st.expander('prediction'):
                            st.write(result_data)

                            tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

                        st.write('---')

                    except FileNotFoundError:
                        st.error(f'Model file not found: {model_path}')




        elif model_source == '[2] Upload your own model (Please upload your model here)':

            file = st.file_uploader("Upload `.csv`file", label_visibility="collapsed", accept_multiple_files=True)

            if len(file) < 2:
                table = PrettyTable(['file name', 'class', 'description'])
                table.add_row(['file_1', 'data set (+test data)', 'data file'])
                table.add_row(['file_2', 'model', 'model'])
                st.write(table)
            elif len(file) == 2:
                df = pd.read_csv(file[0])
                model_file = file[1]

                try:
                    model = pickle.load(model_file)

                    if hasattr(model, 'n_features_in_'):
                        n_model_features = model.n_features_in_
                        st.info(f"ℹ️ This model expects **{n_model_features} features** as input.")
                    else:
                        st.warning("⚠️ This model does not have `n_features_in_` attribute.")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    model = None
                    n_model_features = None

                check_string_NaN(df)

                colored_header(label="Data information", description=" ", color_name="violet-70")
                nrow = st.slider("rows", 1, len(df), 5)
                df_nrow = df.head(nrow)
                st.write(df_nrow)

                colored_header(label="Feature and target", description=" ", color_name="violet-70")

                target_num = st.number_input('target number', min_value=1, max_value=10, value=1)

                col_feature, col_target = st.columns(2)
                # features
                features = df.iloc[:, :-target_num]
                # targets
                targets = df.iloc[:, -target_num:]
                with col_feature:
                    st.write(features.head())
                with col_target:
                    st.write(targets.head())
                colored_header(label="target", description=" ", color_name="violet-70")

                target_selected_option = st.selectbox('target', list(targets)[::-1])

                targets = targets[target_selected_option]
                preprocess = st.selectbox('data preprocess', [None, 'StandardScaler', 'MinMaxScaler'])
                if preprocess == 'StandardScaler':
                    features = StandardScaler().fit_transform(features)
                elif preprocess == 'MinMaxScaler':
                    features = MinMaxScaler().fit_transform(features)

                model = pickle.load(model_file)
                prediction = model.predict(features)
                # st.write(std)
                plot = customPlot()
                plot.pred_vs_actual(targets, prediction)
                r2 = r2_score(targets, prediction)
                st.write('R2: {}'.format(r2))
                result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
                result_data.columns = ['actual', 'prediction']
                with st.expander('prediction'):
                    st.write(result_data)
                    tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write('---')

    elif select_option == "Chat with Model":
        client = OpenAI(

            api_key="sk-214f8e7ee2e943e2a220cd0fd40058d3",  # ⚠️建议用 st.secrets 或环境变量代替明文密钥

            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

        )

        st.title("💬 Chat with 百炼 Qwen 模型")

        # 会话状态用于保存对话历史

        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        # 显示历史对话

        for msg in st.session_state.conversation:
            st.write(f"**{msg['role']}**: {msg['text']}")

        user_input = st.text_input("请输入你的问题:")

        if st.button("发送"):

            if user_input:

                # 添加用户问题到对话

                st.session_state.conversation.append({"role": "用户", "text": user_input})

                try:

                    completion = client.chat.completions.create(

                        model="qwen-plus",

                        messages=[

                            {"role": "system", "content": "You are a helpful assistant."},

                            *[

                                {"role": "user" if m["role"] == "用户" else "assistant", "content": m["text"]}

                                for m in st.session_state.conversation if m["role"] != "系统"

                            ],

                            {"role": "user", "content": user_input}

                        ]

                    )

                    reply = completion.choices[0].message.content

                except Exception as e:

                    reply = f"请求失败: {e}"

                # 添加模型回复

                st.session_state.conversation.append({"role": "模型", "text": reply})

                # 刷新显示对话

                for msg in st.session_state.conversation:
                    st.write(f"**{msg['role']}**: {msg['text']}")

            else:

                st.warning("请输入问题后再点击发送。")


if __name__ == "__main__":
    HEA_PPID()