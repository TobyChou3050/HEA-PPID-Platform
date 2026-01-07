'''
Runs the streamlit app
Call this file in the terminal via `streamlit run app.py`
'''
import streamlit as st

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from streamlit_card import card

# from utils import *
from utils1 import check_nan, download_button, load_model

from prettytable import PrettyTable

import sqlite3
import os

import requests
from openai import OpenAI

import pandas as pd

import joblib

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


        # ===============================

        # 1️⃣ 自动加载模型

        # ===============================

        model_folder = r'G:\HEA_PPID\model_done'

        # 找到所有 .pkl 模型（排除 scaler/pca）

        model_files = [f for f in os.listdir(model_folder)

                       if f.endswith('.pkl') and 'scaler' not in f and 'pca' not in f]

        if not model_files:

            st.error("No model file found in model_done!")

            model = None

        else:

            # 下拉选择模型

            selected_model_name = st.selectbox("Select Model", model_files)

            try:

                model = joblib.load(os.path.join(model_folder, selected_model_name))

                st.success(f"Loaded model: {selected_model_name}")

            except Exception as e:

                st.error(f"Failed to load model: {e}")

                model = None

        # ===============================

        # 2️⃣ 展示标题

        # ===============================

        colored_header(label="Model Inference", description="Enter metal composition to predict TS", color_name="violet-90")

        # ===============================

        # 3️⃣ 用户输入金属比例（22 个元素）

        # ===============================

        FEATURES = [
            'Co', 'Cr', 'Fe', 'Ni', 'Mn', 'Nb', 'Al', 'Ti', 'C',
            'Mo', 'Si', 'Cu', 'V', 'Y', 'Sn', 'Li', 'Mg', 'Zn',
            'Ta', 'Zr', 'Hf', 'W'
        ]

        if 'metal_ratios' not in st.session_state:
            st.session_state['metal_ratios'] = [0.0] * len(FEATURES)

        # 分两行显示，每行 11 列
        first_row_features = FEATURES[:11]
        second_row_features = FEATURES[11:]

        # 第一行
        cols1 = st.columns(11)
        for i, feature in enumerate(first_row_features):
            with cols1[i]:
                st.session_state['metal_ratios'][i] = st.number_input(
                    feature, min_value=0.0, max_value=100.0, step=0.01,
                    key=f"metal_ratio_{i}"
                )

        # 第二行
        cols2 = st.columns(11)
        for j, feature in enumerate(second_row_features):
            with cols2[j]:
                st.session_state['metal_ratios'][j + 11] = st.number_input(
                    feature, min_value=0.0, max_value=100.0, step=0.01,
                    key=f"metal_ratio_{j + 11}"
                )

        # 总和显示
        ratio_total = round(sum(st.session_state['metal_ratios']), 2)
        st.markdown(f"**Total Ratio = {ratio_total} %**")

        # ===============================

        # 4️⃣ 验证按钮 + 预测

        # ===============================

        if st.button("Validate & Predict TS"):

            if model is None:

                st.error("No model loaded!")

            elif ratio_total != 100.0:

                st.error("Error: Total ratio must equal 100%")

            else:

                # 准备完整 DataFrame

                input_df = pd.DataFrame([st.session_state['metal_ratios']], columns=FEATURES)

                # 展示输入

                st.subheader("Input Composition")

                st.dataframe(input_df)

                try:

                    # 使用 Pipeline 预测（包含 scaler + PCA + SVR）

                    predicted_ts = model.predict(input_df)[0]

                    st.subheader("Prediction Result")

                    st.write(f"Predicted TS (MPa): {predicted_ts:.2f}")


                except Exception as e:

                    st.error(f"Prediction failed: {e}")

            # # 最后一列：显示总比例
            # with cols[-1]:
            #     ratio_total = round(sum(metal_ratios), 2)
            #     st.markdown("**Total Ratio**")
            #     st.write(f"{ratio_total} %")
            #
            # if 'df_ready' in locals():
            #     df = df_ready  # 用户手动生成的输入数据
            #
            #     # 检查 NaN
            #     if df.isnull().any().any():
            #         st.error("Error: Input contains NaN values!")
            #         st.stop()
            #
            #     # 显示输入数据
            #     st.subheader("Input Data")
            #     st.write(df)
            #
            #     # ====== 数据预处理 ======
            #     # 这里直接用训练时保存的 scaler 和 pca
            #     features_scaled = scaler.transform(df.values)  # StandardScaler
            #     features_pca = pca.transform(features_scaled)  # PCA降维
            #
            #     # ======== 预测部分 ========
            #     st.subheader("Run Prediction")
            #
            #     if model_selected and st.button("Run Prediction"):
            #         try:
            #             # 模型已经加载好了 joblib.load(model_path)
            #             prediction = model.predict(features_pca)
            #
            #             # 显示预测结果
            #             st.subheader("Prediction Result")
            #             st.write(f"Predicted value: {prediction[0]:.2f}")
            #
            #             # 可选：生成下载 CSV
            #             result_data = df.copy()
            #             result_data['Predicted'] = prediction
            #             csv_file = result_data.to_csv(index=False)
            #             st.download_button("Download Prediction CSV", csv_file, "prediction.csv", "text/csv")
            #
            #         except FileNotFoundError:
            #             st.error(f"Model file not found: {model_path}")


        # elif model_source == '[2] Upload your own model (Please upload your model here)':
        #
        #     file = st.file_uploader("Upload `.csv`file", label_visibility="collapsed", accept_multiple_files=True)
        #
        #     if len(file) < 2:
        #         table = PrettyTable(['file name', 'class', 'description'])
        #         table.add_row(['file_1', 'data set (+test data)', 'data file'])
        #         table.add_row(['file_2', 'model', 'model'])
        #         st.write(table)
        #     elif len(file) == 2:
        #         df = pd.read_csv(file[0])
        #         model_file = file[1]
        #
        #         try:
        #             model = pickle.load(model_file)
        #
        #             if hasattr(model, 'n_features_in_'):
        #                 n_model_features = model.n_features_in_
        #                 st.info(f"ℹ️ This model expects **{n_model_features} features** as input.")
        #             else:
        #                 st.warning("⚠️ This model does not have `n_features_in_` attribute.")
        #         except Exception as e:
        #             st.error(f"Error loading model: {e}")
        #             model = None
        #             n_model_features = None
        #
        #         check_string_NaN(df)
        #
        #         colored_header(label="Data information", description=" ", color_name="violet-70")
        #         nrow = st.slider("rows", 1, len(df), 5)
        #         df_nrow = df.head(nrow)
        #         st.write(df_nrow)
        #
        #         colored_header(label="Feature and target", description=" ", color_name="violet-70")
        #
        #         target_num = st.number_input('target number', min_value=1, max_value=10, value=1)
        #
        #         col_feature, col_target = st.columns(2)
        #         # features
        #         features = df.iloc[:, :-target_num]
        #         # targets
        #         targets = df.iloc[:, -target_num:]
        #         with col_feature:
        #             st.write(features.head())
        #         with col_target:
        #             st.write(targets.head())
        #         colored_header(label="target", description=" ", color_name="violet-70")
        #
        #         target_selected_option = st.selectbox('target', list(targets)[::-1])
        #
        #         targets = targets[target_selected_option]
        #         preprocess = st.selectbox('data preprocess', [None, 'StandardScaler', 'MinMaxScaler'])
        #         if preprocess == 'StandardScaler':
        #             features = StandardScaler().fit_transform(features)
        #         elif preprocess == 'MinMaxScaler':
        #             features = MinMaxScaler().fit_transform(features)
        #
        #         model = pickle.load(model_file)
        #         prediction = model.predict(features)
        #         # st.write(std)
        #         plot = customPlot()
        #         plot.pred_vs_actual(targets, prediction)
        #         r2 = r2_score(targets, prediction)
        #         st.write('R2: {}'.format(r2))
        #         result_data = pd.concat([targets, pd.DataFrame(prediction)], axis=1)
        #         result_data.columns = ['actual', 'prediction']
        #         with st.expander('prediction'):
        #             st.write(result_data)
        #             tmp_download_link = download_button(result_data, f'prediction.csv', button_text='download')
        #             st.markdown(tmp_download_link, unsafe_allow_html=True)
        #         st.write('---')

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