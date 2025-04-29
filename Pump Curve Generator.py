import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# Unit conversion factors
FLOW_CONVERSION = {
    ("GPM", "LPM"): 3.78541,
    ("GPM", "M3/HR"): 0.2271247,
    ("LPM", "GPM"): 1/3.78541,
    ("LPM", "M3/HR"): 0.06,
    ("M3/HR", "GPM"): 1/0.2271247,
    ("M3/HR", "LPM"): 1000/60
}

HEAD_CONVERSION = {
    ("FT", "M"): 0.3048,
    ("M", "FT"): 1/0.3048
}

def convert_units(df, from_flow, to_flow, from_head, to_head):
    if from_flow != to_flow:
        df['Flow'] *= FLOW_CONVERSION[(from_flow, to_flow)]
    if from_head != to_head:
        df['Head'] *= HEAD_CONVERSION[(from_head, to_head)]
    return df

def create_pdf(fig, curves_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Pump Curve Report", ln=True, align='C')
    pdf.ln(10)
    
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='PNG')
    img_buffer.seek(0)
    pdf.image(img_buffer, x=10, y=30, w=190)
    pdf.ln(105)

    for model, data in curves_data.items():
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(0, 10, txt=f"Model: {model}", ln=True)
        pdf.set_font("Arial", size=10)
        for idx, row in data.iterrows():
            pdf.cell(0, 10, txt=f"Flow: {row['Flow']:.2f}, Head: {row['Head']:.2f}", ln=True)
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin1')

st.title("🚀 Pump Curve Tool - 多曲線版")

st.sidebar.header("設定區")
flow_unit = st.sidebar.selectbox("選擇流量單位", ("GPM", "LPM", "M3/HR"))
head_unit = st.sidebar.selectbox("選擇揚程單位", ("FT", "M"))

st.sidebar.markdown("---")

if 'curves' not in st.session_state:
    st.session_state.curves = {}

st.sidebar.subheader("管理曲線")
new_curve = st.sidebar.text_input("新增曲線型號名稱")

if st.sidebar.button("新增曲線"):
    if new_curve:
        st.session_state.curves[new_curve] = pd.DataFrame({"Flow": [], "Head": []})

remove_curve = st.sidebar.selectbox("刪除曲線", ["無"] + list(st.session_state.curves.keys()))
if st.sidebar.button("刪除選定曲線"):
    if remove_curve != "無":
        st.session_state.curves.pop(remove_curve, None)

# 曲線輸入
st.subheader("曲線數據輸入")
selected_curve = st.selectbox("選擇編輯的曲線", list(st.session_state.curves.keys()))

if selected_curve:
    curves_data = st.session_state.curves[selected_curve]
    edited_data = st.data_editor(curves_data, num_rows="dynamic", key=selected_curve)
    st.session_state.curves[selected_curve] = edited_data

# 畫曲線
st.subheader("泵浦性能曲線")
fig, ax = plt.subplots()

for model, data in st.session_state.curves.items():
    if not data.empty:
        display_data = data.copy()
        ax.plot(display_data['Flow'], display_data['Head'], label=model)

ax.set_xlabel(f"Flow ({flow_unit})")
ax.set_ylabel(f"Head ({head_unit})")
ax.set_title("Pump Curves")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# 匯出PDF
if st.button("匯出成PDF"):
    pdf_bytes = create_pdf(fig, st.session_state.curves)
    st.download_button("下載PDF報表", data=pdf_bytes, file_name="pump_curves.pdf", mime="application/pdf")

# 匯出圖片
img_buffer = BytesIO()
fig.savefig(img_buffer, format='png')
img_buffer.seek(0)

st.download_button("下載曲線圖 (PNG)", data=img_buffer, file_name="pump_curve.png", mime="image/png")
