import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="泵曲線繪製工具", layout="wide")

# 標題
st.title("泵曲線繪製工具")
st.markdown("這個應用程式可以幫助您繪製泵曲線，顯示揚程與流量的關係，並在同一圖表上顯示所有單位。")

# 單位轉換函數
def convert_flow(value, from_unit, to_unit):
    # 如果單位相同，不需要轉換
    if from_unit == to_unit:
        return value
    
    # 轉換係數
    # 1 GPM = 3.78541 LPM
    # 1 M3/HR = 4.40287 GPM
    # 1 M3/HR = 16.6667 LPM
    
    # 先轉換到 GPM
    if from_unit == "LPM":
        gpm = value / 3.78541
    elif from_unit == "M3/HR":
        gpm = value * 4.40287
    else:  # GPM
        gpm = value
    
    # 從 GPM 轉換到目標單位
    if to_unit == "LPM":
        return gpm * 3.78541
    elif to_unit == "M3/HR":
        return gpm / 4.40287
    else:  # GPM
        return gpm

def convert_head(value, from_unit, to_unit):
    # 如果單位相同，不需要轉換
    if from_unit == to_unit:
        return value
    
    # 轉換係數
    # 1 FT = 0.3048 M
    if from_unit == "FT" and to_unit == "M":
        return value * 0.3048
    elif from_unit == "M" and to_unit == "FT":
        return value / 0.3048

# 側邊欄，用於輸入泵曲線數據
with st.sidebar:
    st.header("泵曲線數據輸入")
    
    # 基準單位選擇（用於輸入數據）
    st.subheader("數據輸入單位")
    base_flow_unit = st.selectbox("流量輸入單位", ["GPM", "LPM", "M3/HR"])
    base_head_unit = st.selectbox("揚程輸入單位", ["FT", "M"])
    
    # 數據輸入方式選擇
    input_method = st.radio("請選擇數據輸入方式", ["手動輸入數據", "上傳CSV檔案"])
    
    if input_method == "手動輸入數據":
        st.subheader("手動輸入數據")
        
        # 手動輸入數據點
        num_points = st.number_input("數據點數量", min_value=2, max_value=20, value=5)
        
        # 創建多個輸入欄位
        flow_values = []
        head_values = []
        
        for i in range(num_points):
            col1, col2 = st.columns(2)
            with col1:
                flow = st.number_input(f"流量 {i+1} ({base_flow_unit})", value=float(i*100), key=f"flow_{i}")
                flow_values.append(flow)
            with col2:
                head = st.number_input(f"揚程 {i+1} ({base_head_unit})", value=float(100-i*10), key=f"head_{i}")
                head_values.append(head)
    
    else:
        st.subheader("上傳CSV檔案")
        st.markdown(f"請上傳一個包含兩列數據的CSV檔案：流量({base_flow_unit})和揚程({base_head_unit})")
        st.markdown("CSV檔案格式應為：第一列為流量，第二列為揚程，不含標題行")
        uploaded_file = st.file_uploader("選擇CSV檔案", type="csv")

# 主要應用區域
st.header("泵曲線圖")

# 準備數據
if input_method == "手動輸入數據" and len(flow_values) > 0 and len(head_values) > 0:
    df = pd.DataFrame({
        '流量': flow_values,
        '揚程': head_values
    })
elif input_method == "上傳CSV檔案" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = ['流量', '揚程']
else:
    # 預設數據
    df = pd.DataFrame({
        '流量': [0, 100, 200, 300, 400],
        '揚程': [100, 90, 75, 55, 30]
    })

# 顯示原始數據表格
st.subheader("泵曲線原始數據")
st.dataframe(df.rename(columns={
    '流量': f'流量 ({base_flow_unit})', 
    '揚程': f'揚程 ({base_head_unit})'
}))

# 建立所有單位轉換的數據表
st.subheader("所有單位數據表")
all_units_data = []

# 流量單位列表
flow_units = ["GPM", "LPM", "M3/HR"]
# 揚程單位列表
head_units = ["FT", "M"]

# 為每個數據點和每種單位組合創建一行
for i, row in df.iterrows():
    original_flow = row['流量']
    original_head = row['揚程']
    
    for f_unit in flow_units:
        converted_flow = convert_flow(original_flow, base_flow_unit, f_unit)
        
        for h_unit in head_units:
            converted_head = convert_head(original_head, base_head_unit, h_unit)
            
            all_units_data.append({
                '數據點': i+1,
                f'流量 (GPM)': round(convert_flow(original_flow, base_flow_unit, "GPM"), 2),
                f'流量 (LPM)': round(convert_flow(original_flow, base_flow_unit, "LPM"), 2),
                f'流量 (M3/HR)': round(convert_flow(original_flow, base_flow_unit, "M3/HR"), 2),
                f'揚程 (FT)': round(convert_head(original_head, base_head_unit, "FT"), 2),
                f'揚程 (M)': round(convert_head(original_head, base_head_unit, "M"), 2)
            })

# 創建單位轉換表
all_units_df = pd.DataFrame(all_units_data)

# 創建展開版數據，只保留數據點和選定單位
expanded_data = []
for index, row in df.iterrows():
    point_data = {'數據點': index + 1}
    
    for f_unit in flow_units:
        point_data[f'流量 ({f_unit})'] = round(convert_flow(row['流量'], base_flow_unit, f_unit), 2)
    
    for h_unit in head_units:
        point_data[f'揚程 ({h_unit})'] = round(convert_head(row['揚程'], base_head_unit, h_unit), 2)
    
    expanded_data.append(point_data)

expanded_df = pd.DataFrame(expanded_data)
st.dataframe(expanded_df)

# 繪製曲線
fig, ax = plt.subplots(figsize=(12, 8))

# 準備數據線的顏色和樣式
colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
linestyles = ['-', '--', '-.', ':']
color_index = 0

# 為每種單位組合繪製一條線
for h_unit in head_units:
    for f_unit in flow_units:
        # 轉換流量和揚程數據到當前單位
        converted_flow = [convert_flow(val, base_flow_unit, f_unit) for val in df['流量']]
        converted_head = [convert_head(val, base_head_unit, h_unit) for val in df['揚程']]
        
        # 排序數據以確保線條平滑
        sort_idx = np.argsort(converted_flow)
        sorted_flow = np.array(converted_flow)[sort_idx]
        sorted_head = np.array(converted_head)[sort_idx]
        
        # 繪製轉換後的曲線
        line_style = linestyles[(color_index // len(colors)) % len(linestyles)]
        color = colors[color_index % len(colors)]
        ax.plot(sorted_flow, sorted_head, 
                label=f"流量: {f_unit} / 揚程: {h_unit}", 
                color=color, 
                linestyle=line_style,
                marker='o',
                linewidth=2)
        color_index += 1

# 設置圖表
ax.set_xlabel("流量 (多單位)")
ax.set_ylabel("揚程 (多單位)")
ax.set_title("泵性能曲線 (所有單位)")
ax.grid(True)
ax.legend(loc='best')

# 添加文字說明
plt.figtext(0.02, 0.02, f"原始數據單位 - 流量: {base_flow_unit}, 揚程: {base_head_unit}", fontsize=10)

# 顯示圖表
st.pyplot(fig)

# 添加單位換算說明
st.subheader("單位換算參考")
st.markdown("""
- 流量單位: 1 GPM ≈ 3.785 LPM, 1 M3/HR ≈ 4.403 GPM ≈ 16.667 LPM
- 揚程單位: 1 FT ≈ 0.3048 M
""")

# 添加下載功能
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300)
buffer.seek(0)
btn = st.download_button(
    label="下載泵曲線圖",
    data=buffer,
    file_name="pump_curve_all_units.png",
    mime="image/png"
)

# 添加CSV下載功能
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(expanded_df)
st.download_button(
    label="下載所有單位數據CSV",
    data=csv,
    file_name="pump_curve_all_units.csv",
    mime="text/csv",
)
