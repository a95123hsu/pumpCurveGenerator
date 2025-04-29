import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

st.set_page_config(page_title="泵曲線繪製工具 | Pump Curve Plotting Tool", layout="wide")

# 設置頁面標題和說明
st.title("泵曲線繪製工具 | Pump Curve Plotting Tool")
st.markdown("""
這個應用程式可以幫助您繪製各種水泵型號的性能曲線。您可以輸入水泵參數、選擇單位，然後生成曲線圖。
This application helps you plot performance curves for various pump models. You can input pump parameters, select units, and generate curves.
""")

# 建立側邊欄，用於選擇參數
st.sidebar.header("參數設置 | Parameters")

# 選擇曲線類型
pump_model = st.sidebar.selectbox(
    "選擇泵型號 | Select Pump Model",
    ["DS-05", "DS-10", "DS-20", "DS-30", "自定義 | Custom"]
)

# 頻率選擇
frequency = st.sidebar.radio(
    "頻率 | Frequency",
    ["50Hz", "60Hz"]
)

# 單位選擇
head_unit = st.sidebar.radio(
    "揚程單位 | Head Unit",
    ["米 (m)", "英尺 (ft)"]
)

flow_unit = st.sidebar.radio(
    "流量單位 | Flow Unit",
    ["LPM", "GPM", "m³/hr"]
)

# 創建一個函數來生成泵曲線數據
def generate_pump_curve(model, unit_flow, unit_head):
    """根據泵型號和單位生成曲線數據"""
    
    # 預設模型數據 (根據圖片中的DS系列泵)
    # 依據圖片中的曲線調整數據點
    model_data = {
        "DS-05": {
            "50Hz": {
                "flow_lpm": [0, 50, 100, 150, 200, 230],
                "head_m": [9, 7, 5, 3.5, 2, 0.5]
            },
            "60Hz": {
                "flow_lpm": [0, 60, 120, 180, 240, 280],
                "head_m": [13, 10, 7.5, 5, 3, 0.7]
            }
        },
        "DS-10": {
            "50Hz": {
                "flow_lpm": [0, 75, 150, 225, 300, 370],
                "head_m": [13, 11, 8, 6, 4, 0.8]
            },
            "60Hz": {
                "flow_lpm": [0, 90, 180, 270, 360, 440],
                "head_m": [19, 16, 12, 8, 5, 1]
            }
        },
        "DS-20": {
            "50Hz": {
                "flow_lpm": [0, 110, 220, 330, 440, 550],
                "head_m": [16, 14, 11, 8, 6, 0.9]
            },
            "60Hz": {
                "flow_lpm": [0, 130, 260, 390, 520, 650],
                "head_m": [23, 20, 16, 12, 8, 1.3]
            }
        },
        "DS-30": {
            "50Hz": {
                "flow_lpm": [0, 160, 320, 480, 640, 800],
                "head_m": [18, 17, 14, 11, 8, 1]
            },
            "60Hz": {
                "flow_lpm": [0, 190, 380, 570, 760, 950],
                "head_m": [25, 23, 20, 16, 11, 1.5]
            }
        }
    }
    
    # 如果是自定義，則返回空數據，等待用戶輸入
    if model.startswith("自定"):
        return pd.DataFrame(columns=["Flow", "Head"])
    
    # 獲取基本數據
    if model in model_data and frequency in model_data[model]:
        base_data = model_data[model][frequency]
    else:
        # 默認情況
        base_data = model_data["DS-10"]["50Hz"]
    
    df = pd.DataFrame({
        "Flow": base_data["flow_lpm"],
        "Head": base_data["head_m"]
    })
    
    # 單位轉換 - 流量
    if unit_flow == "GPM":
        df["Flow"] = df["Flow"] * 0.26417  # LPM 轉 GPM
    elif unit_flow == "m³/hr":
        df["Flow"] = df["Flow"] * 0.06  # LPM 轉 m³/hr
    
    # 單位轉換 - 揚程
    if unit_head == "英尺 (ft)":
        df["Head"] = df["Head"] * 3.28084  # 米轉英尺
    
    return df

# 顯示當前選擇的參數
st.write(f"當前選擇: {pump_model}, {frequency}, 揚程單位: {head_unit}, 流量單位: {flow_unit}")

# 添加CSV上傳功能
st.subheader("上傳自定義泵曲線數據 | Upload Custom Pump Curve Data")
uploaded_file = st.file_uploader("上傳CSV文件 | Upload CSV file", type=['csv'])

# 初始化曲線數據
curve_data = None

# 處理上傳的CSV文件
if uploaded_file is not None:
    try:
        # 讀取CSV數據
        df = pd.read_csv(uploaded_file)
        st.success("文件上傳成功！| File uploaded successfully!")
        
        # 顯示上傳的數據
        st.write("上傳的數據 | Uploaded data:")
        st.dataframe(df)
        
        # 讓用戶選擇欄位
        st.subheader("選擇數據欄位 | Select Data Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            flow_column = st.selectbox(
                "選擇流量欄位 | Select Flow Column",
                options=df.columns.tolist(),
                index=0 if len(df.columns) > 0 else None
            )
        
        with col2:
            head_column = st.selectbox(
                "選擇揚程欄位 | Select Head Column",
                options=df.columns.tolist(),
                index=1 if len(df.columns) > 1 else 0 if len(df.columns) > 0 else None
            )
        
        if flow_column and head_column:
            # 創建曲線數據
            curve_data = pd.DataFrame({
                "Flow": df[flow_column],
                "Head": df[head_column]
            })
            
            # 添加曲線名稱
            custom_curve_name = st.text_input("曲線名稱 | Curve Name", "上傳的曲線 | Uploaded Curve")
            pump_model = custom_curve_name
    except Exception as e:
        st.error(f"處理文件時出錯: {e} | Error processing file: {e}")
        curve_data = None

# 如果沒有上傳文件或處理出錯，則生成模型數據
if curve_data is None:
    curve_data = generate_pump_curve(
        pump_model, 
        flow_unit, 
        head_unit
    )

# 如果是自定義模式且沒有上傳文件，提供數據輸入界面
if pump_model.startswith("自定") and uploaded_file is None:
    st.subheader("輸入自定義泵曲線數據 | Enter Custom Pump Curve Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"流量 ({flow_unit})")
        flow_data = st.text_area("每行輸入一個流量值", "0\n100\n200\n300")
    
    with col2:
        st.write(f"揚程 ({head_unit})")
        head_data = st.text_area("每行輸入一個揚程值", "20\n15\n10\n5")
    
    # 處理用戶輸入的數據
    try:
        flow_values = [float(x.strip()) for x in flow_data.split('\n') if x.strip()]
        head_values = [float(x.strip()) for x in head_data.split('\n') if x.strip()]
        
        # 確保兩個列表長度相同
        min_len = min(len(flow_values), len(head_values))
        flow_values = flow_values[:min_len]
        head_values = head_values[:min_len]
        
        curve_data = pd.DataFrame({
            "Flow": flow_values,
            "Head": head_values
        })
    except ValueError:
        st.error("請確保輸入的都是數字。")

# 添加多個型號比較選項
st.subheader("添加更多泵型號進行比較 | Add More Pump Models for Comparison")
compare_models = st.multiselect(
    "選擇要比較的其他型號 | Select other models to compare",
    ["DS-05", "DS-10", "DS-20", "DS-30"],
    default=[]
)

# 顯示數據表
st.subheader("泵曲線數據 | Pump Curve Data")
st.dataframe(curve_data.style.format("{:.2f}"))

# 繪製泵曲線圖
st.subheader("泵曲線圖 | Pump Curve Chart")

# 使用Plotly創建交互式圖表
fig = go.Figure()

# 為圖表添加多個X軸和Y軸，類似於原始圖片
# 主軸為預設單位
fig.update_layout(
    xaxis=dict(
        title=f"流量 | Flow ({flow_unit})",
        side="bottom",
        anchor="y",
    ),
    yaxis=dict(
        title=f"揚程 | Head ({head_unit})",
        side="left",
        anchor="x",
    )
)

# 添加第二個X軸（如果主軸不是LPM）
if flow_unit != "LPM":
    fig.update_layout(
        xaxis2=dict(
            title="流量 | Flow (LPM)",
            side="bottom",
            anchor="y",
            overlaying="x",
            position=0.05,
            showgrid=False,
        )
    )

# 添加第三個X軸（如果主軸不是GPM）
if flow_unit != "GPM":
    fig.update_layout(
        xaxis3=dict(
            title="流量 | Flow (GPM)",
            side="bottom",
            anchor="y",
            overlaying="x",
            position=0.1,
            showgrid=False,
        )
    )

# 添加第四個X軸（如果主軸不是m³/hr）
if flow_unit != "m³/hr":
    fig.update_layout(
        xaxis4=dict(
            title="流量 | Flow (m³/hr)",
            side="bottom",
            anchor="y",
            overlaying="x",
            position=0.15,
            showgrid=False,
        )
    )

# 添加第二個Y軸（如果主軸不是米）
if head_unit != "米 (m)":
    fig.update_layout(
        yaxis2=dict(
            title="揚程 | Head (m)",
            side="left",
            anchor="x",
            overlaying="y",
            position=0.05,
            showgrid=False,
        )
    )

# 添加第三個Y軸（如果主軸不是英尺）
if head_unit != "英尺 (ft)":
    fig.update_layout(
        yaxis3=dict(
            title="揚程 | Head (ft)",
            side="left",
            anchor="x",
            overlaying="y",
            position=0.1,
            showgrid=False,
        )
    )

# 單位轉換函數
def convert_units(data, from_flow, to_flow, from_head, to_head):
    converted = data.copy()
    
    # 流量單位轉換
    if from_flow != to_flow:
        if from_flow == "LPM":
            if to_flow == "GPM":
                converted["Flow"] = data["Flow"] * 0.26417
            elif to_flow == "m³/hr":
                converted["Flow"] = data["Flow"] * 0.06
        elif from_flow == "GPM":
            if to_flow == "LPM":
                converted["Flow"] = data["Flow"] / 0.26417
            elif to_flow == "m³/hr":
                converted["Flow"] = data["Flow"] * 0.227125
        elif from_flow == "m³/hr":
            if to_flow == "LPM":
                converted["Flow"] = data["Flow"] / 0.06
            elif to_flow == "GPM":
                converted["Flow"] = data["Flow"] / 0.227125
    
    # 揚程單位轉換
    if from_head != to_head:
        if (from_head == "米 (m)" and to_head == "英尺 (ft)"):
            converted["Head"] = data["Head"] * 3.28084
        elif (from_head == "英尺 (ft)" and to_head == "米 (m)"):
            converted["Head"] = data["Head"] / 3.28084
    
    return converted

# 添加主曲線
fig.add_trace(go.Scatter(
    x=curve_data["Flow"],
    y=curve_data["Head"],
    mode='lines+markers',
    name=pump_model,
    line=dict(color='blue', width=3)
))

# 添加比較曲線
for model in compare_models:
    comp_data = generate_pump_curve(model, flow_unit, head_unit)
    fig.add_trace(go.Scatter(
        x=comp_data["Flow"],
        y=comp_data["Head"],
        mode='lines+markers',
        name=model,
        line=dict(width=2, dash='dash')
    ))

# 添加網格線
fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
)

# 設置圖表佈局
fig.update_layout(
    title=f'泵性能曲線 | Pump Performance Curve ({frequency})',
    legend_title="泵型號 | Pump Model",
    height=600,
    hovermode="closest",
    margin=dict(t=100, b=100)  # 增加上下邊距，為多軸留出空間
)

# 添加DS型號標籤，類似於原圖
for i, model_name in enumerate(["DS-05", "DS-10", "DS-20", "DS-30"]):
    if model_name in [pump_model] + compare_models:
        # 找到相應曲線的中點位置來放置標籤
        if model_name == pump_model:
            data = curve_data
        else:
            data = generate_pump_curve(model_name, flow_unit, head_unit)
        
        if len(data) > 1:
            # 找到曲線的中點位置
            mid_idx = len(data) // 2
            x_pos = data["Flow"].iloc[mid_idx]
            y_pos = data["Head"].iloc[mid_idx]
            
            # 添加標籤
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=model_name,
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="white",
                opacity=0.8
            )

# 顯示圖表
st.plotly_chart(fig, use_container_width=True)

# 創建並顯示表格，包含所有單位
st.subheader("泵曲線數據（全部單位）| Pump Curve Data (All Units)")

# 創建包含所有單位的數據表
all_units_data = pd.DataFrame()

# 添加原始數據
all_units_data[f"流量 | Flow ({flow_unit})"] = curve_data["Flow"]
all_units_data[f"揚程 | Head ({head_unit})"] = curve_data["Head"]

# 添加其他流量單位
for unit in ["LPM", "GPM", "m³/hr"]:
    if unit != flow_unit:
        converted = convert_units(curve_data, flow_unit, unit, head_unit, head_unit)
        all_units_data[f"流量 | Flow ({unit})"] = converted["Flow"]

# 添加其他揚程單位
for unit in ["米 (m)", "英尺 (ft)"]:
    if unit != head_unit:
        converted = convert_units(curve_data, flow_unit, flow_unit, head_unit, unit)
        all_units_data[f"揚程 | Head ({unit})"] = converted["Head"]

# 顯示數據表
st.dataframe(all_units_data.style.format("{:.2f}"))

# 添加效率曲線選項（高級功能）
st.subheader("高級功能 | Advanced Features")
show_efficiency = st.checkbox("顯示效率曲線 | Show Efficiency Curve")

if show_efficiency:
    st.info("效率曲線功能需要更多數據支持，這是一個示範。在實際應用中，您應該提供真實的效率數據。")
    
    # 創建一個簡單的效率曲線示例
    eff_curve = curve_data.copy()
    max_flow = eff_curve["Flow"].max()
    # 假設效率在中間流量時達到最高
    eff_curve["Efficiency"] = 100 * (1 - ((eff_curve["Flow"] - max_flow/2) / max_flow) ** 2)
    
    # 創建雙Y軸圖表
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 添加揚程曲線
    fig2.add_trace(
        go.Scatter(
            x=curve_data["Flow"],
            y=curve_data["Head"],
            name=f"{pump_model} 揚程 | Head",
            line=dict(color='blue', width=3)
        ),
        secondary_y=False
    )
    
    # 添加效率曲線
    fig2.add_trace(
        go.Scatter(
            x=eff_curve["Flow"],
            y=eff_curve["Efficiency"],
            name="效率 | Efficiency",
            line=dict(color='red', width=2, dash='dot')
        ),
        secondary_y=True
    )
    
    # 設置軸標題
    fig2.update_xaxes(title_text=f"流量 | Flow ({flow_unit})")
    fig2.update_yaxes(title_text=f"揚程 | Head ({head_unit})", secondary_y=False)
    fig2.update_yaxes(title_text="效率 | Efficiency (%)", secondary_y=True)
    
    # 更新佈局
    fig2.update_layout(
        title=f'泵性能和效率曲線 | Pump Performance and Efficiency Curve ({frequency})',
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# 添加工作點計算功能
st.subheader("工作點計算 | Working Point Calculation")
col1, col2 = st.columns(2)

with col1:
    system_head = st.number_input(
        f"系統靜態揚程 | Static Head ({head_unit})", 
        min_value=0.0, 
        value=5.0, 
        step=0.5
    )

with col2:
    k_factor = st.number_input(
        "系統阻力係數 K | System Resistance Factor", 
        min_value=0.0, 
        value=0.001, 
        format="%.5f",
        step=0.0001
    )

# 計算系統曲線
if not curve_data.empty:
    max_flow = curve_data["Flow"].max() * 1.2
    flow_range = np.linspace(0, max_flow, 100)
    system_curve = pd.DataFrame({
        "Flow": flow_range,
        "Head": system_head + k_factor * flow_range**2
    })
    
    # 在圖表上添加系統曲線
    fig3 = go.Figure()
    
    # 添加泵曲線
    fig3.add_trace(go.Scatter(
        x=curve_data["Flow"],
        y=curve_data["Head"],
        mode='lines+markers',
        name=f"{pump_model} 曲線 | Curve",
        line=dict(color='blue', width=3)
    ))
    
    # 添加系統曲線
    fig3.add_trace(go.Scatter(
        x=system_curve["Flow"],
        y=system_curve["Head"],
        mode='lines',
        name="系統曲線 | System Curve",
        line=dict(color='green', width=2)
    ))
    
    # 查找交點（工作點）
    # 這是一個簡化的方法，實際應用中可能需要更精確的數值方法
    # 對泵曲線進行插值
    from scipy import interpolate
    
    if len(curve_data) > 1:  # 確保有足夠的點進行插值
        pump_curve_interp = interpolate.interp1d(
            curve_data["Flow"], 
            curve_data["Head"],
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        # 計算誤差函數
        def error_func(flow):
            pump_head = pump_curve_interp(flow)
            system_head_at_flow = system_head + k_factor * flow**2
            return abs(pump_head - system_head_at_flow)
        
        # 在流量範圍內找到工作點
        best_flow = None
        min_error = float('inf')
        
        for flow in flow_range:
            err = error_func(flow)
            if err < min_error:
                min_error = err
                best_flow = flow
        
        if best_flow is not None:
            working_point_head = system_head + k_factor * best_flow**2
            
            # 添加工作點到圖表
            fig3.add_trace(go.Scatter(
                x=[best_flow],
                y=[working_point_head],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name="工作點 | Working Point"
            ))
            
            # 顯示工作點信息
            st.success(f"""
            **工作點 | Working Point:**
            - 流量 | Flow: {best_flow:.2f} {flow_unit}
            - 揚程 | Head: {working_point_head:.2f} {head_unit}
            """)
    
    # 設置圖表佈局
    fig3.update_layout(
        title='泵與系統曲線交點 | Pump and System Curve Intersection',
        xaxis_title=f'流量 | Flow ({flow_unit})',
        yaxis_title=f'揚程 | Head ({head_unit})',
        height=600
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# 添加下載選項
st.subheader("導出數據 | Export Data")
col1, col2 = st.columns(2)

with col1:
    # 下載全部單位的數據
    if st.button("下載數據為CSV（全部單位）| Download Data with All Units as CSV"):
        csv = all_units_data.to_csv(index=False)
        st.download_button(
            label="點擊下載所有單位CSV | Click to Download All Units CSV",
            data=csv,
            file_name=f"{pump_model}_all_units_data.csv",
            mime="text/csv"
        )

with col2:
    # 下載僅當前單位的數據
    if st.button("下載數據為CSV（當前單位）| Download Data with Current Units as CSV"):
        csv = curve_data.to_csv(index=False)
        st.download_button(
            label="點擊下載當前單位CSV | Click to Download Current Units CSV",
            data=csv,
            file_name=f"{pump_model}_current_units_data.csv",
            mime="text/csv"
        )

# 添加曲線數據的CSV模板下載
st.subheader("CSV模板 | CSV Template")
st.write("如果您想準備自己的CSV文件上傳，可以先下載此模板 | If you want to prepare your own CSV file for upload, you can download this template first")

# 創建模板數據
template_data = pd.DataFrame({
    "Flow_LPM": [0, 100, 200, 300, 400, 500],
    "Head_m": [25, 20, 15, 10, 5, 2]
})

# 提供模板下載
csv_template = template_data.to_csv(index=False)
st.download_button(
    label="下載CSV模板 | Download CSV Template",
    data=csv_template,
    file_name="pump_curve_template.csv",
    mime="text/csv"
)

# 添加圖表導出功能（使用Plotly的內置功能）
st.info("要下載圖表為PNG或SVG格式，請使用Plotly圖表右上角的相機圖標 | To download the chart as PNG or SVG, use the camera icon in the top right corner of the Plotly chart")

# 添加CSV格式說明
st.subheader("CSV文件格式說明 | CSV File Format Instructions")
st.markdown("""
為了成功上傳您的泵曲線數據，請確保您的CSV文件符合以下格式要求：

1. 文件應該至少包含兩列數據：一列用於流量值，一列用於揚程值
2. 文件必須包含標題行（列名）
3. 數據應該按照流量升序或降序排列
4. 數值應該使用小數點（.）作為小數分隔符
5. 不應該包含單位符號，只有數字
6. 不應該有空值

**示例CSV內容：**
```
Flow_LPM,Head_m
0,30
100,25
200,15
300,8
400,3
```

上傳後，您可以在界面中選擇哪一列代表流量，哪一列代表揚程。
""")

# 添加說明信息
st.markdown("""
---
### 使用說明 | Usage Instructions

1. **基本功能：**
   - 在側邊欄選擇泵型號、頻率和單位
   - 查看生成的泵曲線圖（包含所有單位）
   - 添加其他泵型號進行比較

2. **自定義數據：**
   - 上傳自己的CSV文件
   - 或使用自定義模式手動輸入數據點

3. **工作點計算：**
   - 設置系統參數計算工作點
   - 查看泵與系統曲線的交點

4. **數據導出：**
   - 下載包含所有單位或當前單位的CSV數據
   - 使用Plotly工具下載圖表為PNG或SVG格式

---
### 注意事項 | Notes

- 本工具顯示的所有單位轉換是基於標準公式計算，可能與實際製造商數據有微小差異
- 計算結果僅供參考，實際設計應諮詢專業工程師
- 上傳的CSV文件不應超過1MB大小
- 使用高質量的數據點以獲得更平滑、更準確的曲線

---
""")

# 添加頁腳
st.markdown("© 2025 泵曲線繪製工具 | Pump Curve Plotting Tool | Created with Streamlit")
