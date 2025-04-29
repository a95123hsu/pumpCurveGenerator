import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    # 這裡使用簡化的數據點，實際應用中應使用更準確的數據
    model_data = {
        "DS-05": {
            "50Hz": {
                "flow_lpm": [0, 100, 200, 230],
                "head_m": [9, 5, 2, 0.5]
            }
        },
        "DS-10": {
            "50Hz": {
                "flow_lpm": [0, 150, 300, 370],
                "head_m": [13, 8, 4, 0.8]
            }
        },
        "DS-20": {
            "50Hz": {
                "flow_lpm": [0, 200, 400, 550],
                "head_m": [16, 11, 6, 0.9]
            }
        },
        "DS-30": {
            "50Hz": {
                "flow_lpm": [0, 300, 600, 800],
                "head_m": [18, 14, 8, 1]
            }
        }
    }
    
    # 如果是自定義，則返回空數據，等待用戶輸入
    if model.startswith("自定"):
        return pd.DataFrame(columns=["Flow", "Head"])
    
    # 獲取基本數據
    base_data = model_data.get(model, {}).get(frequency, model_data["DS-10"]["50Hz"])
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

# 生成泵曲線數據
curve_data = generate_pump_curve(
    pump_model, 
    flow_unit, 
    head_unit
)

# 如果是自定義模式，提供數據輸入界面
if pump_model.startswith("自定"):
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

# 設置圖表佈局
fig.update_layout(
    title=f'泵性能曲線 | Pump Performance Curve ({frequency})',
    xaxis_title=f'流量 | Flow ({flow_unit})',
    yaxis_title=f'揚程 | Head ({head_unit})',
    legend_title="泵型號 | Pump Model",
    height=600,
    hovermode="closest"
)

# 顯示圖表
st.plotly_chart(fig, use_container_width=True)

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
    if st.button("下載數據為CSV | Download Data as CSV"):
        csv = curve_data.to_csv(index=False)
        st.download_button(
            label="點擊下載CSV | Click to Download CSV",
            data=csv,
            file_name=f"{pump_model}_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("下載圖表為PNG | Download Chart as PNG"):
        st.info("在實際應用中，這裏應該提供圖表下載功能。在這個示例中，您可以使用Plotly圖表右上角的相機圖標來下載圖表。")

# 添加說明信息
st.markdown("""
---
### 使用說明 | Usage Instructions

1. 在側邊欄選擇泵型號、頻率和單位
2. 查看生成的泵曲線圖
3. 可選: 添加其他泵型號進行比較
4. 可選: 設置系統參數計算工作點
5. 下載數據或圖表供後續使用

---
### 注意事項 | Notes

- 本工具使用的數據是模擬數據，實際應用中應使用製造商提供的正確參數
- 計算結果僅供參考，實際設計應諮詢專業工程師
- 如需更精確的結果，請輸入更多的數據點

---
""")

# 添加頁腳
st.markdown("© 2025 泵曲線繪製工具 | Pump Curve Plotting Tool")
