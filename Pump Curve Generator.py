import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="泵浦曲線繪製工具", layout="wide")

st.title("泵浦曲線繪製工具 (Pump Curve Plotting Tool)")
st.markdown("這個應用程序可以幫助您繪製泵浦的性能曲線，並在不同單位下顯示結果。")

# 單位轉換函數
def convert_flow(value, from_unit, to_unit):
    # 先轉換到GPM作為標準單位
    if from_unit == "GPM":
        gpm = value
    elif from_unit == "LPM":
        gpm = value / 3.78541
    elif from_unit == "M3/HR":
        gpm = value * 4.40287

    # 從GPM轉換到目標單位
    if to_unit == "GPM":
        return gpm
    elif to_unit == "LPM":
        return gpm * 3.78541
    elif to_unit == "M3/HR":
        return gpm / 4.40287

def convert_head(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    elif from_unit == "FT" and to_unit == "M":
        return value * 0.3048
    elif from_unit == "M" and to_unit == "FT":
        return value / 0.3048

# 側邊欄設置
st.sidebar.header("數據輸入")

input_method = st.sidebar.radio(
    "選擇輸入方式:",
    ["手動輸入數據點", "上傳CSV文件"]
)

# 數據輸入部分
flow_unit = st.sidebar.selectbox("流量輸入單位:", ["GPM", "LPM", "M3/HR"])
head_unit = st.sidebar.selectbox("揚程輸入單位:", ["FT", "M"])

# 根據用戶選擇的輸入方式決定數據來源
if input_method == "手動輸入數據點":
    st.sidebar.subheader("輸入泵浦曲線數據點")
    
    # 初始預設點
    default_points = 5
    num_points = st.sidebar.slider("數據點數量", min_value=3, max_value=10, value=default_points)
    
    flow_data = []
    head_data = []
    
    # 創建輸入字段
    for i in range(num_points):
        col1, col2 = st.sidebar.columns(2)
        flow = col1.number_input(f"流量 {i+1} ({flow_unit})", value=float(i*100))
        head = col2.number_input(f"揚程 {i+1} ({head_unit})", value=float(max(100-(i*20), 10)))
        
        flow_data.append(flow)
        head_data.append(head)
    
    # 創建DataFrame
    data = pd.DataFrame({
        f"流量 ({flow_unit})": flow_data,
        f"揚程 ({head_unit})": head_data
    })
    
else:
    st.sidebar.subheader("上傳CSV文件")
    st.sidebar.markdown("CSV文件應包含兩列：流量和揚程")
    uploaded_file = st.sidebar.file_uploader("選擇CSV文件", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if len(data.columns) < 2:
                st.error("CSV文件應至少包含兩列數據：流量和揚程")
                data = None
            else:
                # 假設第一列是流量，第二列是揚程
                data.columns = [f"流量 ({flow_unit})", f"揚程 ({head_unit})"]
                st.sidebar.success("文件上傳成功！")
        except Exception as e:
            st.error(f"讀取CSV文件時出錯：{e}")
            data = None
    else:
        # 提供示例數據讓用戶有東西可看
        flow_data = [0, 100, 200, 300, 400]
        head_data = [100, 90, 75, 50, 20]
        data = pd.DataFrame({
            f"流量 ({flow_unit})": flow_data,
            f"揚程 ({head_unit})": head_data
        })

# 添加插值選項
st.sidebar.subheader("曲線設置")
interpolate = st.sidebar.checkbox("使用平滑曲線", value=True)
show_points = st.sidebar.checkbox("顯示數據點", value=True)
enable_grid = st.sidebar.checkbox("顯示網格", value=True)

# 展示數據表格
st.subheader("泵浦曲線數據")
st.dataframe(data)

# 創建多單位曲線圖
if data is not None:
    # 獲取原始單位數據
    flow_original = data.iloc[:, 0].values
    head_original = data.iloc[:, 1].values
    
    # 創建具有兩個Y軸的子圖
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 添加原始數據點和曲線
    if show_points:
        fig.add_trace(
            go.Scatter(
                x=flow_original,
                y=head_original,
                mode='markers',
                name=f'數據點 ({flow_unit}, {head_unit})',
                marker=dict(size=10, color='blue')
            ),
            secondary_y=False
        )
    
    # 生成平滑曲線用的更多X點
    if interpolate and len(flow_original) > 1:
        x_smooth = np.linspace(min(flow_original), max(flow_original), 100)
        # 使用numpy的插值
        y_smooth = np.interp(x_smooth, flow_original, head_original)
        
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=f'泵浦曲線 ({flow_unit}, {head_unit})',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False
        )
    else:
        # 不插值時使用原始數據連線
        fig.add_trace(
            go.Scatter(
                x=flow_original,
                y=head_original,
                mode='lines+markers' if show_points else 'lines',
                name=f'泵浦曲線 ({flow_unit}, {head_unit})',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False
        )
    
    # 添加轉換後的單位曲線
    # 流量單位轉換
    flow_units = ["GPM", "LPM", "M3/HR"]
    flow_colors = ['green', 'red', 'purple']
    
    for i, unit in enumerate(flow_units):
        if unit != flow_unit:  # 跳過原始單位
            # 轉換流量單位
            flow_converted = [convert_flow(f, flow_unit, unit) for f in flow_original]
            
            if interpolate and len(flow_original) > 1:
                x_smooth_converted = [convert_flow(f, flow_unit, unit) for f in x_smooth]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_smooth_converted,
                        y=y_smooth,
                        mode='lines',
                        name=f'泵浦曲線 ({unit}, {head_unit})',
                        line=dict(color=flow_colors[i], width=2, dash='dash')
                    ),
                    secondary_y=False
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=flow_converted,
                        y=head_original,
                        mode='lines',
                        name=f'泵浦曲線 ({unit}, {head_unit})',
                        line=dict(color=flow_colors[i], width=2, dash='dash')
                    ),
                    secondary_y=False
                )
    
    # 揚程單位轉換
    head_alt_unit = "M" if head_unit == "FT" else "FT"
    head_converted = [convert_head(h, head_unit, head_alt_unit) for h in head_original]
    
    if interpolate and len(flow_original) > 1:
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=[convert_head(h, head_unit, head_alt_unit) for h in y_smooth],
                mode='lines',
                name=f'泵浦曲線 ({flow_unit}, {head_alt_unit})',
                line=dict(color='orange', width=2, dash='dot')
            ),
            secondary_y=True
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=flow_original,
                y=head_converted,
                mode='lines',
                name=f'泵浦曲線 ({flow_unit}, {head_alt_unit})',
                line=dict(color='orange', width=2, dash='dot')
            ),
            secondary_y=True
        )
    
    # 更新軸標籤
    fig.update_layout(
        title="泵浦性能曲線 (Pump Performance Curve)",
        title_x=0.5,
        xaxis=dict(
            title=f"流量 (Flow Rate)",
            showgrid=enable_grid,
            zeroline=True
        ),
        yaxis=dict(
            title=f"揚程 (Head) - {head_unit}",
            showgrid=enable_grid,
            zeroline=True
        ),
        yaxis2=dict(
            title=f"揚程 (Head) - {head_alt_unit}",
            showgrid=False,
            zeroline=False,
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    # 添加軸說明文字
    fig.update_layout(
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.15,
                text="流量單位: GPM = 加侖/分鐘, LPM = 升/分鐘, M3/HR = 立方米/小時",
                showarrow=False,
                font=dict(size=12)
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.01,
                y=-0.20,
                text="揚程單位: FT = 英尺, M = 米",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    # 顯示圖表
    st.plotly_chart(fig, use_container_width=True)
    
    # 展示轉換後的數據表格
    show_conversions = st.checkbox("顯示所有單位轉換數據表", value=False)
    
    if show_conversions:
        st.subheader("單位轉換數據表")
        
        # 創建包含所有單位轉換的DataFrame
        conversion_data = pd.DataFrame()
        
        # 添加原始數據
        conversion_data[f"流量 ({flow_unit})"] = flow_original
        conversion_data[f"揚程 ({head_unit})"] = head_original
        
        # 添加流量轉換
        for unit in flow_units:
            if unit != flow_unit:
                conversion_data[f"流量 ({unit})"] = [convert_flow(f, flow_unit, unit) for f in flow_original]
        
        # 添加揚程轉換
        conversion_data[f"揚程 ({head_alt_unit})"] = head_converted
        
        st.dataframe(conversion_data)
    
    # 提供下載選項
    st.subheader("下載數據")
    
    # 準備下載表格的數據
    csv = conversion_data.to_csv(index=False) if 'conversion_data' in locals() else data.to_csv(index=False)
    
    st.download_button(
        label="下載CSV數據",
        data=csv,
        file_name="pump_curve_data.csv",
        mime="text/csv",
    )
else:
    st.error("未能載入有效的數據。請檢查您的輸入或上傳的文件。")

# 添加使用說明
with st.expander("使用說明"):
    st.markdown("""
    ### 如何使用這個工具：
    
    1. **輸入數據**：
       - 使用側邊欄中的「手動輸入數據點」或「上傳CSV文件」功能輸入泵浦曲線數據
       - 選擇適當的流量和揚程單位
       
    2. **調整圖表**：
       - 啟用/禁用「使用平滑曲線」以在數據點之間創建平滑曲線
       - 啟用/禁用「顯示數據點」以顯示或隱藏原始數據點
       - 啟用/禁用「顯示網格」以顯示或隱藏圖表網格線
       
    3. **查看轉換數據**：
       - 勾選「顯示所有單位轉換數據表」可查看所有單位下的數據值
       
    4. **下載數據**：
       - 使用「下載CSV數據」按鈕將當前數據保存為CSV文件
    
    ### 單位轉換：
    
    - **流量單位**：
      - GPM（加侖/分鐘）
      - LPM（升/分鐘）
      - M3/HR（立方米/小時）
    
    - **揚程單位**：
      - FT（英尺）
      - M（米）
    """)

# 添加頁腳
st.markdown("---")
st.markdown("© 2025 泵浦曲線繪製工具")
