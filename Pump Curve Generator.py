import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
import base64

st.set_page_config(page_title="泵浦曲線繪製工具", layout="wide")

st.title("泵浦曲線繪製工具 (Pump Curve Chart Generator)")
st.markdown("這個應用程序可以幫助您繪製多條泵浦性能曲線，並可導出為圖像。")

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
st.sidebar.header("圖表設置")

# 頻率設置
frequency = st.sidebar.radio("頻率:", ["50Hz", "60Hz"], index=0)

# 多曲線設置
st.sidebar.subheader("泵浦型號設置")
num_models = st.sidebar.slider("泵浦型號數量", min_value=1, max_value=6, value=4)

# 保存泵浦型號數據的字典
pump_models = {}

# 為每個泵浦型號創建輸入表單
for i in range(num_models):
    st.sidebar.markdown(f"**泵浦型號 {i+1}**")
    model_name = st.sidebar.text_input(f"型號名稱 #{i+1}", value=f"DS-{(i+1)*5}" if i < 4 else f"Model-{i+1}")
    
    # 每個型號的數據點數量
    num_points = st.sidebar.slider(f"型號 {i+1} 數據點數量", min_value=3, max_value=10, value=5, key=f"points_{i}")
    
    # 為此型號創建數據收集點
    flow_data = []
    head_data = []
    
    # 根據型號自動生成一些合理的默認值
    max_flow = 300 * (i + 1)
    max_head = 10 - (i * 0.5) if i < 5 else 7
    
    # 創建折疊區域以減少側邊欄空間
    with st.sidebar.expander(f"編輯型號 {i+1} 數據點"):
        for j in range(num_points):
            # 計算默認值使曲線看起來合理
            default_flow = max_flow * j / (num_points - 1)
            # 使頭部曲線隨流量增加而下降
            default_head = max_head * (1 - (j / (num_points - 1))**1.5)
            
            col1, col2 = st.columns(2)
            flow = col1.number_input(
                f"流量 {j+1}", 
                value=float(default_flow),
                key=f"flow_{i}_{j}"
            )
            head = col2.number_input(
                f"揚程 {j+1}", 
                value=float(default_head),
                key=f"head_{i}_{j}"
            )
            
            flow_data.append(flow)
            head_data.append(head)
    
    # 保存此型號的數據到字典
    pump_models[model_name] = {
        "flow": flow_data,
        "head": head_data,
        "color": f"hsl({(i * 40) % 360}, 70%, 50%)"  # 為每個型號分配不同的顏色
    }

# 輸入單位設置
st.sidebar.subheader("單位設置")
flow_unit = st.sidebar.selectbox("流量輸入單位:", ["LPM", "GPM", "M3/HR"], index=0)
head_unit = st.sidebar.selectbox("揚程輸入單位:", ["M", "FT"], index=0)

# 圖表外觀設置
st.sidebar.subheader("圖表外觀")
interpolate = st.sidebar.checkbox("使用平滑曲線", value=True)
show_points = st.sidebar.checkbox("顯示數據點", value=False)
enable_grid = st.sidebar.checkbox("顯示網格", value=True)
show_legend = st.sidebar.checkbox("顯示圖例", value=True)

# 輸出設置
st.sidebar.subheader("輸出設置")
chart_width = st.sidebar.slider("圖表寬度 (像素)", min_value=600, max_value=1200, value=900)
chart_height = st.sidebar.slider("圖表高度 (像素)", min_value=400, max_value=1000, value=700)
bg_color = st.sidebar.color_picker("背景顏色", "#FFFFFF")
grid_color = st.sidebar.color_picker("網格顏色", "#CCCCCC")

# 圖表區域設置
chart_title = st.text_input("圖表標題", f"泵浦性能曲線 ({frequency})")

# 創建多單位曲線圖
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 添加每個泵浦型號的曲線
for model_name, model_data in pump_models.items():
    flow_original = np.array(model_data["flow"])
    head_original = np.array(model_data["head"])
    color = model_data["color"]
    
    # 添加原始數據點
    if show_points:
        fig.add_trace(
            go.Scatter(
                x=flow_original,
                y=head_original,
                mode='markers',
                name=f'{model_name} 數據點',
                marker=dict(size=8, color=color),
                showlegend=False
            ),
            secondary_y=False
        )
    
    # 添加曲線
    if interpolate and len(flow_original) > 1:
        # 生成平滑曲線用的更多X點
        x_smooth = np.linspace(min(flow_original), max(flow_original), 100)
        # 使用numpy的插值
        y_smooth = np.interp(x_smooth, flow_original, head_original)
        
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=model_name,
                line=dict(color=color, width=3),
                hovertemplate=f'{model_name}<br>流量: %{{x:.1f}} {flow_unit}<br>揚程: %{{y:.2f}} {head_unit}'
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
                name=model_name,
                line=dict(color=color, width=3),
                hovertemplate=f'{model_name}<br>流量: %{{x:.1f}} {flow_unit}<br>揚程: %{{y:.2f}} {head_unit}'
            ),
            secondary_y=False
        )

# 計算軸的範圍
all_flows = []
all_heads = []
for model in pump_models.values():
    all_flows.extend(model["flow"])
    all_heads.extend(model["head"])

# 計算適當的坐標軸範圍，添加一些邊距
max_flow = max(all_flows) * 1.1 if all_flows else 1000
max_head = max(all_heads) * 1.1 if all_heads else 40
min_flow = 0  # 通常流量從0開始
min_head = 0  # 通常揚程從0開始

# 創建轉換單位的標記
head_alt_unit = "FT" if head_unit == "M" else "M"
flow_units = ["LPM", "GPM", "M3/HR"]
other_flow_units = [u for u in flow_units if u != flow_unit]

# 更新第二軸的數據 (揚程轉換)
head_conversion_factor = 3.28084 if head_unit == "M" else 0.3048  # M to FT or FT to M
max_head_alt = max_head * head_conversion_factor

# 更新佈局
fig.update_layout(
    title=chart_title,
    title_x=0.5,
    width=chart_width,
    height=chart_height,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    xaxis=dict(
        title=f"流量 (Flow)",
        showgrid=enable_grid,
        gridcolor=grid_color,
        zeroline=True,
        zerolinecolor=grid_color,
        range=[min_flow, max_flow],
        showticklabels=True
    ),
    yaxis=dict(
        title=f"揚程 (Head) ({head_unit})",
        showgrid=enable_grid,
        gridcolor=grid_color,
        zeroline=True,
        zerolinecolor=grid_color,
        range=[min_head, max_head],
        showticklabels=True
    ),
    yaxis2=dict(
        title=f"揚程 (Head) ({head_alt_unit})",
        showgrid=False,
        range=[min_head, max_head_alt],
        overlaying="y",
        side="right",
        showticklabels=True
    ),
    hovermode="closest",
    showlegend=show_legend,
    legend=dict(
        orientation="h" if num_models <= 4 else "v",
        yanchor="bottom" if num_models <= 4 else "top",
        y=-0.2 if num_models <= 4 else 1,
        xanchor="center" if num_models <= 4 else "left",
        x=0.5 if num_models <= 4 else 1.05
    )
)

# 添加其他單位的刻度
# 首先，計算轉換因子
flow_conversion_factors = {
    "GPM-LPM": 3.78541,  # GPM to LPM
    "LPM-GPM": 1/3.78541,  # LPM to GPM
    "GPM-M3/HR": 1/4.40287,  # GPM to M3/HR
    "M3/HR-GPM": 4.40287,  # M3/HR to GPM
    "LPM-M3/HR": 0.06,  # LPM to M3/HR (近似)
    "M3/HR-LPM": 1/0.06  # M3/HR to LPM (近似)
}

# 添加X軸次級刻度線
if flow_unit == "LPM":
    # 添加GPM刻度線
    secondary_ticks_gpm = np.linspace(0, max_flow * flow_conversion_factors["LPM-GPM"], 6)
    secondary_ticks_gpm = np.round(secondary_ticks_gpm, 0)
    
    # 添加M3/HR刻度線
    secondary_ticks_m3hr = np.linspace(0, max_flow * flow_conversion_factors["LPM-M3/HR"], 6)
    secondary_ticks_m3hr = np.round(secondary_ticks_m3hr, 1)
    
    fig.update_layout(
        xaxis=dict(
            tickvals=np.linspace(0, max_flow, 6),
            ticktext=[f"{int(x)}" for x in np.linspace(0, max_flow, 6)]
        )
    )
    
    # 添加M3/HR的第二個刻度軸
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 添加刻度標記
    for i, (gpm, m3hr) in enumerate(zip(secondary_ticks_gpm, secondary_ticks_m3hr)):
        position = i / 5  # 從0到1的相對位置
        
        # 添加GPM刻度註釋
        fig.add_annotation(
            x=position * max_flow,
            y=0,
            text=f"{int(gpm)}",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            yshift=-40,
            font=dict(size=10)
        )
        
        # 添加M3/HR刻度註釋
        fig.add_annotation(
            x=position * max_flow,
            y=0,
            text=f"{m3hr:.1f}",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            yshift=-70,
            font=dict(size=10)
        )
    
    # 添加單位標籤
    fig.add_annotation(
        x=max_flow * 1.02,
        y=0,
        text=f"({flow_unit})",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        yshift=-10,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=max_flow * 1.02,
        y=0,
        text=f"(GPM)",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        yshift=-40,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=max_flow * 1.02,
        y=0,
        text=f"(M³/hr)",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        yshift=-70,
        font=dict(size=12)
    )
    
    # 添加單位標籤
    fig.add_annotation(
        x=max_flow / 2,
        y=0,
        text="FLOW",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        yshift=-90,
        font=dict(size=12, color="black")
    )

elif flow_unit == "GPM":
    # 類似的代碼用於GPM作為主單位的情況
    secondary_ticks_lpm = np.linspace(0, max_flow * flow_conversion_factors["GPM-LPM"], 6)
    secondary_ticks_lpm = np.round(secondary_ticks_lpm, 0)
    
    secondary_ticks_m3hr = np.linspace(0, max_flow * flow_conversion_factors["GPM-M3/HR"], 6)
    secondary_ticks_m3hr = np.round(secondary_ticks_m3hr, 1)
    
    # 類似的註釋代碼...

elif flow_unit == "M3/HR":
    # 類似的代碼用於M3/HR作為主單位的情況
    secondary_ticks_lpm = np.linspace(0, max_flow * flow_conversion_factors["M3/HR-LPM"], 6)
    secondary_ticks_lpm = np.round(secondary_ticks_lpm, 0)
    
    secondary_ticks_gpm = np.linspace(0, max_flow * flow_conversion_factors["M3/HR-GPM"], 6)
    secondary_ticks_gpm = np.round(secondary_ticks_gpm, 0)
    
    # 類似的註釋代碼...

# 添加頻率註釋
fig.add_annotation(
    x=max_flow * 0.85,
    y=max_head * 0.85,
    text=frequency,
    showarrow=False,
    font=dict(size=20)
)

# 在新列顯示圖表
st.plotly_chart(fig, use_container_width=False)

# 添加導出圖像功能
st.subheader("導出圖像")

# 使用matplotlib替代方法生成圖片
def save_plotly_fig_as_image(fig):
    # 將Plotly圖表保存為HTML
    temp_html = "temp_figure.html"
    fig.write_html(temp_html)
    
    # 使用Streamlit的下載功能
    with open(temp_html, "rb") as file:
        html_content = file.read()
    
    # 提供HTML檔案下載
    st.download_button(
        label="下載泵浦曲線圖 (HTML格式)",
        data=html_content,
        file_name="pump_curve_chart.html",
        mime="text/html",
    )
    
    # 可選嘗試使用matplotlib再創建圖表
    try:
        import matplotlib.pyplot as plt
        
        # 創建matplotlib圖表
        plt.figure(figsize=(chart_width/100, chart_height/100))
        
        # 繪製每個泵浦型號的曲線
        for model_name, model_data in pump_models.items():
            flow_data = model_data["flow"]
            head_data = model_data["head"]
            
            if interpolate and len(flow_data) > 1:
                # 生成插值用的點
                x_smooth = np.linspace(min(flow_data), max(flow_data), 100)
                y_smooth = np.interp(x_smooth, flow_data, head_data)
                plt.plot(x_smooth, y_smooth, label=model_name)
            else:
                plt.plot(flow_data, head_data, label=model_name)
                
            if show_points:
                plt.scatter(flow_data, head_data, s=30)
        
        # 設置坐標軸標籤和標題
        plt.xlabel(f"流量 (Flow) - {flow_unit}")
        plt.ylabel(f"揚程 (Head) - {head_unit}")
        plt.title(chart_title)
        plt.grid(enable_grid)
        
        # 添加頻率標註
        plt.text(max_flow * 0.8, max_head * 0.8, frequency, fontsize=14)
        
        # 添加圖例
        if show_legend:
            plt.legend()
        
        # 保存為臨時檔案
        plt_filename = "pump_curve_plot.png"
        plt.savefig(plt_filename, dpi=300, bbox_inches="tight")
        plt.close()
        
        # 提供PNG檔案下載
        with open(plt_filename, "rb") as file:
            img_data = file.read()
        
        st.download_button(
            label="下載泵浦曲線圖 (PNG格式)",
            data=img_data,
            file_name="pump_curve_chart.png",
            mime="image/png",
        )
    except Exception as e:
        st.warning(f"無法生成PNG圖像，請使用HTML格式。錯誤：{e}")

# 調用函數代替直接使用to_image
save_plotly_fig_as_image(fig)

# 添加數據表顯示
show_data_tables = st.checkbox("顯示所有泵浦型號數據", value=False)

if show_data_tables:
    st.subheader("泵浦型號數據表")
    
    for model_name, model_data in pump_models.items():
        # 創建此型號的DataFrame
        df = pd.DataFrame({
            f"流量 ({flow_unit})": model_data["flow"],
            f"揚程 ({head_unit})": model_data["head"]
        })
        
        # 添加GPM轉換（如果適用）
        if flow_unit != "GPM":
            df[f"流量 (GPM)"] = [convert_flow(f, flow_unit, "GPM") for f in model_data["flow"]]
        
        # 添加LPM轉換（如果適用）
        if flow_unit != "LPM":
            df[f"流量 (LPM)"] = [convert_flow(f, flow_unit, "LPM") for f in model_data["flow"]]
        
        # 添加M3/HR轉換（如果適用）
        if flow_unit != "M3/HR":
            df[f"流量 (M3/HR)"] = [convert_flow(f, flow_unit, "M3/HR") for f in model_data["flow"]]
        
        # 添加另一個揚程單位的轉換
        head_alt_unit = "FT" if head_unit == "M" else "M"
        df[f"揚程 ({head_alt_unit})"] = [convert_head(h, head_unit, head_alt_unit) for h in model_data["head"]]
        
        # 顯示數據表
        st.markdown(f"**型號: {model_name}**")
        st.dataframe(df)

# 添加CSV導出功能
if pump_models:
    st.subheader("導出數據為CSV")
    
    # 準備所有模型的數據
    all_data = []
    for model_name, model_data in pump_models.items():
        for i, (flow, head) in enumerate(zip(model_data["flow"], model_data["head"])):
            all_data.append({
                "型號": model_name,
                f"流量 ({flow_unit})": flow,
                f"揚程 ({head_unit})": head,
                f"流量 (GPM)": convert_flow(flow, flow_unit, "GPM") if flow_unit != "GPM" else flow,
                f"流量 (LPM)": convert_flow(flow, flow_unit, "LPM") if flow_unit != "LPM" else flow,
                f"流量 (M3/HR)": convert_flow(flow, flow_unit, "M3/HR") if flow_unit != "M3/HR" else flow,
                f"揚程 ({head_alt_unit})": convert_head(head, head_unit, head_alt_unit)
            })
    
    # 創建DataFrame
    all_df = pd.DataFrame(all_data)
    
    # 提供CSV下載
    csv = all_df.to_csv(index=False)
    st.download_button(
        label="下載所有泵浦數據 (CSV)",
        data=csv,
        file_name="pump_curves_data.csv",
        mime="text/csv",
    )

# 添加使用說明
with st.expander("使用說明"):
    st.markdown("""
    ### 如何使用這個工具：
    
    1. **設置泵浦型號**：
       - 使用側邊欄設置您想要顯示的泵浦型號數量
       - 為每個型號命名並設置數據點
       
    2. **調整圖表外觀**：
       - 設置圖表尺寸、背景顏色和網格顏色
       - 啟用/禁用平滑曲線、數據點和網格
       
    3. **更改單位**：
       - 選擇您偏好的流量和揚程單位
       - 圖表將自動顯示所有單位的刻度
       
    4. **導出結果**：
       - 點擊"下載泵浦曲線圖"按鈕以PNG格式保存圖表
       - 也可以下載所有數據為CSV格式
    
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
