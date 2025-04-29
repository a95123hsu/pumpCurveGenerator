import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import io
import pandas as pd
from matplotlib.font_manager import FontProperties

# 頁面設置 - 使用寬屏以提供更好的圖表展示
st.set_page_config(page_title="精確泵浦曲線圖生成器", layout="wide", initial_sidebar_state="collapsed")

# 自定義頁面樣式
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stRadio [role=radiogroup] {
        margin-bottom: 1.5rem;
    }
    .download-section {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .footer {
        margin-top: 2rem;
        color: #666;
        text-align: center;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 應用程序標題和說明
col1, col2 = st.columns([3, 1])
with col1:
    st.title("精確泵浦曲線圖生成器")
    st.markdown("此應用程序可生成與實際產品規格表完全匹配的泵浦曲線圖")

# 設置和自定義選項
with col2:
    # 頻率設置
    frequency = st.radio("運行頻率:", ["50Hz", "60Hz"], horizontal=True)
    
    # 圖表尺寸設置
    chart_size = st.radio("圖表尺寸:", ["標準 (800x600)", "大 (1200x900)", "打印適用 (2400x1800)"], index=0, horizontal=True)

# 預設泵浦型號數據 - 精確匹配圖片中的曲線
pump_models = {
    "DS-30": {
        "flow": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        "head": [9.0, 8.9, 8.6, 8.0, 7.2, 6.2, 5.0, 3.7, 2.0, 0.3]
    },
    "DS-20": {
        "flow": [0, 100, 200, 300, 400, 500, 600, 700],
        "head": [7.5, 7.3, 7.0, 6.5, 5.5, 4.3, 2.8, 0.8]
    },
    "DS-10": {
        "flow": [0, 50, 100, 150, 200, 250, 300, 350, 400],
        "head": [6.0, 5.9, 5.7, 5.4, 4.9, 4.2, 3.3, 2.2, 0.8]
    },
    "DS-05": {
        "flow": [0, 50, 100, 150, 175, 200, 220, 240],
        "head": [4.2, 4.1, 3.8, 3.3, 2.9, 2.3, 1.6, 0.8]
    }
}

# 創建圖表函數
def create_exact_match_chart(dpi=100):
    # 根據選擇的尺寸設置圖表大小
    if chart_size == "標準 (800x600)":
        figsize = (10, 8)
        dpi = 100
    elif chart_size == "大 (1200x900)":
        figsize = (12, 9.6)
        dpi = 150
    else:  # 打印適用
        figsize = (12, 9.6)
        dpi = 300

    # 設置圖表樣式與尺寸
    plt.style.use('default')
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # 創建主圖表區域
    ax = fig.add_subplot(111)
    
    # 設置背景與網格
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', color='gray', alpha=0.5, which='both')
    
    # 設置軸範圍
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 35)
    
    # 設置主軸刻度和標籤 - 精確匹配圖片
    # FT刻度 - 左側Y軸
    ft_ticks = [0, 5, 10, 15, 20, 25, 30, 34]
    ax.set_yticks(ft_ticks)
    ax.set_yticklabels([str(tick) for tick in ft_ticks], fontsize=10)
    
    # 流量刻度 - X軸
    flow_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ax.set_xticks(flow_ticks)
    ax.set_xticklabels([str(tick) if tick % 200 == 0 or tick == 0 else "" for tick in flow_ticks], fontsize=10)
    
    # 創建第二個Y軸 - 用於M值
    ax2 = ax.twinx()
    ax2.set_ylim(0, 10)
    m_ticks = [0, 2, 4, 6, 8, 10]
    ax2.set_yticks(m_ticks)
    ax2.set_yticklabels([str(tick) for tick in m_ticks], fontsize=10)
    
    # 添加坐標軸標籤
    ax.set_ylabel('HEAD\n(FT) ( M )', fontsize=12, fontweight='bold')
    
    # 繪製每個泵浦型號的曲線
    for model_name, data in pump_models.items():
        # 獲取數據
        flow = data["flow"]
        head_m = data["head"]
        
        # 將米轉換為英尺
        head_ft = [h * 3.28084 for h in head_m]
        
        # 生成平滑曲線的點 - 使用更多點以獲得更平滑的曲線
        x_smooth = np.linspace(min(flow), max(flow), 200)
        y_smooth_ft = np.interp(x_smooth, flow, head_ft)
        
        # 繪製曲線 - 使用黑色粗線
        line, = ax.plot(x_smooth, y_smooth_ft, 'k-', linewidth=2)
        
        # 找到適合放置標籤的位置 - 為每個型號自定義位置以避免重疊
        if model_name == "DS-30":
            idx = int(len(x_smooth) * 0.4)  # DS-30放在前半部分
        elif model_name == "DS-20":
            idx = int(len(x_smooth) * 0.45)
        elif model_name == "DS-10":
            idx = int(len(x_smooth) * 0.5)
        else:  # DS-05
            idx = int(len(x_smooth) * 0.55)
            
        text_x = x_smooth[idx]
        text_y = y_smooth_ft[idx]
        
        # 計算角度使文字沿著曲線方向
        dx = x_smooth[min(idx+15, len(x_smooth)-1)] - x_smooth[max(idx-15, 0)]
        dy = y_smooth_ft[min(idx+15, len(x_smooth)-1)] - y_smooth_ft[max(idx-15, 0)]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 調整角度使文字可讀
        if angle < -45:
            angle += 180
        
        # 添加型號標籤 - 使用斜體,匹配原圖
        plt.text(text_x, text_y, model_name, fontsize=12, 
                 rotation=angle, rotation_mode='anchor',
                 ha='center', va='center', 
                 fontweight='bold', style='italic')
    
    # 添加頻率標註
    plt.text(800, 30, frequency, fontsize=14, fontweight='bold')
    
    # 添加底部流量刻度及標籤
    # 底部X軸添加M³/h刻度
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    m3h_max = 60
    m3h_ticks = list(range(0, m3h_max+1, 5))
    m3h_positions = [pos * (1000/m3h_max) for pos in m3h_ticks]
    ax3.set_xticks(m3h_positions)
    ax3.set_xticklabels([str(tick) for tick in m3h_ticks], fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_position(('outward', 40))
    
    # 底部添加GPM刻度
    ax4 = ax.twiny()
    ax4.set_xlim(ax.get_xlim())
    gpm_max = 250
    gpm_ticks = list(range(0, gpm_max+1, 50))
    gpm_positions = [pos * (1000/gpm_max) for pos in gpm_ticks]
    ax4.set_xticks(gpm_positions)
    ax4.set_xticklabels([str(tick) for tick in gpm_ticks], fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_position(('outward', 80))
    
    # 添加單位標籤 - 精確定位匹配原圖
    fig.text(0.95, 0.02, '(LPM)', fontsize=10)
    fig.text(0.95, 0.06, '(GPM)', fontsize=10)
    fig.text(0.95, 0.1, '(M³/h)', fontsize=10)
    fig.text(0.5, 0.01, 'FLOW', fontsize=12, ha='center', fontweight='bold')
    
    # 添加藍色"Curve"標題 - 更精確匹配原圖
    # 創建一個藍色的矩形
    ax_title = fig.add_axes([0.05, 0.92, 0.18, 0.05])
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    ax_title.set_facecolor('#0047AB')  # 更接近原圖的藍色
    ax_title.text(0.5, 0.5, 'Curve', color='white', fontsize=14,
                 ha='center', va='center', fontweight='bold')
    ax_title.spines['top'].set_visible(False)
    ax_title.spines['right'].set_visible(False)
    ax_title.spines['bottom'].set_visible(False)
    ax_title.spines['left'].set_visible(False)
    
    # 添加灰色外框 - 更精確匹配原圖
    rect = patches.Rectangle((0.05, 0.05), 0.9, 0.83, 
                            linewidth=1, edgecolor='gray', facecolor='none',
                            transform=fig.transFigure)
    fig.patches.append(rect)
    
    # 添加細線條紋背景 - 匹配原圖
    for i in range(20, 100, 3):
        line = Line2D([0.05, 0.95], [i/100, i/100], 
                      transform=fig.transFigure, 
                      color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
        fig.lines.append(line)
    
    # 調整佈局
    plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.9])
    
    # 返回圖表
    return fig

# 生成圖表
fig = create_exact_match_chart()

# 顯示圖表 - 使用全寬顯示
st.pyplot(fig, use_container_width=True)

# 下載選項部分
st.markdown("<div class='download-section'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # 保存為PNG
    buffer_png = io.BytesIO()
    fig.savefig(buffer_png, format='png', dpi=300, bbox_inches='tight')
    buffer_png.seek(0)
    
    st.download_button(
        label="下載PNG圖片",
        data=buffer_png,
        file_name=f"pump_curve_{frequency}.png",
        mime="image/png",
    )

with col2:
    # 保存為PDF
    buffer_pdf = io.BytesIO()
    fig.savefig(buffer_pdf, format='pdf', bbox_inches='tight')
    buffer_pdf.seek(0)
    
    st.download_button(
        label="下載PDF文檔",
        data=buffer_pdf,
        file_name=f"pump_curve_{frequency}.pdf",
        mime="application/pdf",
    )

with col3:
    # 保存為SVG (用於進一步編輯)
    buffer_svg = io.BytesIO()
    fig.savefig(buffer_svg, format='svg', bbox_inches='tight')
    buffer_svg.seek(0)
    
    st.download_button(
        label="下載SVG矢量圖",
        data=buffer_svg,
        file_name=f"pump_curve_{frequency}.svg",
        mime="image/svg+xml",
    )

# 添加數據表格和詳細資訊
tabs = st.tabs(["性能數據", "技術規格", "使用說明"])

with tabs[0]:
    st.subheader("泵浦曲線性能數據")
    
    # 轉換所有數據並顯示
    for model, data in pump_models.items():
        flow_lpm = data["flow"]
        head_m = data["head"]
        
        # 計算轉換值
        head_ft = [round(h * 3.28084, 2) for h in head_m]
        flow_gpm = [round(f / 3.78541, 2) for f in flow_lpm]
        flow_m3h = [round(f * 0.06, 2) for f in flow_lpm]
        
        # 創建數據表
        df_data = {
            "流量 (LPM)": flow_lpm,
            "流量 (GPM)": flow_gpm,
            "流量 (M³/h)": flow_m3h,
            "揚程 (M)": head_m,
            "揚程 (FT)": head_ft
        }
        
        df = pd.DataFrame(df_data)
        
        # 顯示數據
        st.write(f"**{model} - {frequency} 性能數據**")
        st.dataframe(df, use_container_width=True)
        
        # 添加下載此型號的數據選項
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"下載 {model} 數據 (CSV)",
            data=csv,
            file_name=f"{model}_{frequency}_data.csv",
            mime="text/csv",
            key=f"dl_{model}"
        )
        st.markdown("---")

with tabs[1]:
    st.subheader("泵浦型號技術規格")
    
    # 添加技術規格表
    tech_specs = {
        "型號": ["DS-05", "DS-10", "DS-20", "DS-30"],
        "最大流量 (LPM)": [240, 400, 700, 900],
        "最大揚程 (M)": [4.2, 6.0, 7.5, 9.0],
        "功率 (kW)": [0.5, 1.0, 2.0, 3.0],
        "管徑 (mm)": [25, 32, 40, 50],
        "重量 (kg)": [8.5, 12.3, 18.7, 26.2],
        "尺寸 (mm)": ["320x210x280", "350x230x300", "380x250x320", "420x280x350"]
    }
    
    df_tech = pd.DataFrame(tech_specs)
    st.dataframe(df_tech, use_container_width=True)
    
    # 補充說明
    st.markdown("""
    ### 適用介質
    * 清水
    * 輕度含沙水（濃度 ≤ 0.1%）
    * pH值範圍: 6.5-8.5
    
    ### 工作環境
    * 環境溫度: 0°C ~ 40°C
    * 介質溫度: 0°C ~ 80°C
    * 最大工作壓力: 10 bar
    
    ### 電氣規格
    * 電源: 三相 380V ±10%, 50Hz/60Hz
    * 防護等級: IP55
    * 絕緣等級: F級
    """)

with tabs[2]:
    st.subheader("使用說明")
    
    # 添加使用說明
    st.markdown("""
    ### 泵浦曲線圖解讀方法
    
    泵浦曲線圖表示了不同流量下泵浦的揚程性能。以下是如何閱讀和使用該圖表：
    
    1. **選擇合適的型號**：根據您的需求選擇適當的泵浦型號（DS-05、DS-10、DS-20或DS-30）
    
    2. **確定工作點**：
       - 水平軸（X軸）表示流量，單位可以是LPM（升/分鐘）、GPM（加侖/分鐘）或M³/h（立方米/小時）
       - 垂直軸（Y軸）表示揚程，單位可以是M（米）或FT（英尺）
       - 您的系統工作點是系統阻力曲線與泵浦曲線的交點
    
    3. **評估性能**：
       - 最佳效率點通常在曲線的中間部分
       - 避免在最大流量接近曲線末端處運行泵浦
       - 選擇的工作點應確保泵浦在其設計範圍內運行
    
    ### 如何使用此應用程序
    
    1. 選擇運行頻率（50Hz或60Hz）
    2. 選擇所需的圖表尺寸
    3. 下載所需格式的圖表（PNG、PDF或SVG）
    4. 在「性能數據」標籤中查看詳細的數值數據
    5. 在「技術規格」標籤中查看每個型號的技術參數
    
    ### 注意事項
    
    * 實際安裝時，請考慮管道損失、閥門損失和高度差
    * 為確保泵浦長壽命，應避免空轉和汽蝕現象
    * 定期維護可以確保泵浦保持最佳效率
    """)

# 添加頁腳
st.markdown("<div class='footer'>© 2025 泵浦曲線圖生成器 | 版本 2.0</div>", unsafe_allow_html=True)
