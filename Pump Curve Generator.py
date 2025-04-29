import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import io

# 頁面設置
st.set_page_config(page_title="泵浦曲線圖匹配器", layout="wide")

# 標題
st.title("泵浦曲線圖精確匹配器")
st.markdown("此應用程序生成與範例圖片完全匹配的泵浦曲線圖")

# 頻率設置
frequency = st.radio("頻率:", ["50Hz", "60Hz"], horizontal=True)

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
def create_exact_match_chart():
    # 設置圖表樣式與尺寸（接近原圖的比例）
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # 設置背景與網格
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', color='gray', alpha=0.5)
    
    # 設置軸範圍
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 35)
    
    # 設置主軸刻度和標籤 - 匹配圖片
    # FT刻度 - 左側Y軸
    ft_ticks = [0, 5, 10, 15, 20, 25, 30, 34]
    ax.set_yticks(ft_ticks)
    ax.set_yticklabels([str(tick) for tick in ft_ticks])
    
    # 流量刻度 - X軸
    flow_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ax.set_xticks(flow_ticks)
    ax.set_xticklabels([str(tick) if tick % 200 == 0 or tick == 0 else "" for tick in flow_ticks])
    
    # 創建第二個Y軸 - 用於M值
    ax2 = ax.twinx()
    ax2.set_ylim(0, 10)
    m_ticks = [0, 2, 4, 6, 8, 10]
    ax2.set_yticks(m_ticks)
    ax2.set_yticklabels([str(tick) for tick in m_ticks])
    
    # 添加坐標軸標籤
    ax.set_ylabel('HEAD\n(FT) ( M )', fontsize=12)
    
    # 繪製每個泵浦型號的曲線
    for model_name, data in pump_models.items():
        # 獲取數據
        flow = data["flow"]
        head_m = data["head"]
        
        # 將米轉換為英尺
        head_ft = [h * 3.28084 for h in head_m]
        
        # 生成平滑曲線的點
        x_smooth = np.linspace(min(flow), max(flow), 100)
        y_smooth_ft = np.interp(x_smooth, flow, head_ft)
        
        # 繪製曲線
        line, = ax.plot(x_smooth, y_smooth_ft, 'k-', linewidth=2)
        
        # 找到合適位置添加型號標籤
        # 找到曲線中間位置
        idx = len(x_smooth) // 2
        text_x = x_smooth[idx]
        text_y = y_smooth_ft[idx]
        
        # 計算角度使文字沿著曲線方向
        dx = x_smooth[idx+5] - x_smooth[idx-5]
        dy = y_smooth_ft[idx+5] - y_smooth_ft[idx-5]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 調整角度使文字可讀
        if angle < -30:
            angle += 180
        
        # 添加型號標籤
        plt.text(text_x, text_y, model_name, fontsize=12, 
                 rotation=angle, rotation_mode='anchor',
                 ha='center', va='center')
    
    # 添加頻率標註
    plt.text(800, 30, frequency, fontsize=14)
    
    # 添加底部流量刻度及標籤
    # 底部X軸添加M³/h刻度
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    m3h_max = 60
    m3h_ticks = list(range(0, m3h_max+1, 5))
    m3h_positions = [pos * (1000/m3h_max) for pos in m3h_ticks]
    ax3.set_xticks(m3h_positions)
    ax3.set_xticklabels([str(tick) for tick in m3h_ticks])
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_position(('outward', 40))
    
    # 底部添加GPM刻度
    ax4 = ax.twiny()
    ax4.set_xlim(ax.get_xlim())
    gpm_max = 250
    gpm_ticks = list(range(0, gpm_max+1, 50))
    gpm_positions = [pos * (1000/gpm_max) for pos in gpm_ticks]
    ax4.set_xticks(gpm_positions)
    ax4.set_xticklabels([str(tick) for tick in gpm_ticks])
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_position(('outward', 80))
    
    # 添加單位標籤
    fig.text(0.94, 0.02, '(LPM)', fontsize=10)
    fig.text(0.94, 0.06, '(GPM)', fontsize=10)
    fig.text(0.94, 0.1, '(M³/h)', fontsize=10)
    fig.text(0.5, 0.01, 'FLOW', fontsize=12, ha='center')
    
    # 添加藍色"Curve"標題
    # 創建一個藍色的矩形
    ax_title = fig.add_axes([0.05, 0.92, 0.2, 0.06])
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    ax_title.set_facecolor('blue')
    ax_title.text(0.5, 0.5, 'Curve', color='white', fontsize=14,
                 ha='center', va='center')
    ax_title.spines['top'].set_visible(False)
    ax_title.spines['right'].set_visible(False)
    ax_title.spines['bottom'].set_visible(False)
    ax_title.spines['left'].set_visible(False)
    
    # 添加灰色外框
    rect = patches.Rectangle((0.05, 0.05), 0.9, 0.83, 
                            linewidth=1, edgecolor='gray', facecolor='none',
                            transform=fig.transFigure)
    fig.patches.append(rect)
    
    # 調整佈局
    plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.9])
    
    # 返回圖表
    return fig

# 生成圖表
fig = create_exact_match_chart()

# 顯示圖表
st.pyplot(fig)

# 保存圖表到臨時文件
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
buffer.seek(0)

# 下載按鈕
st.download_button(
    label="下載泵浦曲線圖 (PNG)",
    data=buffer,
    file_name=f"pump_curve_{frequency}.png",
    mime="image/png",
)

# 添加數據表格
show_data = st.checkbox("顯示曲線數據")
if show_data:
    st.subheader("泵浦曲線數據表")
    for model, data in pump_models.items():
        flow_lpm = data["flow"]
        head_m = data["head"]
        # 計算轉換值
        head_ft = [round(h * 3.28084, 1) for h in head_m]
        flow_gpm = [round(f / 3.78541, 1) for f in flow_lpm]
        flow_m3h = [round(f * 0.06, 2) for f in flow_lpm]
        
        # 創建數據表
        df_data = {
            "流量 (LPM)": flow_lpm,
            "流量 (GPM)": flow_gpm,
            "流量 (M³/h)": flow_m3h,
            "揚程 (M)": head_m,
            "揚程 (FT)": head_ft
        }
        
        st.write(f"**{model}**")
        st.dataframe(df_data)

# 添加簡單說明
st.markdown("---")
st.markdown("""
**說明**: 此應用程序生成的圖表精確匹配了範例圖片中的泵浦曲線圖。包含:
- DS-05、DS-10、DS-20 和 DS-30 四種型號的泵浦曲線
- 流量單位: LPM (主要)、GPM 和 M³/h (底部刻度)
- 揚程單位: FT (左側) 和 M (右側)
- 與原圖片相同的刻度和網格
- 相似的藍色標題和整體佈局
""")
