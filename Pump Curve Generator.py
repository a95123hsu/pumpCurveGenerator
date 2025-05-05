import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.interpolate import interp1d

def main():
    st.set_page_config(page_title="泵曲線生成器", layout="wide")
    
    st.title("泵曲線生成器工具 / Pump Curve Generator")
    
    st.markdown("""
    此工具允許您生成泵性能曲線，支援不同模型具有不同測量點的數據。
    """)
    
    # Chart configuration
    st.subheader("圖表配置 / Chart Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chart_style = st.selectbox("圖表樣式 / Chart Style", ["現代 / Modern", "經典 / Classic"])
        chart_style = "Modern" if "Modern" in chart_style else "Classic"
    
    with col2:
        show_grid = st.checkbox("顯示網格 / Show Grid", value=True)
    
    with col3:
        show_system = st.checkbox("顯示系統曲線 / Show System Curve", value=False)
    
    # System curve parameters
    if show_system:
        col4, col5 = st.columns(2)
        
        with col4:
            static_head = st.number_input("靜壓頭 / Static Head (m)", 
                                      min_value=0.0, value=2.0, step=0.5)
        
        with col5:
            k_factor = st.number_input("摩擦因子 / Friction Factor (k)", 
                                    min_value=0.00001, value=0.0001, 
                                    format="%.6f", step=0.00001)
    else:
        static_head = 2.0
        k_factor = 0.0001
    
    # Axis range settings
    st.subheader("坐標軸範圍 / Axis Range")
    
    col6, col7, col8, col9 = st.columns(4)
    
    with col6:
        min_flow = st.number_input("最小流量 / Min Flow", 
                               min_value=0.0, value=0.0, step=10.0)
    
    with col7:
        max_flow = st.number_input("最大流量 / Max Flow (0為自動 / 0 for auto)", 
                               min_value=0.0, value=0.0, step=50.0)
        max_flow = None if max_flow == 0 else max_flow
    
    with col8:
        min_head = st.number_input("最小揚程 / Min Head", 
                               min_value=0.0, value=0.0, step=1.0)
    
    with col9:
        max_head = st.number_input("最大揚程 / Max Head (0為自動 / 0 for auto)", 
                               min_value=0.0, value=0.0, step=5.0)
        max_head = None if max_head == 0 else max_head
    
    st.markdown("---")
    
    # Data input section
    st.subheader("泵數據輸入 / Pump Data Input")
    
    input_method = st.radio("選擇輸入方式 / Select Input Method", 
                         ["上傳 CSV / Upload CSV", "手動輸入 / Manual Input"])
    
    flow_unit = st.selectbox("流量單位 / Flow Unit", ["LPM", "GPM", "m³/h"], index=0)
    
    if input_method == "上傳 CSV / Upload CSV":
        # CSV upload option
        uploaded_file = st.file_uploader("選擇 CSV 文件 / Choose CSV file", type="csv")
        
        # Sample template
        sample_data = pd.DataFrame({
            'Head (m)': [0, 5, 10, 15, 20, 28, 0, 5, 10, 15, 20, 23, 0, 5, 10, 15, 20, 28],
            'Flow (LPM)': [230, 220, 170, 100, 70, 0, 235, 225, 200, 150, 70, 0, 235, 232, 220, 200, 180, 0],
            'Model': ['GD-15', 'GD-15', 'GD-15', 'GD-15', 'GD-15', 'GD-15', 
                      'GD-20', 'GD-20', 'GD-20', 'GD-20', 'GD-20', 'GD-20',
                      'GD-30', 'GD-30', 'GD-30', 'GD-30', 'GD-30', 'GD-30']
        })
        
        st.markdown("### 下載 CSV 模板 / Download CSV Template")
        csv = sample_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="pump_data_template.csv">下載模板 / Download Template</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("CSV 文件已上傳 / CSV file uploaded successfully")
                
                # Display the uploaded data
                st.dataframe(df)
                
                # Check if required columns exist
                if not all(col in df.columns for col in ['Head (m)', 'Flow (LPM)', 'Model']):
                    if 'Head' in df.columns[0] and any('Flow' in col for col in df.columns):
                        # Try to auto-detect and convert format
                        st.info("檢測到傳統格式，正在轉換... / Detected traditional format, converting...")
                        df = convert_traditional_format(df)
                        st.dataframe(df)
                    else:
                        st.error("CSV 格式不正確。需要 'Head (m)', 'Flow (LPM)', 'Model' 列 / CSV format incorrect. Requires 'Head (m)', 'Flow (LPM)', 'Model' columns")
                        st.stop()
            except Exception as e:
                st.error(f"讀取 CSV 時出錯 / Error reading CSV: {e}")
                st.stop()
        else:
            # Use sample data for demonstration
            if st.checkbox("使用樣本數據 / Use sample data for demonstration"):
                df = sample_data
                st.dataframe(df)
            else:
                st.info("請上傳 CSV 文件或使用樣本數據 / Please upload a CSV file or use sample data")
                st.stop()
    else:
        # Manual input
        st.markdown("### 手動輸入泵數據 / Manually Input Pump Data")
        
        # If session state for manual data doesn't exist, initialize it
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = []
            
            # Add some initial data for each model
            for model in ["GD-15", "GD-20", "GD-30"]:
                if model == "GD-15":
                    points = [
                        {"head": 0, "flow": 230, "model": model},
                        {"head": 5, "flow": 220, "model": model},
                        {"head": 10, "flow": 170, "model": model},
                        {"head": 15, "flow": 100, "model": model},
                        {"head": 20, "flow": 70, "model": model},
                        {"head": 28, "flow": 0, "model": model},
                    ]
                elif model == "GD-20":
                    points = [
                        {"head": 0, "flow": 235, "model": model},
                        {"head": 5, "flow": 225, "model": model},
                        {"head": 10, "flow": 200, "model": model},
                        {"head": 15, "flow": 150, "model": model},
                        {"head": 20, "flow": 70, "model": model},
                        {"head": 23, "flow": 0, "model": model},
                    ]
                else:  # GD-30
                    points = [
                        {"head": 0, "flow": 235, "model": model},
                        {"head": 5, "flow": 232, "model": model},
                        {"head": 10, "flow": 220, "model": model},
                        {"head": 15, "flow": 200, "model": model},
                        {"head": 20, "flow": 180, "model": model},
                        {"head": 28, "flow": 0, "model": model},
                    ]
                st.session_state.manual_data.extend(points)
        
        # Get existing models from manual data
        existing_models = sorted(list(set([p["model"] for p in st.session_state.manual_data])))
        
        # Allow selecting model to edit or add new model
        model_options = existing_models + ["添加新模型 / Add New Model"]
        selected_model = st.selectbox("選擇模型編輯 / Select Model to Edit", model_options)
        
        # If adding new model
        if selected_model == "添加新模型 / Add New Model":
            new_model = st.text_input("輸入新模型名稱 / Enter New Model Name")
            if new_model and new_model not in existing_models:
                selected_model = new_model
        
        # If a valid model is selected
        if selected_model and selected_model != "添加新模型 / Add New Model":
            # Get existing data for the selected model
            model_data = [p for p in st.session_state.manual_data if p["model"] == selected_model]
            
            # Convert to DataFrame for editing
            if model_data:
                model_df = pd.DataFrame(model_data)
            else:
                model_df = pd.DataFrame(columns=["head", "flow", "model"])
            
            # Allow editing the data
            st.markdown(f"### 編輯 {selected_model} 數據 / Edit {selected_model} Data")
            
            # Allow adding new data point
            with st.form(f"add_point_form_{selected_model}"):
                col1, col2 = st.columns(2)
                with col1:
                    new_head = st.number_input("揚程 / Head (m)", min_value=0.0, step=1.0)
                with col2:
                    new_flow = st.number_input(f"流量 / Flow ({flow_unit})", min_value=0.0, step=10.0)
                
                add_point = st.form_submit_button("添加數據點 / Add Data Point")
                
                if add_point:
                    # Add new point to session state
                    st.session_state.manual_data.append({
                        "head": new_head,
                        "flow": new_flow,
                        "model": selected_model
                    })
                    st.experimental_rerun()
            
            # Show and edit existing data
            edited_df = st.data_editor(
                model_df,
                key=f"model_editor_{selected_model}",
                column_config={
                    "head": st.column_config.NumberColumn("揚程 / Head (m)", min_value=0.0),
                    "flow": st.column_config.NumberColumn(f"流量 / Flow ({flow_unit})", min_value=0.0),
                    "model": st.column_config.TextColumn("模型 / Model", disabled=True)
                },
                use_container_width=True,
                num_rows="dynamic"
            )
            
            # Update session state if data was changed
            if not edited_df.equals(model_df):
                # Remove old data for this model
                st.session_state.manual_data = [p for p in st.session_state.manual_data if p["model"] != selected_model]
                
                # Add updated data
                for _, row in edited_df.iterrows():
                    st.session_state.manual_data.append({
                        "head": row["head"],
                        "flow": row["flow"],
                        "model": selected_model
                    })
        
        # Show all data
        st.markdown("### 所有泵數據 / All Pump Data")
        all_data_df = pd.DataFrame(st.session_state.manual_data)
        if not all_data_df.empty:
            all_data_df = all_data_df.sort_values(["model", "head"])
            st.dataframe(all_data_df)
            
            # Convert to format needed for plotting
            df = pd.DataFrame({
                'Head (m)': all_data_df["head"],
                'Flow (LPM)': all_data_df["flow"],
                'Model': all_data_df["model"]
            })
        else:
            st.warning("無數據，請添加數據點 / No data, please add data points")
            st.stop()
    
    # Plot pump curve
    if st.button("生成泵曲線 / Generate Pump Curve"):
        try:
            fig = generate_pump_curve(df, chart_style, show_system, 
                                    static_head, k_factor, 
                                    min_flow, max_flow, min_head, max_head, 
                                    show_grid, flow_unit)
            st.pyplot(fig)
            
            # Add download button
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="下載圖表 / Download Plot (PNG)",
                data=buf,
                file_name="pump_curve_plot.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"生成圖表時出錯 / Error generating chart: {e}")

def convert_traditional_format(df):
    """Convert traditional format to required format"""
    # This is a placeholder - you might need to implement this
    # based on your specific traditional format
    new_df = pd.DataFrame(columns=['Head (m)', 'Flow (LPM)', 'Model'])
    return new_df

def generate_pump_curve(df, chart_style, show_system_curve, static_head, k_factor,
                     min_flow, max_flow, min_head, max_head, show_grid, flow_unit):
    """Generate pump curves for irregular data points"""
    # Create figure
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
    else:
        plt.style.use('classic')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2, right=0.9)
    
    # Get unique models
    models = df['Model'].unique()
    
    # Get colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot each model
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].sort_values('Head (m)')
        
        color = colors[i % len(colors)]
        ax.plot(model_data['Flow (LPM)'], model_data['Head (m)'], 
                linestyle='-', linewidth=2.5, label=model,
                color=color, marker='o', markersize=4)
        
        # Add label
        if len(model_data) > 0:
            max_head_idx = model_data['Head (m)'].idxmax()
            x_pos = model_data.loc[max_head_idx, 'Flow (LPM)']
            y_pos = model_data.loc[max_head_idx, 'Head (m)']
            ax.annotate(f"{model}", xy=(x_pos, y_pos), xytext=(5, 0),
                       textcoords='offset points', color=color, 
                       fontweight='bold', va='center')
    
    # Add system curve if requested
    if show_system_curve:
        # Calculate system curve
        max_flow_val = df['Flow (LPM)'].max() * 1.2 if max_flow is None else max_flow
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve
        ax.plot(system_flows, system_heads, 'r-', linewidth=2, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)')
        
        # Find intersection points
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].sort_values('Flow (LPM)')
            
            if len(model_data) >= 2:
                color = colors[i % len(colors)]
                flows = model_data['Flow (LPM)'].values
                heads = model_data['Head (m)'].values
                
                try:
                    # Create interpolation function
                    model_func = interp1d(flows, heads, bounds_error=False, fill_value="extrapolate")
                    
                    # Find intersection manually (simple approach)
                    for j in range(len(system_flows) - 1):
                        pump_head1 = model_func(system_flows[j])
                        pump_head2 = model_func(system_flows[j+1])
                        sys_head1 = system_heads[j]
                        sys_head2 = system_heads[j+1]
                        
                        # Check if there's an intersection
                        if ((pump_head1 - sys_head1) * (pump_head2 - sys_head2) <= 0 and 
                            not np.isnan(pump_head1) and not np.isnan(pump_head2)):
                            # Approximate intersection point
                            t = abs(pump_head1 - sys_head1) / abs((pump_head2 - sys_head2) - (pump_head1 - sys_head1))
                            op_flow = system_flows[j] + t * (system_flows[j+1] - system_flows[j])
                            op_head = sys_head1 + t * (sys_head2 - sys_head1)
                            
                            # Plot operating point
                            ax.plot(op_flow, op_head, 'o', markersize=8, color=color)
                            
                            # Add label
                            ax.annotate(f"{model}: ({op_flow:.1f}, {op_head:.1f})",
                                       xy=(op_flow, op_head), xytext=(10, 10),
                                       textcoords='offset points', color=color,
                                       fontweight='bold')
                            break
                except:
                    # Skip if interpolation fails
                    pass
    
    # Set axis labels
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Head (m)', fontsize=12, fontweight='bold')
    
    # Set axis limits
    if max_flow is None:
        ax.set_xlim(left=float(min_flow), right=df['Flow (LPM)'].max() * 1.1)
    else:
        ax.set_xlim(left=float(min_flow), right=float(max_flow))
    
    if max_head is None:
        ax.set_ylim(bottom=float(min_head), top=df['Head (m)'].max() * 1.1)
    else:
        ax.set_ylim(bottom=float(min_head), top=float(max_head))
    
    # Add grid
    if show_grid:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5)
        if chart_style == "Modern":
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    
    # Add secondary axes for unit conversion
    if flow_unit == "LPM":
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis
        ax_gpm = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.264172, lambda x: x/0.264172))
        ax_gpm.set_xlabel(f'Flow (GPM)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Add secondary y-axis for ft
    ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
    ax_ft.set_ylabel('Head (ft)', fontsize=12, fontweight='bold', labelpad=15)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
    plt.title('Pump Performance Curves', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    main()
