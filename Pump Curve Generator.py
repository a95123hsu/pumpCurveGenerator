import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy.interpolate import interp1d

def main():
    # Create a function to auto-update chart when configuration changes
    def update_chart_on_config_change():
        if st.session_state.current_df is not None and st.session_state.chart_generated:
            st.rerun()

    st.set_page_config(page_title="泵曲線生成器", layout="wide")
    
    # Initialize session state for language
    if 'language' not in st.session_state:
        st.session_state.language = "Chinese"  # Default language
    
    # Create a language switcher in the sidebar
    language = st.sidebar.selectbox(
        "選擇語言 / Select Language",
        ["中文", "English"],
        index=0 if st.session_state.language == "Chinese" else 1
    )
    
    # Update language in session state
    st.session_state.language = "Chinese" if language == "中文" else "English"
    
    # Set language-specific UI text
    if st.session_state.language == "Chinese":
        title_text = "泵曲線生成器工具"
        intro_text = """
        此工具允許您生成類似於製造商規格的泵性能曲線。
        首先，配置您的圖表設置，然後上傳或輸入您的泵數據以生成曲線。
        支援不同模型具有不同測量點的數據。
        """
        tab1_text = "創建泵曲線"
        tab2_text = "關於泵曲線"
        config_text = "圖表配置"
        freq_display_text = "頻率顯示"
        chart_style_text = "圖表樣式"
        show_system_text = "顯示系統曲線"
        show_grid_text = "顯示網格"
        static_head_text = "靜壓頭 (m)"
        k_factor_text = "摩擦因子 (k)"
        axis_range_text = "坐標軸範圍設置"
        min_flow_text = "最小流量"
        max_flow_text = "最大流量 (0 為自動)"
        min_head_text = "最小揚程"
        max_head_text = "最大揚程 (0 為自動)"
        input_method_text = "選擇輸入方式"
        upload_csv_text = "上傳 CSV"
        manual_input_text = "手動輸入"
        excel_copy_text = "您可以從 Excel 複製數據並直接粘貼到這些表格中。在 Excel 中選擇單元格，複製 (Ctrl+C)，點擊下面表格中的起始單元格，然後粘貼 (Ctrl+V)。"
        flow_unit_text = "流量單位"
        head_unit_text = "揚程單位"
        num_models_text = "泵型號數量"
        model_name_text = "型號 {} 名稱"
        num_points_text = "每個模型的數據點數量"
        use_template_text = "使用模板數據"
        edit_pump_data_text = "### 編輯下方泵數據"
        edit_data_info_text = "編輯 {} 泵數據。每個模型可以有不同的揚程測量點。"
        generate_curve_text = "生成泵曲線"
        refresh_form_text = "刷新表單"
        csv_template_text = "### 下載 CSV 模板"
        download_template_text = "下載 CSV 模板"
        detected_head_text = "檢測到以揚程為首的數據格式。正在轉換為圖表所需的格式。"
        no_flow_columns_text = "在 CSV 中未檢測到流量列。請確保列名包含'Flow'。"
        download_label = "下載圖表 (PNG)"
        chart_generate_error = "生成圖表時出錯: {}"
        click_generate_chart = "點擊生成泵曲線按鈕創建圖表。"
        csv_error = "讀取 CSV 文件時出錯: {}"
        form_refreshed = "表單已刷新，數據已保存"
        irregular_data_mode = "不規則數據模式"
        irregular_data_info = "啟用此選項以處理不同模型具有不同揚程測量點的情況"
        data_model_format = "數據格式: 揚程, 流量, 模型名稱"
        edit_models_text = "編輯泵模型數據"
        add_point_text = "添加數據點"
        remove_point_text = "刪除數據點"
        head_text = "揚程 (m)"
        flow_text = "流量"
        model_text = "模型"
    else:
        title_text = "Pump Curve Generator Tool"
        intro_text = """
        This tool allows you to generate pump performance curves similar to manufacturer specifications.
        First, configure your chart settings, then upload or enter your pump data to generate the curve.
        Supports data where different models have different measurement points.
        """
        tab1_text = "Create Pump Curves"
        tab2_text = "About Pump Curves"
        config_text = "Chart Configuration"
        freq_display_text = "Frequency Display"
        chart_style_text = "Chart Style"
        show_system_text = "Show System Curve"
        show_grid_text = "Show Grid"
        static_head_text = "Static Head (m)"
        k_factor_text = "Friction Factor (k)"
        axis_range_text = "Axis Range Settings"
        min_flow_text = "Min Flow"
        max_flow_text = "Max Flow (0 for auto)"
        min_head_text = "Min Head"
        max_head_text = "Max Head (0 for auto)"
        input_method_text = "Select Input Method"
        upload_csv_text = "Upload CSV"
        manual_input_text = "Manual Input"
        excel_copy_text = "You can copy data from Excel and paste directly into these tables. Select cells in Excel, copy (Ctrl+C), click on the starting cell in the table below, and paste (Ctrl+V)."
        flow_unit_text = "Flow Rate Unit"
        head_unit_text = "Head Unit"
        num_models_text = "Number of Pump Models"
        model_name_text = "Model {} Name"
        num_points_text = "Data Points per Model"
        use_template_text = "Use Template Data"
        edit_pump_data_text = "### Edit Pump Data Below"
        edit_data_info_text = "Edit {} pump data below. Each model can have different head measurement points."
        generate_curve_text = "Generate Pump Curve"
        refresh_form_text = "Refresh Form"
        csv_template_text = "### Download Sample CSV Template"
        download_template_text = "Download CSV Template"
        detected_head_text = "Detected head-first format data. Converting to the format needed for the chart."
        no_flow_columns_text = "No flow columns detected in the CSV. Please ensure column names contain 'Flow'."
        download_label = "Download Plot (PNG)"
        chart_generate_error = "Error generating chart: {}"
        click_generate_chart = "Click Generate Pump Curve to create the chart."
        csv_error = "Error reading CSV file: {}"
        form_refreshed = "Form refreshed, data preserved"
        irregular_data_mode = "Irregular Data Mode"
        irregular_data_info = "Enable this option to handle cases where different models have different head measurement points"
        data_model_format = "Data format: Head, Flow, Model Name"
        edit_models_text = "Edit Pump Model Data"
        add_point_text = "Add Data Point"
        remove_point_text = "Remove Data Point"
        head_text = "Head (m)"
        flow_text = "Flow"
        model_text = "Model"
    
    # Initialize session state for tracking changes and storing persistent data
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    
    if 'chart_params' not in st.session_state:
        st.session_state.chart_params = {
            'frequency_option': "Both",  # Default to showing both frequencies
            'chart_style': "Modern",
            'show_system_curve': False,
            'static_head': 2.0,
            'k_factor': 0.0001,
            'max_flow': None,  # Auto-scale by default
            'max_head': None,  # Auto-scale by default
            'min_flow': 0.0,   # Start at 0 by default - use float
            'min_head': 0.0,   # Start at 0 by default - use float
            'show_grid': True, # Show grid by default
            'irregular_data': False, # Default to regular data mode
        }
    
    # Initialize input reset key if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    # Initialize data storage
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    
    # Initialize chart generation flag
    if 'chart_generated' not in st.session_state:
        st.session_state.chart_generated = False
    
    # Initialize number of data points
    if 'num_data_points' not in st.session_state:
        st.session_state.num_data_points = 11
        
    # Initialize manual input data storage
    if 'manual_input_data' not in st.session_state:
        st.session_state.manual_input_data = {
            'flow_unit': "LPM",
            'head_unit': "m",
            'model_names': ["Model-A", "Model-B"],
            'use_template': True,
            'edited_data': {}
        }
    
    # Initialize irregular data mode storage
    if 'irregular_data_points' not in st.session_state:
        st.session_state.irregular_data_points = []
        # Initialize with sample data
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
            st.session_state.irregular_data_points.extend(points)
    
    # Initialize form refresh notification
    if 'show_refresh_notification' not in st.session_state:
        st.session_state.show_refresh_notification = False
    
    st.title(title_text)
    st.markdown(intro_text)
    
    tab1, tab2 = st.tabs([tab1_text, tab2_text])
    
    with tab1:
        # Configuration options
        st.subheader(config_text)
        
        # Create columns for chart options
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
        
        # Frequency options translations
        freq_options = {
            "Chinese": ["僅 50Hz", "僅 60Hz", "兩者都顯示"],
            "English": ["50Hz Only", "60Hz Only", "Both"]
        }
        
        # Chart style options translations
        style_options = {
            "Chinese": ["現代", "經典"],
            "English": ["Modern", "Classic"]
        }
        
        # When any option changes, immediately update session state and trigger chart refresh
        with col_a:
            frequency_option_display = freq_options[st.session_state.language]
            frequency_option_values = ["50Hz Only", "60Hz Only", "Both"]
            
            # Find index of current value in the values list
            current_index = frequency_option_values.index(st.session_state.chart_params.get('frequency_option', "Both"))
            
            frequency_option = st.selectbox(
                freq_display_text, 
                frequency_option_display,
                index=current_index,
                key="frequency_option_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'frequency_option': frequency_option_values[frequency_option_display.index(st.session_state.frequency_option_select)]}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_b:
            style_option_display = style_options[st.session_state.language]
            style_option_values = ["Modern", "Classic"]
            
            # Find index of current value in the values list
            current_style_index = style_option_values.index(st.session_state.chart_params['chart_style'])
            
            chart_style = st.selectbox(
                chart_style_text, 
                style_option_display, 
                index=current_style_index,
                key="chart_style_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'chart_style': style_option_values[style_option_display.index(st.session_state.chart_style_select)]}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_c:
            show_system = st.checkbox(
                show_system_text, 
                value=st.session_state.chart_params['show_system_curve'],
                key="system_curve_checkbox",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'show_system_curve': st.session_state.system_curve_checkbox}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_d:
            show_grid = st.checkbox(
                show_grid_text, 
                value=st.session_state.chart_params['show_grid'],
                key="show_grid_checkbox",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'show_grid': st.session_state.show_grid_checkbox}
                    ),
                    update_chart_on_config_change()
                )
            )
            
            # Add irregular data mode checkbox
            irregular_data = st.checkbox(
                irregular_data_mode,
                value=st.session_state.chart_params.get('irregular_data', False),
                key="irregular_data_checkbox",
                help=irregular_data_info,
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'irregular_data': st.session_state.irregular_data_checkbox}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        # System curve parameters (only shown if show_system_curve is True)
        if st.session_state.chart_params['show_system_curve']:
            col_e, col_f = st.columns(2)
            with col_e:
                static_head = st.number_input(
                    static_head_text, 
                    min_value=0.0, 
                    value=st.session_state.chart_params['static_head'], 
                    step=0.5,
                    key="static_head_input",
                    on_change=lambda: (
                        setattr(
                            st.session_state, 'chart_params', 
                            {**st.session_state.chart_params, 'static_head': st.session_state.static_head_input}
                        ),
                        update_chart_on_config_change()
                    )
                )
            
            with col_f:
                k_factor = st.number_input(
                    k_factor_text, 
                    min_value=0.00001, 
                    value=st.session_state.chart_params['k_factor'], 
                    format="%.6f", 
                    step=0.00001,
                    key="k_factor_input",
                    on_change=lambda: (
                        setattr(
                            st.session_state, 'chart_params', 
                            {**st.session_state.chart_params, 'k_factor': st.session_state.k_factor_input}
                        ),
                        update_chart_on_config_change()
                    )
                )
        
        # Add axis range controls
        st.subheader(axis_range_text)
        col_g, col_h, col_i, col_j = st.columns(4)
        
        with col_g:
            min_flow = st.number_input(
                min_flow_text, 
                min_value=0.0,
                value=float(st.session_state.chart_params['min_flow'] or 0.0), 
                step=10.0,
                key="min_flow_input",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'min_flow': st.session_state.min_flow_input}
                    ),
                    update_chart_on_config_change()
                )
            )
            
        with col_h:
            # Handle max_flow special case (None value)
            max_flow_value = st.session_state.chart_params['max_flow']
            max_flow_value = float(max_flow_value) if max_flow_value is not None else None
            
            max_flow = st.number_input(
                max_flow_text, 
                min_value=0.0,
                value=float(max_flow_value or 0.0), 
                step=100.0,
                key="max_flow_input",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'max_flow': st.session_state.max_flow_input if st.session_state.max_flow_input > 0 else None}
                    ),
                    update_chart_on_config_change()
                )
            )
            
        with col_i:
            min_head = st.number_input(
                min_head_text, 
                min_value=0.0,
                value=float(st.session_state.chart_params['min_head'] or 0.0), 
                step=1.0,
                key="min_head_input",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'min_head': st.session_state.min_head_input}
                    ),
                    update_chart_on_config_change()
                )
            )
            
        with col_j:
            # Handle max_head special case (None value)
            max_head_value = st.session_state.chart_params['max_head']
            max_head_value = float(max_head_value) if max_head_value is not None else None
            
            max_head = st.number_input(
                max_head_text, 
                min_value=0.0,
                value=float(max_head_value or 0.0), 
                step=1.0,
                key="max_head_input",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'max_head': st.session_state.max_head_input if st.session_state.max_head_input > 0 else None}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        st.markdown("---")
        
        # Show refresh notification if needed
        if st.session_state.show_refresh_notification:
            st.success(form_refreshed)
            st.session_state.show_refresh_notification = False
        
        # Input method selection
        input_method_options = {
            "Chinese": ["上傳 CSV", "手動輸入"],
            "English": ["Upload CSV", "Manual Input"]
        }
        
        input_method_values = ["Upload CSV", "Manual Input"]
        input_method_display = input_method_options[st.session_state.language]
        
        input_method = st.radio(
            input_method_text,
            input_method_display
        )
        
        selected_method = input_method_values[input_method_display.index(input_method)]
        
        if selected_method == "Upload CSV":
            df = handle_csv_upload(st.session_state.language)
        else:
            if st.session_state.chart_params['irregular_data']:
                df = handle_irregular_manual_input(
                    st.session_state.language,
                    st.session_state.manual_input_data.get('flow_unit', "LPM"),
                    head_text,
                    flow_text,
                    model_text,
                    add_point_text,
                    remove_point_text,
                    edit_models_text
                )
            else:
                df = handle_manual_input(
                    st.session_state.chart_params.get('frequency_option', "Both"), 
                    st.session_state.language
                )
        
        # Generate and display the pump curve if data is available
        if df is not None and not df.empty:
            st.session_state.current_df = df  # Store the current dataframe
            
            # For CSV uploads, automatically set chart_generated to True
            if selected_method == "Upload CSV" and not st.session_state.chart_generated:
                st.session_state.chart_generated = True
            
            # Generate curve using parameters from session state
            if st.session_state.chart_generated:
                params = st.session_state.chart_params
                try:
                    # Pass parameters to the plotting function
                    if params.get('irregular_data', False):
                        fig = generate_irregular_pump_curve(
                            df, 
                            chart_style=params['chart_style'], 
                            show_system_curve=params['show_system_curve'], 
                            static_head=params['static_head'], 
                            k_factor=params['k_factor'],
                            min_flow=params['min_flow'],
                            max_flow=params['max_flow'],
                            min_head=params['min_head'],
                            max_head=params['max_head'],
                            show_grid=params['show_grid'],
                            flow_unit=st.session_state.manual_input_data.get('flow_unit', "LPM")
                        )
                    else:
                        fig = generate_pump_curve(
                            df, 
                            frequency_option=params.get('frequency_option', "Both"),
                            chart_style=params['chart_style'], 
                            show_system_curve=params['show_system_curve'], 
                            static_head=params['static_head'], 
                            k_factor=params['k_factor'],
                            min_flow=params['min_flow'],
                            max_flow=params['max_flow'],
                            min_head=params['min_head'],
                            max_head=params['max_head'],
                            show_grid=params['show_grid']
                        )
                    st.pyplot(fig)
                    
                    # Add download button for the plot
                    download_button_for_plot(fig, st.session_state.language)
                except Exception as e:
                    st.error(chart_generate_error.format(e))
            else:
                st.info(click_generate_chart)
    
    with tab2:
        if st.session_state.language == "Chinese":
            st.subheader("了解泵曲線")
            st.markdown("""
            ### 什麼是泵曲線？
            
            泵曲線（或性能曲線）以圖形方式表示以下關係：
            
            - **流量**：泵每單位時間可以輸送的液體體積（以 LPM、m³/h 或 GPM 計量）
            - **揚程**：泵可以提升液體的壓力或高度（以米或英尺計量）
            
            ### 讀取泵曲線
            
            - 每條曲線代表特定的泵型號或葉輪尺寸
            - X軸顯示流量
            - Y軸顯示揚程
            - 隨著流量增加，揚程通常會減少
            - 泵的工作點由泵曲線與系統曲線的交點確定
            
            ### 系統曲線
            
            系統曲線表示管道系統中的阻力：
            
            - 它由靜壓頭（垂直高度）和摩擦損失組成
            - 數學表達式為：H = Hs + k × Q²
              - H = 總揚程
              - Hs = 靜壓頭
              - k = 摩擦係數
              - Q = 流量
            
            ### 頻率影響（50Hz 與 60Hz）
            
            改變電頻影響泵的性能：
            
            - 流量 (Q) 與轉速 (n) 成正比：Q₂ = Q₁ × (n₂/n₁)
            - 揚程 (H) 與轉速的平方成正比：H₂ = H₁ × (n₂/n₁)²
            - 功率 (P) 與轉速的立方成正比：P₂ = P₁ × (n₂/n₁)³
            
            50Hz 到 60Hz 的轉換：
            - 流量增加 20%（60/50 = 1.2）
            - 揚程增加 44%（1.2² = 1.44）
            - 功率增加 73%（1.2³ = 1.728）
            
            ### 選擇合適的泵
            
            選擇泵時，考慮以下因素：
            1. 所需流量
            2. 所需揚程
            3. 系統效率
            4. NPSH（淨正吸入揚程）
            5. 功耗
            
            ### 不規則數據點的處理
            
            在實際應用中，不同型號的泵可能在不同的測量點有數據：
            - 使用內插法生成平滑曲線
            - 計算系統曲線時考慮所有可用數據點
            - 確保正確顯示每個型號的性能範圍
            """)
        else:
            st.subheader("Understanding Pump Curves")
            st.markdown("""
            ### What is a Pump Curve?
            
            A pump curve (or performance curve) graphically represents the relationship between:
            
            - **Flow Rate**: The volume of liquid a pump can move per unit time (measured in LPM, m³/h, or GPM)
            - **Head**: The pressure or height to which a pump can raise liquid (measured in meters or feet)
            
            ### Reading Pump Curves
            
            - Each curve represents a specific pump model or impeller size
            - The x-axis shows flow rate
            - The y-axis shows head
            - As flow increases, head typically decreases
            - The operating point of a pump is determined by where the pump curve intersects with the system curve
            
            ### System Curve
            
            A system curve represents the resistance in your piping system:
            
            - It consists of static head (vertical height) and friction losses
            - Mathematically expressed as: H = Hs + k × Q²
              - H = Total head
              - Hs = Static head
              - k = Friction coefficient
              - Q = Flow rate
            
            ### Frequency Impact (50Hz vs 60Hz)
            
            Changing the electrical frequency affects pump performance:
            
            - Flow (Q) is proportional to speed (n): Q₂ = Q₁ × (n₂/n₁)
            - Head (H) is proportional to speed squared: H₂ = H₁ × (n₂/n₁)²
            - Power (P) is proportional to speed cubed: P₂ = P₁ × (n₂/n₁)³
            
            For 50Hz to 60Hz conversion:
            - Flow increases by 20% (60/50 = 1.2)
            - Head increases by 44% (1.2² = 1.44)
            - Power increases by 73% (1.2³ = 1.728)
            
            ### Selecting the Right Pump
            
            When selecting a pump, consider:
            1. Required flow rate
            2. Required head
            3. System efficiency
            4. NPSH (Net Positive Suction Head)
            5. Power consumption
            
            ### Handling Irregular Data Points
            
            In practical applications, different pump models may have data at different measurement points:
            - Interpolation is used to generate smooth curves
            - All available data points are considered when calculating system curves
            - Ensures accurate representation of each model's performance range
            """)

def handle_csv_upload(language):
    if language == "Chinese":
        st.subheader("上傳 CSV 文件")
        uploaded_file = st.file_uploader("選擇 CSV 文件", type="csv")
        csv_template_text = "### 下載 CSV 模板"
        download_template_text = "下載 CSV 模板"
        detected_head_text = "檢測到以揚程為首的數據格式。正在轉換為圖表所需的格式。"
        no_flow_columns_text = "在 CSV 中未檢測到流量列。請確保列名包含'Flow'。"
    else:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        csv_template_text = "### Download Sample CSV Template"
        download_template_text = "Download CSV Template"
        detected_head_text = "Detected head-first format data. Converting to the format needed for the chart."
        no_flow_columns_text = "No flow columns detected in the CSV. Please ensure column names contain 'Flow'."
    
    # Sample CSV template for download - Updated to include both 50Hz and 60Hz data
    sample_data = pd.DataFrame({
        'Head (ft)': [4.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 49.8],
        'Model-A 50Hz Flow (GPM)': [90.09, 89.96, 84.81, 77.43, 69.85, 61.89, 53.93, 41.81, 29.34, 18.52, 8.45],
        'Model-A 60Hz Flow (GPM)': [108.11, 107.95, 101.77, 92.92, 83.82, 74.27, 64.72, 50.17, 35.21, 22.22, 10.14],
        'Model-B 50Hz Flow (GPM)': [105.50, 104.75, 95.22, 86.34, 78.43, 69.45, 60.76, 48.45, 35.21, 21.93, 11.24],
        'Model-B 60Hz Flow (GPM)': [126.60, 125.70, 114.26, 103.61, 94.12, 83.34, 72.91, 58.14, 42.25, 26.32, 13.49]
    })
    
    # Create irregular data format sample template
    irregular_sample_data = pd.DataFrame({
        'Head (m)': [0, 5, 10, 15, 20, 28, 0, 5, 10, 15, 20, 23, 0, 5, 10, 15, 20, 28],
        'Flow (LPM)': [230, 220, 170, 100, 70, 0, 235, 225, 200, 150, 70, 0, 235, 232, 220, 200, 180, 0],
        'Model': ['GD-15', 'GD-15', 'GD-15', 'GD-15', 'GD-15', 'GD-15', 
                  'GD-20', 'GD-20', 'GD-20', 'GD-20', 'GD-20', 'GD-20',
                  'GD-30', 'GD-30', 'GD-30', 'GD-30', 'GD-30', 'GD-30']
    })
    
    st.markdown(csv_template_text)
    
    # Create standard template with both frequencies
    csv_standard = sample_data.to_csv(index=False)
    b64_standard = base64.b64encode(csv_standard.encode()).decode()
    href_standard = f'<a href="data:file/csv;base64,{b64_standard}" download="pump_curve_template.csv">{download_template_text} (Standard)</a>'
    
    # Create irregular template
    csv_irregular = irregular_sample_data.to_csv(index=False)
    b64_irregular = base64.b64encode(csv_irregular.encode()).decode()
    href_irregular = f'<a href="data:file/csv;base64,{b64_irregular}" download="pump_curve_irregular_template.csv">{download_template_text} (Irregular Data)</a>'
    
    st.markdown(href_standard, unsafe_allow_html=True)
    st.markdown(href_irregular, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if this is an irregular data format (has Model column)
            if 'Model' in df.columns:
                st.session_state.chart_params['irregular_data'] = True
                
                # Create list of data points for session state
                irregular_data = []
                for _, row in df.iterrows():
                    point = {
                        "head": row['Head (m)'],
                        "flow": row['Flow (LPM)'],
                        "model": row['Model']
                    }
                    irregular_data.append(point)
                
                st.session_state.irregular_data_points = irregular_data
                return df
            
            # Transform the data into format needed for plotting (Head vs Flow)
            if 'Head' in df.columns[0]:
                st.info(detected_head_text)
                
                # Extract head values and units
                head_col = df.columns[0]
                head_unit = head_col.split('(')[-1].split(')')[0]
                head_values = df[head_col].values
                
                # Create new DataFrame for transformed data
                transformed_df = pd.DataFrame()
                
                # Get all flow columns
                flow_cols = [col for col in df.columns if 'Flow' in col]
                
                if not flow_cols:
                    st.warning(no_flow_columns_text)
                    return None
                
                # Process columns by model and frequency
                model_freq_dict = {}
                
                # Identify models and their frequencies from column names
                for col in flow_cols:
                    parts = col.split(' ')
                    model_name = parts[0]  # Extract model name
                    
                    # Check if column has frequency info
                    frequency = None
                    for part in parts:
                        if '50Hz' in part:
                            frequency = '50Hz'
                            break
                        elif '60Hz' in part:
                            frequency = '60Hz'
                            break
                    
                    # If no frequency specified, assume 50Hz
                    if frequency is None:
                        frequency = '50Hz'
                    
                    # Group by model and frequency
                    if model_name not in model_freq_dict:
                        model_freq_dict[model_name] = {}
                    
                    model_freq_dict[model_name][frequency] = df[col].values
                
                # Create new structure for transformed data
                for model_name, freq_data in model_freq_dict.items():
                    # Get flow unit from first flow column
                    flow_unit = flow_cols[0].split('(')[-1].split(')')[0]
                    
                    # Add 50Hz data if available
                    if '50Hz' in freq_data:
                        # If first model, initialize flow column
                        if len(transformed_df) == 0:
                            transformed_df[f'Flow ({flow_unit})'] = freq_data['50Hz']
                        
                        # Add head column for this model at 50Hz
                        transformed_df[f'{model_name} 50Hz Head ({head_unit})'] = head_values
                    
                    # Add 60Hz data if available
                    if '60Hz' in freq_data:
                        # If no 50Hz data and first model, initialize flow column
                        if '50Hz' not in freq_data and len(transformed_df) == 0:
                            transformed_df[f'Flow ({flow_unit})'] = freq_data['60Hz']
                        
                        # Add head column for this model at 60Hz
                        transformed_df[f'{model_name} 60Hz Head ({head_unit})'] = head_values
                
                df = transformed_df
            
            # Basic validation
            if not any('Flow' in col for col in df.columns):
                st.warning(no_flow_columns_text)
                return None
            
            # Automatically set chart_generated to True when a file is uploaded
            st.session_state.chart_generated = True
                
            return df
        except Exception as e:
            if language == "Chinese":
                st.error(f"讀取 CSV 文件時出錯: {e}")
            else:
                st.error(f"Error reading CSV file: {e}")
            return None
    
    return None

def handle_manual_input(frequency_option="Both", language="English"):
    if language == "Chinese":
        st.subheader("手動數據輸入")
        excel_copy_text = "您可以從 Excel 複製數據並直接粘貼到這些表格中。在 Excel 中選擇單元格，複製 (Ctrl+C)，點擊下面表格中的起始單元格，然後粘貼 (Ctrl+V)。"
        flow_unit_text = "流量單位"
        head_unit_text = "揚程單位"
        num_models_text = "泵型號數量"
        model_name_text = "型號 {} 名稱"
        num_points_text = "數據點數量"
        use_template_text = "使用模板數據"
        edit_pump_data_text = "### 編輯下方泵數據"
        edit_data_info_text = "編輯 {} 泵數據。揚程值在所有型號中是共用的。"
        generate_curve_text = "生成泵曲線"
        refresh_form_text = "刷新表單"
    else:
        st.subheader("Manual Data Input")
        excel_copy_text = "You can copy data from Excel and paste directly into these tables. Select cells in Excel, copy (Ctrl+C), click on the starting cell in the table below, and paste (Ctrl+V)."
        flow_unit_text = "Flow Rate Unit"
        head_unit_text = "Head Unit"
        num_models_text = "Number of Pump Models"
        model_name_text = "Model {} Name"
        num_points_text = "Number of Data Points"
        use_template_text = "Use Template Data"
        edit_pump_data_text = "### Edit Pump Data Below"
        edit_data_info_text = "Edit {} pump data below. Head values are common across models."
        generate_curve_text = "Generate Pump Curve"
        refresh_form_text = "Refresh Form"
    
    # Add instructions for copying from Excel
    st.info(excel_copy_text)
    
    # Initialize session state for manually input data if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    # Create a form for manual input
    with st.form(f"manual_input_form_{st.session_state.input_reset_key}"):
        # Units selection
        col1, col2 = st.columns(2)
        with col1:
            # Use session state values if available
            default_flow_unit = st.session_state.manual_input_data.get('flow_unit', "LPM")
            flow_unit = st.selectbox(flow_unit_text, ["GPM", "LPM", "m³/h"], 
                                   index=["GPM", "LPM", "m³/h"].index(default_flow_unit),
                                   key=f"flow_unit_{st.session_state.input_reset_key}")
        with col2:
            default_head_unit = st.session_state.manual_input_data.get('head_unit', "m")
            head_unit = st.selectbox(head_unit_text, ["ft", "m"], 
                                   index=["ft", "m"].index(default_head_unit),
                                   key=f"head_unit_{st.session_state.input_reset_key}")
        
        # Number of pump models (use previous value from session state if available)
        default_num_models = len(st.session_state.manual_input_data.get('model_names', ["Model-A", "Model-B"]))
        num_models = st.number_input(num_models_text, min_value=1, max_value=5, value=default_num_models,
                                   key=f"num_models_{st.session_state.input_reset_key}")
        
        # Model names
        model_names = []
        stored_model_names = st.session_state.manual_input_data.get('model_names', ["Model-A", "Model-B"])
        
        cols = st.columns(min(num_models, 5))
        for i, col in enumerate(cols):
            # Use stored model name if available, otherwise use default
            default_name = stored_model_names[i] if i < len(stored_model_names) else f"Model-{chr(65+i)}"
            model_name = col.text_input(model_name_text.format(i+1), value=default_name,
                                      key=f"model_name_{i}_{st.session_state.input_reset_key}")
            model_names.append(model_name)
        
        # Number of data points - New control
        num_points = st.number_input(
            num_points_text, 
            min_value=3, 
            max_value=30, 
            value=st.session_state.num_data_points,
            key=f"num_points_{st.session_state.input_reset_key}"
        )
        
        # Store the number of data points in session state for persistence
        st.session_state.num_data_points = num_points
        
        # Frequencies to display (based on frequency_option)
        frequencies_to_show = []
        if frequency_option == "50Hz Only" or frequency_option == "Both":
            frequencies_to_show.append("50Hz")
        if frequency_option == "60Hz Only" or frequency_option == "Both":
            frequencies_to_show.append("60Hz")
            
        # Option to use template data
        default_use_template = st.session_state.manual_input_data.get('use_template', True)
        use_template = st.checkbox(use_template_text, value=default_use_template,
                                 key=f"use_template_{st.session_state.input_reset_key}")
        
        # Generate head and flow data
        template_head, template_flow_50hz, template_flow_60hz = generate_template_data(
            num_points, model_names, use_template
        )
        
        # Create editable dataframes for each frequency
        st.markdown(edit_pump_data_text)
        
        # Create tabs for each frequency
        frequency_tabs = st.tabs([f"{freq} Data" for freq in frequencies_to_show])
        
        # Dictionary to store edited data for each frequency
        edited_data = {}
        
        # Create data editor for each frequency
        for i, freq in enumerate(frequencies_to_show):
            with frequency_tabs[i]:
                st.info(edit_data_info_text.format(freq))
                
                # Try to get previously stored data for this frequency
                stored_data = st.session_state.manual_input_data.get('edited_data', {}).get(freq, None)
                
                # Create the base dataframe for this frequency
                if stored_data is not None:
                    # Use stored data if available
                    df_freq = stored_data
                else:
                    # Create new dataframe from templates
                    df_freq = pd.DataFrame({
                        f'Head ({head_unit})': template_head
                    })
                    
                    # Add flow columns for each model
                    for model in model_names:
                        if freq == "50Hz":
                            df_freq[f'{model} Flow ({flow_unit})'] = template_flow_50hz[model]
                        else:  # 60Hz
                            df_freq[f'{model} Flow ({flow_unit})'] = template_flow_60hz[model]
                
                # Create a data editor for this frequency
                edited_df = st.data_editor(
                    df_freq, 
                    use_container_width=True,
                    num_rows="fixed",
                    height=min(500, 70 + 40*num_points),
                    key=f"data_editor_{freq}_{st.session_state.input_reset_key}",
                    hide_index=False  # Show row indices for easier selection
                )
                
                # Store the edited data
                edited_data[freq] = edited_df
        
        # Submit button and Refresh Form button
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(generate_curve_text)
        with col2:
            # Keep the Refresh Form button
            refresh_data = st.form_submit_button(refresh_form_text)
        
        if submitted:
            # Save the current form state to session state
            st.session_state.manual_input_data = {
                'flow_unit': flow_unit,
                'head_unit': head_unit,
                'model_names': model_names,
                'use_template': use_template,
                'edited_data': edited_data
            }
            
            # Transform the edited data back into the format needed for plotting
            transformed_df = pd.DataFrame()
            
            # Process data for each frequency
            for freq in frequencies_to_show:
                if freq in edited_data:
                    df_freq = edited_data[freq]
                    
                    # Extract head values
                    head_col = df_freq.columns[0]
                    head_values = df_freq[head_col].values
                    
                    # Get flow columns
                    flow_cols = [col for col in df_freq.columns if 'Flow' in col]
                    
                    # Process each model's flow column
                    for flow_col in flow_cols:
                        model_name = flow_col.split(' ')[0]  # Extract model name
                        flow_values = df_freq[flow_col].values
                        
                        # Add to transformed dataframe
                        if len(transformed_df) == 0:
                            # First frequency and model, initialize with flow column
                            transformed_df[f'Flow ({flow_unit})'] = flow_values
                        
                        # Add head column for this model and frequency
                        transformed_df[f'{model_name} {freq} Head ({head_unit})'] = head_values
            
            # Automatically generate the chart when data is submitted
            st.session_state.chart_generated = True
            return transformed_df
        elif refresh_data:
            # Save current data first
            st.session_state.manual_input_data = {
                'flow_unit': flow_unit,
                'head_unit': head_unit,
                'model_names': model_names,
                'use_template': use_template,
                'edited_data': edited_data
            }
            
            # Capture current data before refreshing
            transformed_df = pd.DataFrame()
            
            # Process data for each frequency
            for freq in frequencies_to_show:
                if freq in edited_data:
                    df_freq = edited_data[freq]
                    
                    # Extract head values
                    head_col = df_freq.columns[0]
                    head_values = df_freq[head_col].values
                    
                    # Get flow columns
                    flow_cols = [col for col in df_freq.columns if 'Flow' in col]
                    
                    # Process each model's flow column
                    for flow_col in flow_cols:
                        model_name = flow_col.split(' ')[0]  # Extract model name
                        flow_values = df_freq[flow_col].values
                        
                        # Add to transformed dataframe
                        if len(transformed_df) == 0:
                            # First frequency and model, initialize with flow column
                            transformed_df[f'Flow ({flow_unit})'] = flow_values
                        
                        # Add head column for this model and frequency
                        transformed_df[f'{model_name} {freq} Head ({head_unit})'] = head_values
            
            # Increment the reset key to force form refresh
            st.session_state.input_reset_key += 1
            
            # Show notification that form has been refreshed
            st.session_state.show_refresh_notification = True
            
            # Set chart_generated to True if we have data
            if not transformed_df.empty:
                st.session_state.chart_generated = True
                st.session_state.current_df = transformed_df
            
            # Return the transformed data
            return transformed_df
    
    return None

def handle_irregular_manual_input(language, flow_unit="LPM", head_text="Head (m)", flow_text="Flow", 
                                model_text="Model", add_point_text="Add Data Point", 
                                remove_point_text="Remove Data Point", edit_models_text="Edit Pump Models Data"):
    """Handle manual input for irregular data points where each model has different head points"""
    st.subheader(edit_models_text)
    
    # Get existing data points from session state
    data_points = st.session_state.irregular_data_points
    
    # Extract unique models from data points
    models = sorted(list(set([point["model"] for point in data_points])))
    
    # Create a DataFrame from the data points
    if data_points:
        df = pd.DataFrame(data_points)
    else:
        df = pd.DataFrame(columns=["head", "flow", "model"])
    
    # Add model selection
    selected_model = st.selectbox("Select Model", options=models + ["Add New Model"])
    
    # If "Add New Model" is selected, show input field for new model name
    if selected_model == "Add New Model":
        new_model = st.text_input("Enter New Model Name")
        if new_model and new_model not in models:
            selected_model = new_model
            models.append(new_model)
    
    # Create a form for adding new points
    with st.form("add_point_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_head = st.number_input(head_text, min_value=0.0, step=1.0)
        with col2:
            new_flow = st.number_input(flow_text, min_value=0.0, step=10.0)
        with col3:
            model = st.selectbox(model_text, options=models, index=models.index(selected_model) if selected_model in models else 0)
        
        submitted = st.form_submit_button(add_point_text)
        if submitted:
            # Add new data point
            data_points.append({"head": new_head, "flow": new_flow, "model": model})
            # Update session state
            st.session_state.irregular_data_points = data_points
            # Set flag to regenerate chart
            st.session_state.chart_generated = True
            # Force refresh
            st.rerun()
    
    # Filter data points for selected model
    model_points = [point for point in data_points if point["model"] == selected_model]
    
    # Display data points for selected model
    if model_points:
        model_df = pd.DataFrame(model_points)
        edited_df = st.data_editor(
            model_df,
            use_container_width=True,
            column_config={
                "head": st.column_config.NumberColumn(head_text, min_value=0.0, step=1.0),
                "flow": st.column_config.NumberColumn(flow_text, min_value=0.0, step=10.0),
                "model": st.column_config.TextColumn(model_text, disabled=True)
            },
            key=f"model_data_editor_{st.session_state.input_reset_key}_{selected_model}"
        )
        
        # Update session state with edited data
        if not edited_df.equals(model_df):
            # Remove old points for this model
            st.session_state.irregular_data_points = [p for p in data_points if p["model"] != selected_model]
            # Add updated points
            for _, row in edited_df.iterrows():
                st.session_state.irregular_data_points.append({
                    "head": row["head"],
                    "flow": row["flow"],
                    "model": row["model"]
                })
    
    # Button to remove the last point of the selected model
    if model_points:
        if st.button(remove_point_text):
            # Get index of last point for this model
            for i in range(len(data_points) - 1, -1, -1):
                if data_points[i]["model"] == selected_model:
                    data_points.pop(i)
                    break
            # Update session state
            st.session_state.irregular_data_points = data_points
            # Force refresh
            st.rerun()
    
    # Button to generate the pump curve
    generate_button = st.button(generate_curve_text)
    if generate_button:
        # Set flag to regenerate chart
        st.session_state.chart_generated = True
        # Convert data points to DataFrame
        df = pd.DataFrame(data_points)
        # Update manual input data with flow unit
        st.session_state.manual_input_data['flow_unit'] = flow_unit
        return df
    
    # If chart is already generated, return the DataFrame
    if st.session_state.chart_generated and data_points:
        df = pd.DataFrame(data_points)
        return df
    
    return None

def generate_template_data(num_points, model_names, use_template=True):
    """Generate template data for pump curves"""
    # Common head values for all models - Generate based on num_points
    template_head = np.linspace(5.0, 50.0, num_points).tolist()
    
    if use_template:
        # Flow values for each model at 50Hz - Generate curves based on num_points
        template_flow_50hz = {}
        
        # Generate base model flow curve with appropriate number of points
        base_flow_50hz = []
        for i in range(num_points):
            # Create a curve that gradually decreases from max to min flow
            # as head increases (non-linear curve with steeper drop-off at end)
            progress = i / (num_points - 1)  # 0 to 1
            # Apply non-linear transformation for more realistic curve
            flow_factor = 1 - (progress ** 1.5)  # Non-linear decrease
            flow_value = 90.0 * flow_factor
            base_flow_50hz.append(flow_value)
        
        # Set first model's flow
        template_flow_50hz[model_names[0]] = base_flow_50hz
        
        # Flow values for each model at 60Hz (20% higher)
        template_flow_60hz = {
            model_names[0]: [flow * 1.2 for flow in base_flow_50hz],
        }
        
        # Generate values for other models
        for i, model in enumerate(model_names):
            if i > 0:  # Skip first model which is already set
                multiplier = 1.0 + (0.15 * i)  # 15% increase for each model
                
                # Apply multiplier to base model flow values
                template_flow_50hz[model] = [flow * multiplier for flow in template_flow_50hz[model_names[0]]]
                template_flow_60hz[model] = [flow * multiplier for flow in template_flow_60hz[model_names[0]]]
    else:
        # Generate default head values (common for all models)
        # Generate flow values for each model at different frequencies
        template_flow_50hz = {}
        template_flow_60hz = {}
        base_flow_50hz = 90.0  # Starting flow for first model at 50Hz
        
        for i, model in enumerate(model_names):
            # Generate decreasing flow as head increases for 50Hz
            max_flow_50hz = base_flow_50hz * (1.0 + 0.15 * i)  # Increase capacity for each model
            min_flow_50hz = max_flow_50hz * 0.1  # End at 10% of max flow
            
            # 50Hz values
            flows_50hz = np.linspace(max_flow_50hz, min_flow_50hz, num_points).tolist()
            template_flow_50hz[model] = flows_50hz
            
            # 60Hz values (20% higher flow)
            template_flow_60hz[model] = [flow * 1.2 for flow in flows_50hz]
            
    return template_head, template_flow_50hz, template_flow_60hz

def generate_irregular_pump_curve(df, chart_style="Modern", show_system_curve=False, static_head=0.0, 
                                k_factor=0.0, min_flow=0.0, max_flow=None, min_head=0.0, max_head=None, 
                                show_grid=True, flow_unit="LPM"):
    """Generate pump curves for irregular data points where models have different head measurements"""
    # Create a larger figure to prevent text overlap
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Increase figure margins to make room for axes labels
    plt.subplots_adjust(bottom=0.2, right=0.9)
    
    # Get unique models from the data
    models = df['model'].unique()
    
    # Get color cycle for plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create a plot for each model
    for i, model in enumerate(models):
        # Get data for this model
        model_data = df[df['model'] == model]
        
        # Sort by head (important for correct line plotting)
        model_data = model_data.sort_values('head')
        
        # Plot the curve for this model
        color = colors[i % len(colors)]
        ax.plot(model_data['flow'], model_data['head'], 
                linestyle='-', linewidth=2.5, 
                label=f"{model}", color=color,
                marker='o', markersize=4)
        
        # Add model name label at the end of each curve
        if len(model_data) > 0:
            last_idx = len(model_data) - 1
            # Find the point with the highest head (usually 0 flow)
            max_head_idx = model_data['head'].idxmax()
            x_pos = model_data.iloc[max_head_idx]['flow']
            y_pos = model_data.iloc[max_head_idx]['head']
            x_padding = 5  # Fixed padding for annotation
            ax.annotate(f"{model}", 
                        xy=(x_pos, y_pos), 
                        xytext=(x_pos + x_padding, y_pos),
                        color=color, fontweight='bold', va='center')
    
    # Add system curve if requested
    if show_system_curve:
        # Use the same range for system curve as displayed in plot
        if max_flow is None:
            max_flow_val = df['flow'].max() * 1.2
        else:
            max_flow_val = max_flow
        
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve with red line
        ax.plot(system_flows, system_heads, 'r-', linewidth=2, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)')
        
        # Find and plot intersection points for all models
        for i, model in enumerate(models):
            # Get data for this model
            model_data = df[df['model'] == model]
            
            if len(model_data) >= 2:  # Need at least 2 points for interpolation
                color = colors[i % len(colors)]
                
                # Sort by flow for correct interpolation
                model_data = model_data.sort_values('flow')
                
                try:
                    # Create interpolation function for this model
                    # Need to flip x and y for interp1d to work
                    # We interpolate head as a function of flow
                    flows = model_data['flow'].values
                    heads = model_data['head'].values
                    
                    # Check if we have enough unique points
                    if len(np.unique(flows)) >= 2:
                        pump_curve_func = interp1d(flows, heads, bounds_error=False, fill_value="extrapolate")
                        
                        # Find intersection with system curve
                        # Create a function for the difference between curves
                        diff_func = lambda q: pump_curve_func(q) - (static_head + k_factor * q**2)
                        
                        # Try to find root (where difference is zero)
                        from scipy.optimize import fsolve
                        # Start from middle of flow range as initial guess
                        initial_guess = np.mean(flows)
                        try:
                            op_flow = fsolve(diff_func, initial_guess)[0]
                            
                            # Check if solution is within flow range
                            if min(flows) <= op_flow <= max(flows):
                                op_head = static_head + k_factor * op_flow**2
                                
                                # Plot operating point
                                ax.plot(op_flow, op_head, 'o', markersize=8, color=color)
                                
                                # Add operating point annotation
                                ax.annotate(f"{model}: ({op_flow:.1f}, {op_head:.1f})",
                                           xy=(op_flow, op_head),
                                           xytext=(10, 10),
                                           textcoords='offset points',
                                           color=color,
                                           fontweight='bold')
                        except:
                            # If fsolve fails, skip intersection for this model
                            pass
                except:
                    # If interpolation fails, skip intersection for this model
                    pass
    
    # Set up the primary x and y axes
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Head (m)', fontsize=12, fontweight='bold')
    
    # Limit number of ticks to prevent overcrowding
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    
    # Format tick labels to 1 decimal place
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # Add grid based on user preference
    if show_grid:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5)
        if chart_style == "Modern":
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply custom axis limits if provided
    # Set x-axis limits with expanded range for all data
    if max_flow is None:
        ax.set_xlim(left=float(min_flow), right=df['flow'].max() * 1.1)
    else:
        ax.set_xlim(left=float(min_flow), right=float(max_flow))
    
    # Set y-axis limits
    if max_head is None:
        # Find maximum head across all models
        max_head_val = df['head'].max()
        ax.set_ylim(bottom=float(min_head), top=max_head_val * 1.1)
    else:
        ax.set_ylim(bottom=float(min_head), top=float(max_head))
    
    # Add secondary x-axis for alternative flow units if primary is not already in those units
    if flow_unit == "LPM":
        # Add m³/h axis at bottom
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis below
        ax_gpm = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.264172, lambda x: x/0.264172))
        ax_gpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_gpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_gpm.set_xlabel(f'Flow (GPM)', fontsize=12, fontweight='bold', labelpad=10)
    elif flow_unit == "GPM":
        # Add LPM axis
        ax_lpm = ax.secondary_xaxis(-0.15, functions=(lambda x: x*3.78541, lambda x: x/3.78541))
        ax_lpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_lpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_lpm.set_xlabel(f'Flow (LPM)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.227125, lambda x: x/0.227125))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Add secondary y-axis for alternative head units
    # Position the ft axis
    ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
    ax_ft.yaxis.set_major_locator(MaxNLocator(7))
    ax_ft.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax_ft.set_ylabel(f'Head (ft)', fontsize=12, fontweight='bold', labelpad=15)
    
    # Add legend with model names
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
    plt.title('Pump Performance Curves', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def generate_pump_curve(df, frequency_option="Both", chart_style="Modern", show_system_curve=False, 
                       static_head=0.0, k_factor=0.0, refresh_counter=0, min_flow=0.0, max_flow=None,
                       min_head=0.0, max_head=None, show_grid=True):
    # Create a larger figure to prevent text overlap
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Increase figure margins to make room for axes labels
    plt.subplots_adjust(bottom=0.2, right=0.9)
    
    # Determine flow unit and head unit from column names
    flow_col = df.columns[0]
    flow_unit = flow_col.split('(')[-1].split(')')[0]
    
    # Determine which frequencies to plot based on data and user selection
    frequencies_present = []
    if any('50Hz' in col for col in df.columns):
        frequencies_present.append('50Hz')
    if any('60Hz' in col for col in df.columns):
        frequencies_present.append('60Hz')
    
    # If no frequency specified in columns, assume 50Hz only
    if not frequencies_present:
        frequencies_present = ['50Hz']
        
    # Filter frequencies based on user selection
    frequencies_to_plot = []
    if frequency_option == "50Hz Only" and '50Hz' in frequencies_present:
        frequencies_to_plot = ['50Hz']
    elif frequency_option == "60Hz Only" and '60Hz' in frequencies_present:
        frequencies_to_plot = ['60Hz']
    else:  # "Both" or mixed case
        frequencies_to_plot = frequencies_present
    # Get all head columns
    head_cols = [col for col in df.columns if 'Head' in col]
    
    # Group columns by model and frequency
    model_data = {}
    for col in head_cols:
        parts = col.split(' ')
        model_name = parts[0]  # Extract model name
        
        # Find frequency in the column name
        freq = None
        for part in parts:
            if part in ['50Hz', '60Hz']:
                freq = part
                break
        
        # If no frequency found, assume 50Hz
        if freq is None:
            freq = '50Hz'
            
        # Skip if frequency not selected to plot
        if freq not in frequencies_to_plot:
            continue
            
        # Get column unit
        head_unit = col.split('(')[-1].split(')')[0]
        
        # Create model entry if doesn't exist
        if model_name not in model_data:
            model_data[model_name] = {}
            
        # Store data for this model and frequency
        model_data[model_name][freq] = df[col].values
    
    # Get color cycle for plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot each pump model with distinct colors, with different line styles for frequencies
    model_names = list(model_data.keys())
    for i, model_name in enumerate(model_names):
        color = colors[i % len(colors)]
        
        # Plot different frequencies with different line styles
        for freq in frequencies_to_plot:
            if freq in model_data[model_name]:
                # Set line style based on frequency
                line_style = '-' if freq == '50Hz' else '--'
                
                # Plot curve for this model and frequency
                ax.plot(df[flow_col], model_data[model_name][freq], 
                        linestyle=line_style, linewidth=2.5, 
                        label=f"{model_name} ({freq})", color=color)
                
                # Add model name and frequency label at the end of each curve
                last_idx = len(model_data[model_name][freq]) - 1
                if last_idx >= 0:
                    x_pos = df[flow_col].iloc[last_idx]
                    y_pos = model_data[model_name][freq][last_idx]
                    x_padding = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
                    ax.annotate(f"{model_name} ({freq})", 
                                xy=(x_pos + x_padding, y_pos), 
                                color=color, fontweight='bold', va='center')
    
    # Add system curve if requested
    if show_system_curve:
        max_flow_val = df[flow_col].max() * 1.5 if max_flow is None else max_flow
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve with red line
        ax.plot(system_flows, system_heads, 'r-', linewidth=2, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)')
        
        # Find and plot intersection points for all models and frequencies
        for i, model_name in enumerate(model_names):
            color = colors[i % len(colors)]
            
            for freq in frequencies_to_plot:
                if freq in model_data[model_name]:
                    # Get pump curve data
                    pump_heads = model_data[model_name][freq]
                    
                    # Interpolate pump curve for intersection
                    pump_heads_interp = np.interp(system_flows, df[flow_col], pump_heads, 
                                                 left=np.nan, right=np.nan)
                    
                    # Find intersection point
                    diff = np.abs(pump_heads_interp - system_heads)
                    valid_idx = ~np.isnan(diff)
                    if np.any(valid_idx):
                        op_idx = np.argmin(diff[valid_idx])
                        op_flow = system_flows[valid_idx][op_idx]
                        op_head = pump_heads_interp[valid_idx][op_idx]
                        
                        # Set marker style based on frequency
                        marker_style = 'o' if freq == '50Hz' else 's'
                        
                        # Plot operating point
                        ax.plot(op_flow, op_head, marker_style, markersize=8, color=color)
                        
                        # Add operating point annotation
                        ax.annotate(f"{freq}: ({op_flow:.1f}, {op_head:.1f})",
                                   xy=(op_flow, op_head),
                                   xytext=(10, (5 if freq == '50Hz' else -15)),
                                   textcoords='offset points',
                                   color=color,
                                   fontweight='bold')
    
    # Set up the primary x and y axes
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Head ({head_unit})', fontsize=12, fontweight='bold')
    
    # Limit number of ticks to prevent overcrowding
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    
    # Format tick labels to 1 decimal place
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # Add grid based on user preference
    if show_grid:
        ax.grid(True, which='major', linestyle='-', linewidth=0.5)
        if chart_style == "Modern":
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    else:
        ax.grid(False)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply custom axis limits if provided
    # Set x-axis limits with expanded range for all data
    if max_flow is None:
        ax.set_xlim(left=float(min_flow), right=df[flow_col].max() * 1.1)
    else:
        ax.set_xlim(left=float(min_flow), right=float(max_flow))
    
    # Set y-axis limits
    if max_head is None:
        # Find maximum head across all models and frequencies
        max_head_val = 0
        for model_name in model_data:
            for freq in model_data[model_name]:
                max_head_val = max(max_head_val, np.max(model_data[model_name][freq]))
                
        ax.set_ylim(bottom=float(min_head), top=max_head_val * 1.1)
    else:
        ax.set_ylim(bottom=float(min_head), top=float(max_head))
    
    # Add secondary x-axis for alternative flow units if primary is not already in those units
    if flow_unit == "LPM":
        # Add m³/h axis at bottom
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis below
        ax_gpm = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.264172, lambda x: x/0.264172))
        ax_gpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_gpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_gpm.set_xlabel(f'Flow (GPM)', fontsize=12, fontweight='bold', labelpad=10)
    elif flow_unit == "GPM":
        # Add LPM axis
        ax_lpm = ax.secondary_xaxis(-0.15, functions=(lambda x: x*3.78541, lambda x: x/3.78541))
        ax_lpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_lpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_lpm.set_xlabel(f'Flow (LPM)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.227125, lambda x: x/0.227125))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Add secondary y-axis for alternative head units
    if head_unit == "m":
        # Position the ft axis
        ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
        ax_ft.yaxis.set_major_locator(MaxNLocator(7))
        ax_ft.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax_ft.set_ylabel(f'Head (ft)', fontsize=12, fontweight='bold', labelpad=15)
    elif head_unit == "ft":
        # Position the meters axis
        ax_m = ax.secondary_yaxis(1.05, functions=(lambda x: x/3.28084, lambda x: x*3.28084))
        ax_m.yaxis.set_major_locator(MaxNLocator(7))
        ax_m.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax_m.set_ylabel(f'Head (m)', fontsize=12, fontweight='bold', labelpad=15)
    
    # Add frequency information at top left corner
    if len(frequencies_to_plot) == 1:
        # Single frequency
        freq_text = frequencies_to_plot[0]
    else:
        # Multiple frequencies
        freq_text = " / ".join(frequencies_to_plot)
        
    plt.text(0.02, 0.98, freq_text, 
             transform=ax.transAxes, 
             fontsize=14,
             fontweight='bold',
             horizontalalignment='left',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add legend with model names and frequencies
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
    plt.title('Pump Performance Curves', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def download_button_for_plot(fig, language):
    # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Create download button with language-specific label
    if language == "Chinese":
        download_label = "下載圖表 (PNG)"
    else:
        download_label = "Download Plot (PNG)"
        
    btn = st.download_button(
        label=download_label,
        data=buf,
        file_name="pump_curve_plot.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()
