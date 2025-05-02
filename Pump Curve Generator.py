import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import matplotlib.font_manager as fm
import os

# Check if Chinese font is available
chinese_fonts = [f for f in fm.findSystemFonts() if 'noto' in f.lower() and 'cjk' in f.lower()]
if chinese_fonts:
    chinese_font = chinese_fonts[0]
else:
    chinese_font = None  # Will fall back to a default font

# Translation dictionary for all UI elements
translations = {
    'en': {
        # Page settings
        'page_title': 'Pump Curve Generator',
        'title': 'Pump Curve Generator Tool',
        'intro': 'This tool allows you to generate pump performance curves similar to manufacturer specifications. First, configure your chart settings, then upload or enter your pump data to generate the curve.',
        
        # Tabs
        'tab_create': 'Create Pump Curves',
        'tab_about': 'About Pump Curves',
        
        # Configuration
        'chart_config': 'Chart Configuration',
        'freq_display': 'Frequency Display',
        'freq_50hz': '50Hz Only',
        'freq_60hz': '60Hz Only',
        'freq_both': 'Both',
        'chart_style': 'Chart Style',
        'style_modern': 'Modern',
        'style_classic': 'Classic',
        'show_system': 'Show System Curve',
        'show_grid': 'Show Grid',
        'static_head': 'Static Head (m)',
        'friction_factor': 'Friction Factor (k)',
        
        # Axis settings
        'axis_settings': 'Axis Range Settings',
        'min_flow': 'Min Flow',
        'max_flow': 'Max Flow (0 for auto)',
        'min_head': 'Min Head',
        'max_head': 'Max Head (0 for auto)',
        
        # Input method
        'input_method': 'Select Input Method',
        'upload_csv': 'Upload CSV',
        'manual_input': 'Manual Input',
        
        # CSV upload
        'upload_title': 'Upload CSV File',
        'choose_csv': 'Choose a CSV file',
        'sample_template': 'Download Sample CSV Template',
        'download_template': 'Download CSV Template',
        'head_format': 'Detected head-first format data. Converting to the format needed for the chart.',
        'no_flow_columns': 'No flow columns detected in the CSV. Please ensure column names contain \'Flow\'.',
        'no_flow_column': 'No Flow column detected. Please ensure your CSV has at least one Flow column.',
        
        # Manual input
        'manual_title': 'Manual Data Input',
        'excel_tip': 'You can copy data from Excel and paste directly into these tables. Select cells in Excel, copy (Ctrl+C), click on the starting cell in the table below, and paste (Ctrl+V).',
        'flow_unit': 'Flow Rate Unit',
        'head_unit': 'Head Unit',
        'num_models': 'Number of Pump Models',
        'model_name': 'Model {} Name',
        'use_template': 'Use Template Data',
        'num_points': 'Number of Data Points',
        'edit_data': 'Edit Pump Data Below',
        'edit_freq_data': 'Edit {} pump data below. Head values are common across models.',
        'data_freq': '{} Data',
        
        # Buttons
        'generate_curve': 'Generate Pump Curve',
        'refresh_form': 'Refresh Form',
        'download_plot': 'Download Plot (PNG)',
        
        # Plot elements
        'flow_label': 'Flow ({})',
        'head_label': 'Head ({})',
        'system_curve': 'System Curve (H={}+{}×Q²)',
        'operating_point': '{}: ({}, {})',
        
        # About section
        'understanding': 'Understanding Pump Curves',
        'what_is': 'What is a Pump Curve?',
        'what_is_desc': '''
        A pump curve (or performance curve) graphically represents the relationship between:
        
        - **Flow Rate**: The volume of liquid a pump can move per unit time (measured in LPM, m³/h, or GPM)
        - **Head**: The pressure or height to which a pump can raise liquid (measured in meters or feet)
        ''',
        'reading': 'Reading Pump Curves',
        'reading_desc': '''
        - Each curve represents a specific pump model or impeller size
        - The x-axis shows flow rate
        - The y-axis shows head
        - As flow increases, head typically decreases
        - The operating point of a pump is determined by where the pump curve intersects with the system curve
        ''',
        'system': 'System Curve',
        'system_desc': '''
        A system curve represents the resistance in your piping system:
        
        - It consists of static head (vertical height) and friction losses
        - Mathematically expressed as: H = Hs + k × Q²
          - H = Total head
          - Hs = Static head
          - k = Friction coefficient
          - Q = Flow rate
        ''',
        'frequency': 'Frequency Impact (50Hz vs 60Hz)',
        'frequency_desc': '''
        Changing the electrical frequency affects pump performance:
        
        - Flow (Q) is proportional to speed (n): Q₂ = Q₁ × (n₂/n₁)
        - Head (H) is proportional to speed squared: H₂ = H₁ × (n₂/n₁)²
        - Power (P) is proportional to speed cubed: P₂ = P₁ × (n₂/n₁)³
        
        For 50Hz to 60Hz conversion:
        - Flow increases by 20% (60/50 = 1.2)
        - Head increases by 44% (1.2² = 1.44)
        - Power increases by 73% (1.2³ = 1.728)
        ''',
        'selecting': 'Selecting the Right Pump',
        'selecting_desc': '''
        When selecting a pump, consider:
        1. Required flow rate
        2. Required head
        3. System efficiency
        4. NPSH (Net Positive Suction Head)
        5. Power consumption
        ''',
        
        # Language selection
        'language': 'Language',
        'lang_en': 'English',
        'lang_zh': '繁體中文'
    },
    'zh': {
        # Page settings
        'page_title': '水泵曲線生成器',
        'title': '水泵曲線生成工具',
        'intro': '此工具可讓您生成類似於製造商規格的水泵性能曲線。首先，配置您的圖表設置，然後上傳或輸入您的水泵數據以生成曲線。',
        
        # Tabs
        'tab_create': '建立水泵曲線',
        'tab_about': '關於水泵曲線',
        
        # Configuration
        'chart_config': '圖表配置',
        'freq_display': '頻率顯示',
        'freq_50hz': '僅 50Hz',
        'freq_60hz': '僅 60Hz',
        'freq_both': '兩者皆顯示',
        'chart_style': '圖表風格',
        'style_modern': '現代風格',
        'style_classic': '傳統風格',
        'show_system': '顯示系統曲線',
        'show_grid': '顯示網格',
        'static_head': '靜壓頭 (m)',
        'friction_factor': '摩擦係數 (k)',
        
        # Axis settings
        'axis_settings': '坐標軸範圍設置',
        'min_flow': '最小流量',
        'max_flow': '最大流量 (0 為自動)',
        'min_head': '最小壓頭',
        'max_head': '最大壓頭 (0 為自動)',
        
        # Input method
        'input_method': '選擇輸入方法',
        'upload_csv': '上傳 CSV',
        'manual_input': '手動輸入',
        
        # CSV upload
        'upload_title': '上傳 CSV 文件',
        'choose_csv': '選擇 CSV 文件',
        'sample_template': '下載範例 CSV 模板',
        'download_template': '下載 CSV 模板',
        'head_format': '檢測到壓頭優先格式數據。正在轉換為圖表所需的格式。',
        'no_flow_columns': 'CSV 中未檢測到流量列。請確保列名包含「Flow」。',
        'no_flow_column': '未檢測到流量列。請確保您的 CSV 至少有一個流量列。',
        
        # Manual input
        'manual_title': '手動數據輸入',
        'excel_tip': '您可以從 Excel 複製數據並直接粘貼到這些表格中。在 Excel 中選擇單元格，複製 (Ctrl+C)，點擊下表中的起始單元格，然後粘貼 (Ctrl+V)。',
        'flow_unit': '流量單位',
        'head_unit': '壓頭單位',
        'num_models': '水泵型號數量',
        'model_name': '型號 {} 名稱',
        'use_template': '使用模板數據',
        'num_points': '數據點數量',
        'edit_data': '編輯下方水泵數據',
        'edit_freq_data': '編輯下方 {} 水泵數據。壓頭值在所有型號間共用。',
        'data_freq': '{} 數據',
        
        # Buttons
        'generate_curve': '生成水泵曲線',
        'refresh_form': '重置表單',
        'download_plot': '下載圖表 (PNG)',
        
        # Plot elements
        'plot_title': '水泵性能曲線',
        'flow_label': '流量 ({})',
        'head_label': '壓頭 ({})',
        'system_curve': '系統曲線 (H={}+{}×Q²)',
        'operating_point': '{}: ({}, {})',
        
        # About section
        'understanding': '了解水泵曲線',
        'what_is': '什麼是水泵曲線？',
        'what_is_desc': '''
        水泵曲線（或性能曲線）以圖形方式表示以下關係：
        
        - **流量**：水泵每單位時間可移動的液體體積（以 LPM、m³/h 或 GPM 測量）
        - **壓頭**：水泵可以將液體提升的壓力或高度（以米或英尺測量）
        ''',
        'reading': '閱讀水泵曲線',
        'reading_desc': '''
        - 每條曲線代表特定的水泵型號或葉輪尺寸
        - X 軸顯示流量
        - Y 軸顯示壓頭
        - 隨著流量增加，壓頭通常會降低
        - 水泵的工作點由水泵曲線與系統曲線的交點確定
        ''',
        'system': '系統曲線',
        'system_desc': '''
        系統曲線表示管道系統中的阻力：
        
        - 它由靜壓頭（垂直高度）和摩擦損失組成
        - 數學表達式：H = Hs + k × Q²
          - H = 總壓頭
          - Hs = 靜壓頭
          - k = 摩擦係數
          - Q = 流量
        ''',
        'frequency': '頻率影響 (50Hz vs 60Hz)',
        'frequency_desc': '''
        改變電頻會影響水泵性能：
        
        - 流量 (Q) 與速度 (n) 成正比：Q₂ = Q₁ × (n₂/n₁)
        - 壓頭 (H) 與速度的平方成正比：H₂ = H₁ × (n₂/n₁)²
        - 功率 (P) 與速度的立方成正比：P₂ = P₁ × (n₂/n₁)³
        
        從 50Hz 轉換到 60Hz：
        - 流量增加 20%（60/50 = 1.2）
        - 壓頭增加 44%（1.2² = 1.44）
        - 功率增加 73%（1.2³ = 1.728）
        ''',
        'selecting': '選擇合適的水泵',
        'selecting_desc': '''
        選擇水泵時，請考慮：
        1. 所需流量
        2. 所需壓頭
        3. 系統效率
        4. NPSH（淨正吸入壓頭）
        5. 功耗
        ''',
        
        # Language selection
        'language': '語言',
        'lang_en': 'English',
        'lang_zh': '繁體中文'
    }
}

def main():
    # Set default language if not in session state
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    
    # Create a function to auto-update chart when configuration changes
    def update_chart_on_config_change():
        if st.session_state.current_df is not None and st.session_state.chart_generated:
            st.rerun()
    
    # Get the translation dictionary for the current language
    t = translations[st.session_state.language]
    
    st.set_page_config(page_title=t['page_title'], layout="wide")
    
    # Language switcher in sidebar
    with st.sidebar:
        st.subheader(t['language'])
        selected_lang = st.radio(
            label="",
            options=['en', 'zh'],
            format_func=lambda x: translations[x]['lang_' + x],
            index=0 if st.session_state.language == 'en' else 1,
            key="language_selector"
        )
        
        # Update language if changed
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
    
    st.title(t['title'])
    
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
    
    st.markdown(t['intro'])
    
    tab1, tab2 = st.tabs([t['tab_create'], t['tab_about']])
    
    with tab1:
        # Configuration options
        st.subheader(t['chart_config'])
        
        # Create columns for chart options
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
        
        # When any option changes, immediately update session state and trigger chart refresh
        with col_a:
            frequency_options = [t['freq_50hz'], t['freq_60hz'], t['freq_both']]
            frequency_option = st.selectbox(
                t['freq_display'], 
                frequency_options,
                index=frequency_options.index(
                    t[f"freq_{st.session_state.chart_params.get('frequency_option', 'Both').lower().replace(' ', '')}"]
                    if st.session_state.chart_params.get('frequency_option', 'Both') in ["50Hz Only", "60Hz Only", "Both"]
                    else frequency_options.index(t['freq_both'])
                ),
                key="frequency_option_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'frequency_option': 
                         "50Hz Only" if st.session_state.frequency_option_select == t['freq_50hz'] else
                         "60Hz Only" if st.session_state.frequency_option_select == t['freq_60hz'] else
                         "Both"
                        }
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_b:
            style_options = [t['style_modern'], t['style_classic']]
            chart_style = st.selectbox(
                t['chart_style'], 
                style_options, 
                index=style_options.index(
                    t['style_modern'] if st.session_state.chart_params['chart_style'] == "Modern" else t['style_classic']
                ),
                key="chart_style_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'chart_style': 
                         "Modern" if st.session_state.chart_style_select == t['style_modern'] else "Classic"}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_c:
            show_system = st.checkbox(
                t['show_system'], 
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
                t['show_grid'], 
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
        
        # System curve parameters (only shown if show_system_curve is True)
        if st.session_state.chart_params['show_system_curve']:
            col_e, col_f = st.columns(2)
            with col_e:
                static_head = st.number_input(
                    t['static_head'], 
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
                    t['friction_factor'], 
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
        st.subheader(t['axis_settings'])
        col_g, col_h, col_i, col_j = st.columns(4)
        
        with col_g:
            min_flow = st.number_input(
                t['min_flow'], 
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
                t['max_flow'], 
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
                t['min_head'], 
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
                t['max_head'], 
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
        
        # Sidebar for input method selection
        input_method = st.radio(
            t['input_method'],
            [t['upload_csv'], t['manual_input']]
        )
        
        if input_method == t['upload_csv']:
            df = handle_csv_upload(t)
        else:
            df = handle_manual_input(
                frequency_option=st.session_state.chart_params.get('frequency_option', "Both"),
                t=t
            )
        
        # Generate and display the pump curve if data is available
        if df is not None and not df.empty:
            st.session_state.current_df = df  # Store the current dataframe
            
            # For CSV uploads, automatically set chart_generated to True
            if input_method == t['upload_csv'] and not st.session_state.chart_generated:
                st.session_state.chart_generated = True
            
            # Generate curve using parameters from session state
            if st.session_state.chart_generated:
                params = st.session_state.chart_params
                try:
                    # Pass frequency option to the plotting function
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
                        show_grid=params['show_grid'],
                        t=t,
                        lang=st.session_state.language
                    )
                    st.pyplot(fig)
                    
                    # Add download button for the plot
                    download_button_for_plot(fig, t)
                except Exception as e:
                    st.error(f"Error generating chart: {e}")
            else:
                st.info("Click Generate Chart to create the pump curve.")
    
    with tab2:
        st.subheader(t['understanding'])
        st.markdown(f"### {t['what_is']}")
        st.markdown(t['what_is_desc'])
        
        st.markdown(f"### {t['reading']}")
        st.markdown(t['reading_desc'])
        
        st.markdown(f"### {t['system']}")
        st.markdown(t['system_desc'])
        
        st.markdown(f"### {t['frequency']}")
        st.markdown(t['frequency_desc'])
        
        st.markdown(f"### {t['selecting']}")
        st.markdown(t['selecting_desc'])

def handle_csv_upload(t):
    st.subheader(t['upload_title'])
    
    # File uploader
    uploaded_file = st.file_uploader(t['choose_csv'], type="csv")
    
    # Sample CSV template for download - Updated to include both 50Hz and 60Hz data
    sample_data = pd.DataFrame({
        'Head (ft)': [4.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 49.8],
        'Model-A 50Hz Flow (GPM)': [90.09, 89.96, 84.81, 77.43, 69.85, 61.89, 53.93, 41.81, 29.34, 18.52, 8.45],
        'Model-A 60Hz Flow (GPM)': [108.11, 107.95, 101.77, 92.92, 83.82, 74.27, 64.72, 50.17, 35.21, 22.22, 10.14],
        'Model-B 50Hz Flow (GPM)': [105.50, 104.75, 95.22, 86.34, 78.43, 69.45, 60.76, 48.45, 35.21, 21.93, 11.24],
        'Model-B 60Hz Flow (GPM)': [126.60, 125.70, 114.26, 103.61, 94.12, 83.34, 72.91, 58.14, 42.25, 26.32, 13.49]
    })
    
    st.markdown(f"### {t['sample_template']}")
    
    # Create template with both frequencies
    csv_standard = sample_data.to_csv(index=False)
    b64_standard = base64.b64encode(csv_standard.encode()).decode()
    href_standard = f'<a href="data:file/csv;base64,{b64_standard}" download="pump_curve_template.csv">{t["download_template"]}</a>'
    
    st.markdown(href_standard, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Transform the data into format needed for plotting (Head vs Flow)
            if 'Head' in df.columns[0]:
                st.info(t['head_format'])
                
                # Extract head values and units
                head_col = df.columns[0]
                head_unit = head_col.split('(')[-1].split(')')[0]
                head_values = df[head_col].values
                
                # Create new DataFrame for transformed data
                transformed_df = pd.DataFrame()
                
                # Get all flow columns
                flow_cols = [col for col in df.columns if 'Flow' in col]
                
                if not flow_cols:
                    st.warning(t['no_flow_columns'])
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
                st.warning(t['no_flow_column'])
                return None
            
            # Automatically set chart_generated to True when a file is uploaded
            st.session_state.chart_generated = True
                
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    
    return None

def handle_manual_input(frequency_option="Both", t=None):
    st.subheader(t['manual_title'])
    
    # Add instructions for copying from Excel
    st.info(t['excel_tip'])
    
    # Initialize session state for manually input data if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    # Create a form for manual input
    with st.form(f"manual_input_form_{st.session_state.input_reset_key}"):
        # Units selection
        col1, col2 = st.columns(2)
        with col1:
            flow_unit = st.selectbox(t['flow_unit'], ["GPM", "LPM", "m³/h"], 
                                   key=f"flow_unit_{st.session_state.input_reset_key}")
        with col2:
            head_unit = st.selectbox(t['head_unit'], ["ft", "m"], 
                                   key=f"head_unit_{st.session_state.input_reset_key}")
        
        # Number of pump models
        num_models = st.number_input(t['num_models'], min_value=1, max_value=5, value=2,
                                   key=f"num_models_{st.session_state.input_reset_key}")
        
        # Model names
        model_names = []
        cols = st.columns(min(num_models, 5))
        for i, col in enumerate(cols):
            model_name = col.text_input(t['model_name'].format(i+1), value=f"Model-{chr(65+i)}",
                                      key=f"model_name_{i}_{st.session_state.input_reset_key}")
            model_names.append(model_name)
        
        # Frequencies to display (based on frequency_option)
        frequencies_to_show = []
        if frequency_option == "50Hz Only" or frequency_option == "Both":
            frequencies_to_show.append("50Hz")
        if frequency_option == "60Hz Only" or frequency_option == "Both":
            frequencies_to_show.append("60Hz")
            
        # Option to use template data
        use_template = st.checkbox(t['use_template'], value=True,
                                 key=f"use_template_{st.session_state.input_reset_key}")
        
        # Template data for different frequencies
        if use_template:
            # Common head values for all models
            num_points = 11
            template_head = [4.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 49.8]
            
            # Flow values for each model at 50Hz
            template_flow_50hz = {
                model_names[0]: [90.09, 89.96, 84.81, 77.43, 69.85, 61.89, 53.93, 41.81, 29.34, 18.52, 8.45],
            }
            
            # Flow values for each model at 60Hz
            template_flow_60hz = {
                model_names[0]: [108.11, 107.95, 101.77, 92.92, 83.82, 74.27, 64.72, 50.17, 35.21, 22.22, 10.14],
            }
            
            # Generate values for other models
            for i, model in enumerate(model_names):
                if i > 0:  # Skip first model which is already set
                    multiplier = 1.0 + (0.15 * i)  # 15% increase for each model
                    
                    # Apply multiplier to base model flow values
                    template_flow_50hz[model] = [flow * multiplier for flow in template_flow_50hz[model_names[0]]]
                    template_flow_60hz[model] = [flow * multiplier for flow in template_flow_60hz[model_names[0]]]
        else:
            # Number of data points
            num_points = st.number_input(t['num_points'], min_value=3, max_value=20, value=8,
                                      key=f"num_points_{st.session_state.input_reset_key}")
            
            # Generate default head values (common for all models)
            template_head = np.linspace(5.0, 50.0, num_points).tolist()
            
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
        
        # Create editable dataframes for each frequency
        st.markdown(f"### {t['edit_data']}")
        
        # Create tabs for each frequency
        frequency_tabs = st.tabs([t['data_freq'].format(freq) for freq in frequencies_to_show])
        
        # Dictionary to store edited data for each frequency
        edited_data = {}
        
        # Create data editor for each frequency
        for i, freq in enumerate(frequencies_to_show):
            with frequency_tabs[i]:
                st.info(t['edit_freq_data'].format(freq))
                
                # Create the base dataframe for this frequency
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
            submitted = st.form_submit_button(t['generate_curve'])
        with col2:
            # Keep the Refresh Form button
            refresh_data = st.form_submit_button(t['refresh_form'])
        
        if submitted:
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
            # Just return the current data to update the form
            return None
    
    return None

def generate_pump_curve(df, frequency_option="Both", chart_style="Modern", show_system_curve=False, 
                       static_head=0.0, k_factor=0.0, refresh_counter=0, min_flow=0.0, max_flow=None,
                       min_head=0.0, max_head=None, show_grid=True, t=None, lang="en"):
    # Create a larger figure to prevent text overlap
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set Chinese font if available and language is Chinese
    if lang == "zh" and chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False
    
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
                label=t['system_curve'].format(static_head, k_factor))
        
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
                        ax.annotate(t['operating_point'].format(freq, f"{op_flow:.1f}", f"{op_head:.1f}"),
                                   xy=(op_flow, op_head),
                                   xytext=(10, (5 if freq == '50Hz' else -15)),
                                   textcoords='offset points',
                                   color=color,
                                   fontweight='bold')
    
    # Set up the primary x and y axes
    ax.set_xlabel(t['flow_label'].format(flow_unit), fontsize=12, fontweight='bold')
    ax.set_ylabel(t['head_label'].format(head_unit), fontsize=12, fontweight='bold')
    
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
        ax_m3h.set_xlabel(t['flow_label'].format("m³/h"), fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis below
        ax_gpm = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.264172, lambda x: x/0.264172))
        ax_gpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_gpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_gpm.set_xlabel(t['flow_label'].format("GPM"), fontsize=12, fontweight='bold', labelpad=10)
    elif flow_unit == "GPM":
        # Add LPM axis
        ax_lpm = ax.secondary_xaxis(-0.15, functions=(lambda x: x*3.78541, lambda x: x/3.78541))
        ax_lpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_lpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_lpm.set_xlabel(t['flow_label'].format("LPM"), fontsize=12, fontweight='bold', labelpad=10)
        
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.227125, lambda x: x/0.227125))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(t['flow_label'].format("m³/h"), fontsize=12, fontweight='bold', labelpad=10)
    
    # Add secondary y-axis for alternative head units
    if head_unit == "m":
        # Position the ft axis
        ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
        ax_ft.yaxis.set_major_locator(MaxNLocator(7))
        ax_ft.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax_ft.set_ylabel(t['head_label'].format("ft"), fontsize=12, fontweight='bold', labelpad=15)
    elif head_unit == "ft":
        # Position the meters axis
        ax_m = ax.secondary_yaxis(1.05, functions=(lambda x: x/3.28084, lambda x: x*3.28084))
        ax_m.yaxis.set_major_locator(MaxNLocator(7))
        ax_m.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax_m.set_ylabel(t['head_label'].format("m"), fontsize=12, fontweight='bold', labelpad=15)
    
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
    
    plt.title(t['plot_title'], fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def download_button_for_plot(fig, t):
    # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Create download button
    btn = st.download_button(
        label=t['download_plot'],
        data=buf,
        file_name="pump_curve_plot.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()
