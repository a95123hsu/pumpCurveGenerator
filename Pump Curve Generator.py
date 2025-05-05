import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def main():
    # Create a function to auto-update chart when configuration changes
    def update_chart_on_config_change():
        if st.session_state.current_df is not None and st.session_state.chart_generated:
            st.rerun()

    st.set_page_config(page_title="Pump Curve Generator", layout="wide")
    
    st.title("Pump Curve Generator Tool")
    
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
    
    st.markdown("""
    This tool allows you to generate pump performance curves similar to manufacturer specifications.
    First, configure your chart settings, then upload or enter your pump data to generate the curve.
    """)
    
    tab1, tab2 = st.tabs(["Create Pump Curves", "About Pump Curves"])
    
    with tab1:
        # Configuration options
        st.subheader("Chart Configuration")
        
        # Create columns for chart options
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
        
        # When any option changes, immediately update session state and trigger chart refresh
        with col_a:
            frequency_option = st.selectbox(
                "Frequency Display", 
                ["50Hz Only", "60Hz Only", "Both"],
                index=["50Hz Only", "60Hz Only", "Both"].index(st.session_state.chart_params.get('frequency_option', "Both")),
                key="frequency_option_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'frequency_option': st.session_state.frequency_option_select}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_b:
            chart_style = st.selectbox(
                "Chart Style", 
                ["Modern", "Classic"], 
                index=["Modern", "Classic"].index(st.session_state.chart_params['chart_style']),
                key="chart_style_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'chart_style': st.session_state.chart_style_select}
                    ),
                    update_chart_on_config_change()
                )
            )
        
        with col_c:
            show_system = st.checkbox(
                "Show System Curve", 
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
                "Show Grid", 
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
                    "Static Head (m)", 
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
                    "Friction Factor (k)", 
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
        st.subheader("Axis Range Settings")
        col_g, col_h, col_i, col_j = st.columns(4)
        
        with col_g:
            min_flow = st.number_input(
                "Min Flow", 
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
                "Max Flow (0 for auto)", 
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
                "Min Head", 
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
                "Max Head (0 for auto)", 
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
        
        # Input method selection
        input_method = st.radio(
            "Select Input Method",
            ["Upload CSV", "Manual Input"]
        )
        
        if input_method == "Upload CSV":
            df = handle_csv_upload()
        else:
            df = handle_manual_input(st.session_state.chart_params.get('frequency_option', "Both"))
        
        # Generate and display the pump curve if data is available
        if df is not None and not df.empty:
            st.session_state.current_df = df  # Store the current dataframe
            
            # For CSV uploads, automatically set chart_generated to True
            if input_method == "Upload CSV" and not st.session_state.chart_generated:
                st.session_state.chart_generated = True
            
            # Generate curve using parameters from session state
            if st.session_state.chart_generated:
                params = st.session_state.chart_params
                try:
                    # Debug: Display the dataframe structure
                    with st.expander("Debug: View Data Structure"):
                        st.write("DataFrame Structure:")
                        st.write(df)
                    
                    # Check if data is in standard format (with Flow column)
                    is_standard_format = any('Flow' in col for col in df.columns)
                    
                    # Pass data to appropriate plotting function
                    if is_standard_format:
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
                            debug_mode=True  # Enable debug mode
                        )
                    else:
                        # Handle data with flow values in each column (head first format)
                        fig = generate_pump_curve_head_first(
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
                            debug_mode=True  # Enable debug mode
                        )
                    
                    st.pyplot(fig)
                    
                    # Add download button for the plot
                    download_button_for_plot(fig)
                except Exception as e:
                    st.error(f"Error generating chart: {e}")
                    st.error("Please check your data format and try again.")
                    st.write("Debug info:", df.columns)
            else:
                st.info("Click Generate Chart to create the pump curve.")
    
    with tab2:
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
        """)

def debug_plot_data(df, title="DataFrame Debug Info"):
    """Print debugging information about the dataframe"""
    debug_info = []
    debug_info.append(f"=== {title} ===")
    debug_info.append(f"Columns in dataframe: {df.columns.tolist()}")
    debug_info.append("Data sample:")
    debug_info.append(str(df.head()))
    
    # Check for NaN or strange values in each model's data
    for col in df.columns:
        if 'Head' in col or 'Flow' in col:
            debug_info.append(f"Stats for {col}:")
            debug_info.append(f"  Min: {df[col].min()}")
            debug_info.append(f"  Max: {df[col].max()}")
            debug_info.append(f"  NaN count: {df[col].isna().sum()}")
    
    return "\n".join(debug_info)

def handle_csv_upload():
    st.subheader("Upload CSV File")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Sample CSV templates
    st.markdown("### Download Sample CSV Templates")
    
    # Standard format template (flow first)
    sample_data_standard = pd.DataFrame({
        'Flow (LPM)': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'Model-A 50Hz Head (m)': [50, 49, 47, 44, 40, 35, 29, 22, 14, 5, 0],
        'Model-A 60Hz Head (m)': [72, 70.6, 67.7, 63.4, 57.6, 50.4, 41.8, 31.7, 20.2, 7.2, 0],
        'Model-B 50Hz Head (m)': [60, 59, 56.5, 53, 48, 42, 35, 26.5, 16.5, 6, 0],
        'Model-B 60Hz Head (m)': [86.4, 84.5, 81.4, 76.3, 69.1, 60.5, 50.4, 38.2, 23.8, 8.6, 0]
    })
    
    # Alternate format template (head first)
    sample_data_alternate = pd.DataFrame({
        'Head (m)': [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0],
        'Model-A Flow (LPM)': [0, 15, 30, 42, 52, 61, 70, 77, 85, 95, 100],
        'Model-B Flow (LPM)': [0, 18, 35, 50, 62, 72, 81, 90, 97, 105, 110]
    })
    
    # Create download links for both templates
    csv_standard = sample_data_standard.to_csv(index=False)
    b64_standard = base64.b64encode(csv_standard.encode()).decode()
    href_standard = f'<a href="data:file/csv;base64,{b64_standard}" download="pump_curve_template_standard.csv">Download Standard Format (Flow First)</a>'
    
    csv_alternate = sample_data_alternate.to_csv(index=False)
    b64_alternate = base64.b64encode(csv_alternate.encode()).decode()
    href_alternate = f'<a href="data:file/csv;base64,{b64_alternate}" download="pump_curve_template_alternate.csv">Download Alternate Format (Head First)</a>'
    
    st.markdown(f"{href_standard} &nbsp; | &nbsp; {href_alternate}", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data
            st.write("Uploaded Data Preview:")
            st.dataframe(df.head())
            
            # Check if this is an alternate format (Head vs Flow)
            if 'Head' in df.columns[0] and not any(col.startswith('Flow (') for col in df.columns):
                # This is likely the alternate format
                st.info("Detected head-first format. Processing as is.")
                
                # No transformation needed for alternate format
                # Automatically set chart_generated to True when a file is uploaded
                st.session_state.chart_generated = True
                return df
            
            # Check if this is already in the standard format
            if any('Flow (' in col for col in df.columns) and any('Head (' in col for col in df.columns):
                st.info("Detected standard flow-first format. Processing as is.")
                # Already in standard format - no transformation needed
                st.session_state.chart_generated = True
                return df
                
            # Transform the data if it's in a different format
            if 'Head' in df.columns[0]:
                st.info("Detected head-first format data. Converting to the format needed for the chart.")
                
                # Extract head values and units
                head_col = df.columns[0]
                head_unit = head_col.split('(')[-1].split(')')[0] if '(' in head_col else "m"
                head_values = df[head_col].values
                
                # Create new DataFrame for transformed data
                transformed_df = pd.DataFrame()
                
                # Get all flow columns
                flow_cols = [col for col in df.columns if 'Flow' in col]
                
                if not flow_cols:
                    # Try to detect any other columns as flow columns
                    flow_cols = [col for col in df.columns if col != head_col]
                    
                    if not flow_cols:
                        st.warning("No flow columns detected in the CSV. Please ensure column names contain 'Flow'.")
                        return None
                
                # Process columns by model and frequency
                model_freq_dict = {}
                
                # Identify models and their frequencies from column names
                for col in flow_cols:
                    # Try to extract model name and frequency
                    parts = col.split(' ')
                    
                    # Default model name is the full column name
                    model_name = col
                    
                    # Check if the name follows a pattern like "Model-A Flow (LPM)"
                    if 'Flow' in col:
                        # Extract the part before "Flow" as the model name
                        model_name = col.split('Flow')[0].strip()
                    
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
                    # Get flow unit from first flow column or default to LPM
                    flow_unit = "LPM"
                    if '(' in flow_cols[0]:
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
                
                # Debug: Print transformed data
                st.write("Transformed Data Structure:")
                st.write(transformed_df)
                
                df = transformed_df
            
            # Basic validation
            if not any('Flow' in col for col in df.columns):
                st.warning("No Flow column detected. Please ensure your CSV has at least one Flow column.")
                return None
            
            # Automatically set chart_generated to True when a file is uploaded
            st.session_state.chart_generated = True
                
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    
    return None

def handle_manual_input(frequency_option="Both"):
    st.subheader("Manual Data Input")
    
    # Add instructions for copying from Excel
    st.info("You can copy data from Excel and paste directly into these tables. Select cells in Excel, copy (Ctrl+C), click on the starting cell in the table below, and paste (Ctrl+V).")
    
    # Allow selection of input format
    format_type = st.radio(
        "Select Input Format",
        ["Standard (Flow-Head)", "Alternative (Head-Flow)"]
    )
    
    # Initialize session state for manually input data if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    if format_type == "Standard (Flow-Head)":
        # Create a form for standard format input
        with st.form(f"manual_input_form_{st.session_state.input_reset_key}"):
            # Units selection
            col1, col2 = st.columns(2)
            with col1:
                flow_unit = st.selectbox("Flow Rate Unit", ["GPM", "LPM", "m³/h"], 
                                    index=["GPM", "LPM", "m³/h"].index(st.session_state.manual_input_data.get('flow_unit', "LPM")),
                                    key=f"flow_unit_{st.session_state.input_reset_key}")
            with col2:
                head_unit = st.selectbox("Head Unit", ["ft", "m"], 
                                    index=["ft", "m"].index(st.session_state.manual_input_data.get('head_unit', "m")),
                                    key=f"head_unit_{st.session_state.input_reset_key}")
            
            # Number of pump models
            num_models = st.number_input("Number of Pump Models", min_value=1, max_value=5, value=2,
                                    key=f"num_models_{st.session_state.input_reset_key}")
            
            # Model names
            model_names = []
            stored_model_names = st.session_state.manual_input_data.get('model_names', ["Model-A", "Model-B"])
            
            cols = st.columns(min(num_models, 5))
            for i, col in enumerate(cols):
                # Use stored model name if available, otherwise use default
                default_name = stored_model_names[i] if i < len(stored_model_names) else f"Model-{chr(65+i)}"
                model_name = col.text_input(f"Model {i+1} Name", value=default_name,
                                        key=f"model_name_{i}_{st.session_state.input_reset_key}")
                model_names.append(model_name)
            
            # Number of data points
            num_points = st.number_input(
                "Number of Data Points", 
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
            use_template = st.checkbox("Use Template Data", 
                                    value=st.session_state.manual_input_data.get('use_template', True),
                                    key=f"use_template_{st.session_state.input_reset_key}")
            
            # Template data for different frequencies
            if use_template:
                # Generate base flow values for the x-axis
                base_flow_values = np.linspace(0, 100, num_points).tolist()
                
                # Create template data for each model at 50Hz
                template_heads_50hz = {}
                for i, model in enumerate(model_names):
                    # Scale max head based on model number for variety
                    max_head = 50.0 * (1 + 0.2 * i)  # Increase by 20% for each model
                    
                    # Generate head values that decrease as flow increases
                    heads = []
                    for j, flow in enumerate(base_flow_values):
                        # Calculate head at this flow using a quadratic formula
                        # This creates a curve that goes from max_head at zero flow to zero at max_flow
                        norm_flow = flow / base_flow_values[-1]  # Normalize to 0-1
                        head = max_head * (1 - norm_flow**1.5)  # Non-linear decrease
                        head = max(0, head)  # Ensure non-negative head
                        heads.append(head)
                    
                    template_heads_50hz[model] = heads
                
                # 60Hz values (44% higher head, 20% higher flow)
                template_heads_60hz = {}
                for model in model_names:
                    template_heads_60hz[model] = [head * 1.44 for head in template_heads_50hz[model]]
            else:
                # Generate empty template data
                base_flow_values = np.linspace(0, 100, num_points).tolist()
                template_heads_50hz = {model: [0] * num_points for model in model_names}
                template_heads_60hz = {model: [0] * num_points for model in model_names}
            
            # Create tabs for each frequency
            frequency_tabs = st.tabs([f"{freq} Data" for freq in frequencies_to_show])
            
            # Dictionary to store edited data for each frequency
            edited_data = {}
            
            # Load previously stored data if available
            stored_data = st.session_state.manual_input_data.get('edited_data', {})
            
            # Create data editor for each frequency
            for i, freq in enumerate(frequencies_to_show):
                with frequency_tabs[i]:
                    st.info(f"Edit {freq} pump data below. Flow values are common across models.")
                    
                    # Try to get previously stored data for this frequency
                    df_freq = None
                    if freq in stored_data:
                        # Attempt to use stored data if available
                        try:
                            df_freq = stored_data[freq]
                            # Check if data needs to be updated for different model names or counts
                            stored_models = [col.split(' ')[0] for col in df_freq.columns if 'Head' in col]
                            if set(model_names) != set(stored_models):
                                # Models have changed, need to recreate template
                                df_freq = None
                        except:
                            df_freq = None
                    
                    if df_freq is None:
                        # Create new dataframe from templates
                        df_freq = pd.DataFrame({
                            f'Flow ({flow_unit})': base_flow_values
                        })
                        
                        # Add head columns for each model
                        for model in model_names:
                            if freq == "50Hz":
                                df_freq[f'{model} Head ({head_unit})'] = template_heads_50hz.get(model, [0] * num_points)
                            else:  # 60Hz
                                df_freq[f'{model} Head ({head_unit})'] = template_heads_60hz.get(model, [0] * num_points)
                    
                    # Create a data editor for this frequency
                    edited_df = st.data_editor(
                        df_freq, 
                        use_container_width=True,
                        num_rows="fixed",
                        height=min(500, 70 + 40*num_points),
                        key=f"data_editor_{freq}_{st.session_state.input_reset_key}"
                    )
                    
                    # Store the edited data
                    edited_data[freq] = edited_df
            
            # Submit button and Refresh Form button
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Generate Pump Curve")
            with col2:
                # Keep the Refresh Form button
                refresh_data = st.form_submit_button("Refresh Form")
            
            if submitted:
                # Save the current form state to session state
                st.session_state.manual_input_data = {
                    'flow_unit': flow_unit,
                    'head_unit': head_unit,
                    'model_names': model_names,
                    'use_template': use_template,
                    'edited_data': edited_data
                }
                
                # Transform the edited data into a standard format dataframe
                transformed_df = pd.DataFrame()
                
                # Process each frequency to create a standard format dataframe
                for freq in frequencies_to_show:
                    if freq in edited_data:
                        df_freq = edited_data[freq]
                        
                        # Get the flow column from the first frequency
                        if len(transformed_df) == 0:
                            flow_col = df_freq.columns[0]  # Assume first column is flow
                            transformed_df[flow_col] = df_freq[flow_col]
                        
                        # Add head columns for each model at this frequency
                        head_cols = [col for col in df_freq.columns if 'Head' in col]
                        for head_col in head_cols:
                            model_name = head_col.split(' ')[0]  # Extract model name
                            transformed_df[f'{model_name} {freq} {head_col.split(model_name)[1]}'] = df_freq[head_col]
                
                # Debug the transformed data
                st.write("Debug: Standard Format Data")
                st.write(transformed_df)
                
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
                
                # Increment the reset key to force form refresh
                st.session_state.input_reset_key += 1
                
                # Construct data to return
                transformed_df = pd.DataFrame()
                
                # Process each frequency to create a standard format dataframe
                for freq in frequencies_to_show:
                    if freq in edited_data:
                        df_freq = edited_data[freq]
                        
                        # Get the flow column from the first frequency
                        if len(transformed_df) == 0:
                            flow_col = df_freq.columns[0]  # Assume first column is flow
                            transformed_df[flow_col] = df_freq[flow_col]
                        
                        # Add head columns for each model at this frequency
                        head_cols = [col for col in df_freq.columns if 'Head' in col]
                        for head_col in head_cols:
                            model_name = head_col.split(' ')[0]  # Extract model name
                            transformed_df[f'{model_name} {freq} {head_col.split(model_name)[1]}'] = df_freq[head_col]
                
                if not transformed_df.empty:
                    st.session_state.current_df = transformed_df
                    st.session_state.chart_generated = True
                
                st.rerun()
    else:
        # Handle alternative format (Head-Flow)
        with st.form(f"alt_manual_input_form_{st.session_state.input_reset_key}"):
            # Units selection
            col1, col2 = st.columns(2)
            with col1:
                flow_unit = st.selectbox("Flow Rate Unit", ["GPM", "LPM", "m³/h"], 
                                    index=["GPM", "LPM", "m³/h"].index(st.session_state.manual_input_data.get('flow_unit', "LPM")),
                                    key=f"alt_flow_unit_{st.session_state.input_reset_key}")
            with col2:
                head_unit = st.selectbox("Head Unit", ["ft", "m"], 
                                    index=["ft", "m"].index(st.session_state.manual_input_data.get('head_unit', "m")),
                                    key=f"alt_head_unit_{st.session_state.input_reset_key}")
            
            # Number of pump models
            num_models = st.number_input("Number of Pump Models", min_value=1, max_value=5, value=2,
                                    key=f"alt_num_models_{st.session_state.input_reset_key}")
            
            # Model names
            model_names = []
            stored_model_names = st.session_state.manual_input_data.get('model_names', ["Model-A", "Model-B"])
            
            cols = st.columns(min(num_models, 5))
            for i, col in enumerate(cols):
                # Use stored model name if available, otherwise use default
                default_name = stored_model_names[i] if i < len(stored_model_names) else f"Model-{chr(65+i)}"
                model_name = col.text_input(f"Model {i+1} Name", value=default_name,
                                        key=f"alt_model_name_{i}_{st.session_state.input_reset_key}")
                model_names.append(model_name)
            
            # Number of data points
            num_points = st.number_input(
                "Number of Data Points", 
                min_value=3, 
                max_value=30, 
                value=st.session_state.num_data_points,
                key=f"alt_num_points_{st.session_state.input_reset_key}"
            )
            
            # Option to use template data
            use_template = st.checkbox("Use Template Data", 
                                    value=st.session_state.manual_input_data.get('use_template', True),
                                    key=f"alt_use_template_{st.session_state.input_reset_key}")
            
            # Generate template data for the alternative format
            if use_template:
                # Generate template head and flow values
                template_head = np.linspace(0, 50, num_points).tolist()
                template_flows = {}
                
                # Create realistic pump curve data for each model
                for model_idx, model in enumerate(model_names):
                    flows = []
                    for i, head in enumerate(template_head):
                        # Create a curve that decreases flow as head increases
                        # Scale max flow based on model number for variety
                        max_flow = 100 * (1 + 0.15 * model_idx)  # 15% increase per model
                        
                        # Calculate flow at this head using a quadratic formula
                        # This creates a curve that goes from max_flow at zero head to zero at max_head
                        norm_head = head / template_head[-1]  # Normalize to 0-1
                        flow = max_flow * (1 - norm_head ** 0.7)  # Non-linear decrease
                        flow = max(0, flow)  # Ensure non-negative flow
                        flows.append(flow)
                    
                    template_flows[model] = flows
            else:
                # Create empty template data
                template_head = [0] * num_points
                template_flows = {model: [0] * num_points for model in model_names}
            
            # Create dataframe for editing
            st.markdown("### Edit Pump Data Below")
            
            # Try to get previously stored data for alternate format
            alt_data = st.session_state.manual_input_data.get('alt_data', None)
            
            # Create the base dataframe
            if alt_data is not None and len(alt_data.columns) >= len(model_names) + 1:
                # Try to use stored data
                df_alt = alt_data
            else:
                # Create from template
                df_alt = pd.DataFrame({
                    f'Head ({head_unit})': template_head
                })
                
                # Add flow columns for each model
                for model in model_names:
                    df_alt[f'{model} Flow ({flow_unit})'] = template_flows.get(model, [0] * num_points)
            
            # Create data editor
            edited_alt_df = st.data_editor(
                df_alt,
                use_container_width=True,
                num_rows="fixed",
                height=min(500, 70 + 40*num_points),
                key=f"alt_data_editor_{st.session_state.input_reset_key}"
            )
            
            # Submit and Refresh buttons
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Generate Pump Curve")
            with col2:
                refresh_data = st.form_submit_button("Refresh Form")
            
            if submitted:
                # Save the current form state to session state
                st.session_state.manual_input_data = {
                    'flow_unit': flow_unit,
                    'head_unit': head_unit,
                    'model_names': model_names,
                    'use_template': use_template,
                    'alt_data': edited_alt_df
                }
                
                # Set chart generated flag
                st.session_state.chart_generated = True
                
                # Debug: Print the data structure
                st.write("Debug: Alternative Format Data")
                st.write(edited_alt_df)
                
                # Return the edited dataframe
                return edited_alt_df
            
            elif refresh_data:
                # Save the current form state to session state
                st.session_state.manual_input_data = {
                    'flow_unit': flow_unit,
                    'head_unit': head_unit,
                    'model_names': model_names,
                    'use_template': use_template,
                    'alt_data': edited_alt_df
                }
                
                # Increment the reset key to force form refresh
                st.session_state.input_reset_key += 1
                
                # Set the data and chart generated flag
                st.session_state.current_df = edited_alt_df
                st.session_state.chart_generated = True
                
                st.rerun()
    
    return None

def generate_pump_curve_head_first(df, frequency_option="Both", chart_style="Modern", show_system_curve=False, 
                                static_head=0.0, k_factor=0.0, min_flow=0.0, max_flow=None, min_head=0.0, 
                                max_head=None, show_grid=True, debug_mode=False):
    """Generate pump curves for data where head is in the first column and flow values are in separate columns"""
    if debug_mode:
        st.write(debug_plot_data(df, "Head-First Format Data"))
    
    # Create a larger figure to prevent text overlap
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Increase figure margins to make room for axes labels
    plt.subplots_adjust(bottom=0.2, right=0.9)
    
    # Identify head and flow columns
    head_col = df.columns[0]  # Assuming head is in the first column
    head_unit = "m"
    if "(" in head_col:
        head_unit = head_col.split("(")[1].split(")")[0]
    
    # Get all flow columns (all columns except the head column)
    flow_cols = [col for col in df.columns if col != head_col]
    
    # Determine flow unit from the first flow column
    flow_unit = "LPM"
    if flow_cols and "(" in flow_cols[0]:
        flow_unit = flow_cols[0].split("(")[1].split(")")[0]
    
    # Determine which models to plot
    model_names = []
    for col in flow_cols:
        # Extract model name from column name
        if "Flow" in col:
            model_name = col.split("Flow")[0].strip()
        else:
            model_name = col
        
        model_names.append(model_name)
    
    # Debug model detection
    if debug_mode:
        st.write(f"Models detected: {model_names}")
    
    # Determine which frequencies to plot based on user selection
    frequencies_to_plot = []
    if "50Hz" in frequency_option or frequency_option == "Both":
        frequencies_to_plot.append('50Hz')
    if "60Hz" in frequency_option or frequency_option == "Both":
        frequencies_to_plot.append('60Hz')
    
    # Get color cycle for plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot each pump model with distinct colors
    for i, (model_name, flow_col) in enumerate(zip(model_names, flow_cols)):
        color = colors[i % len(colors)]
        
        # Extract data for this model
        head_values = df[head_col].values
        flow_values = df[flow_col].values
        
        # Debug values
        if debug_mode:
            st.write(f"Data for {model_name}:")
            st.write(f"  Head values: {head_values}")
            st.write(f"  Flow values: {flow_values}")
        
        # Plot 50Hz curve (this is the base data)
        if '50Hz' in frequencies_to_plot:
            # Use solid line for 50Hz and higher zorder to ensure visibility
            ax.plot(flow_values, head_values, 
                    linestyle='-', linewidth=3.0, 
                    label=f"{model_name} (50Hz)", color=color,
                    zorder=10+i)  # Higher zorder means plotted on top
            
            # Debug output
            if debug_mode:
                st.write(f"Plotting 50Hz data for {model_name}:")
                st.write(f"  Flow values: {flow_values}")
                st.write(f"  Head values: {head_values}")
        
        # Plot 60Hz curve if requested (20% higher flow, 44% higher head)
        if '60Hz' in frequencies_to_plot:
            # Convert to 60Hz values
            flow_values_60hz = flow_values * 1.2
            head_values_60hz = head_values * 1.44
            
            # Use dashed line for 60Hz with even higher zorder
            ax.plot(flow_values_60hz, head_values_60hz, 
                    linestyle='--', linewidth=3.0, 
                    label=f"{model_name} (60Hz)", color=color,
                    zorder=20+i)  # Higher zorder than 50Hz curves
            
            # Debug output
            if debug_mode:
                st.write(f"Plotting 60Hz data for {model_name}:")
                st.write(f"  Flow values (60Hz): {flow_values_60hz}")
                st.write(f"  Head values (60Hz): {head_values_60hz}")
    
    # Add system curve if requested
    if show_system_curve:
        # Determine maximum flow value for system curve
        if max_flow is None:
            max_flow_val = df[flow_cols].max().max() * 1.5
        else:
            max_flow_val = max_flow
        
        # Create system curve
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve with high zorder to ensure visibility
        ax.plot(system_flows, system_heads, 'r-', linewidth=2.5, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)',
                zorder=30)  # Highest zorder to ensure visibility
        
        # Find and plot intersection points for all models and frequencies
        for i, (model_name, flow_col) in enumerate(zip(model_names, flow_cols)):
            color = colors[i % len(colors)]
            
            # Get data for this model
            head_values = df[head_col].values
            flow_values = df[flow_col].values
            
            # For 50Hz (base data)
            if '50Hz' in frequencies_to_plot:
                # Find intersection point using interpolation
                try:
                    # Create interpolation functions
                    from scipy.interpolate import interp1d
                    
                    # Create pump curve function (head as function of flow)
                    # Sort data points by flow for interpolation
                    sorted_indices = np.argsort(flow_values)
                    sorted_flows = flow_values[sorted_indices]
                    sorted_heads = head_values[sorted_indices]
                    
                    # Check if we have enough unique points for interpolation
                    if len(np.unique(sorted_flows)) >= 2:
                        pump_curve_func = interp1d(sorted_flows, sorted_heads, 
                                                bounds_error=False, fill_value="extrapolate")
                        
                        # Find intersection with system curve
                        # Create a function for the difference between curves
                        diff_func = lambda q: pump_curve_func(q) - (static_head + k_factor * q**2)
                        
                        # Try to find the intersection
                        from scipy.optimize import fsolve, minimize_scalar
                        
                        # Try to find intersection point within the flow range
                        try:
                            # Use minimize_scalar to find the minimum absolute difference
                            result = minimize_scalar(lambda q: abs(diff_func(q)), 
                                                    bounds=(min(sorted_flows), max(sorted_flows)), 
                                                    method='bounded')
                            
                            if result.success:
                                op_flow = result.x
                                op_head = static_head + k_factor * op_flow**2
                                
                                # Check if the point is within the data range
                                if min(sorted_flows) <= op_flow <= max(sorted_flows):
                                    # Plot operating point
                                    ax.plot(op_flow, op_head, 'o', markersize=8, color=color, zorder=40)
                                    
                                    # Add annotation
                                    ax.annotate(f"{model_name} (50Hz): ({op_flow:.1f}, {op_head:.1f})",
                                            xy=(op_flow, op_head),
                                            xytext=(10, 5),
                                            textcoords='offset points',
                                            color=color,
                                            fontweight='bold',
                                            zorder=45)
                        except Exception as e:
                            if debug_mode:
                                st.write(f"Error finding intersection for {model_name} at 50Hz: {e}")
                except Exception as e:
                    if debug_mode:
                        st.write(f"Error with interpolation for {model_name} at 50Hz: {e}")
            
            # For 60Hz
            if '60Hz' in frequencies_to_plot:
                # Convert to 60Hz values
                flow_values_60hz = flow_values * 1.2
                head_values_60hz = head_values * 1.44
                
                # Find intersection point using interpolation
                try:
                    # Create interpolation functions
                    from scipy.interpolate import interp1d
                    
                    # Create pump curve function (head as function of flow)
                    # Sort data points by flow for interpolation
                    sorted_indices = np.argsort(flow_values_60hz)
                    sorted_flows = flow_values_60hz[sorted_indices]
                    sorted_heads = head_values_60hz[sorted_indices]
                    
                    # Check if we have enough unique points for interpolation
                    if len(np.unique(sorted_flows)) >= 2:
                        pump_curve_func = interp1d(sorted_flows, sorted_heads, 
                                                bounds_error=False, fill_value="extrapolate")
                        
                        # Find intersection with system curve
                        # Create a function for the difference between curves
                        diff_func = lambda q: pump_curve_func(q) - (static_head + k_factor * q**2)
                        
                        # Try to find the intersection
                        from scipy.optimize import fsolve, minimize_scalar
                        
                        # Try to find intersection point within the flow range
                        try:
                            # Use minimize_scalar to find the minimum absolute difference
                            result = minimize_scalar(lambda q: abs(diff_func(q)), 
                                                    bounds=(min(sorted_flows), max(sorted_flows)), 
                                                    method='bounded')
                            
                            if result.success:
                                op_flow = result.x
                                op_head = static_head + k_factor * op_flow**2
                                
                                # Check if the point is within the data range
                                if min(sorted_flows) <= op_flow <= max(sorted_flows):
                                    # Plot operating point
                                    ax.plot(op_flow, op_head, 's', markersize=8, color=color, zorder=40)
                                    
                                    # Add annotation
                                    ax.annotate(f"{model_name} (60Hz): ({op_flow:.1f}, {op_head:.1f})",
                                            xy=(op_flow, op_head),
                                            xytext=(10, -15),
                                            textcoords='offset points',
                                            color=color,
                                            fontweight='bold',
                                            zorder=45)
                        except Exception as e:
                            if debug_mode:
                                st.write(f"Error finding intersection for {model_name} at 60Hz: {e}")
                except Exception as e:
                    if debug_mode:
                        st.write(f"Error with interpolation for {model_name} at 60Hz: {e}")
    
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
    
    # Get maximum values for each column for axis scaling
    max_flow_data = df[flow_cols].max().max()
    max_head_data = df[head_col].max()
    
    # Print debug info for axis limits
    if debug_mode:
        st.write(f"Max flow in data: {max_flow_data}")
        st.write(f"Max head in data: {max_head_data}")
    
    # Apply custom axis limits if provided
    # Set x-axis limits with additional padding to ensure all data is visible
    x_padding = max_flow_data * 0.1  # 10% padding
    if max_flow is None:
        if '60Hz' in frequencies_to_plot:
            # Add extra space for 60Hz curves (20% higher flow)
            max_flow_to_use = max_flow_data * 1.2 * 1.1  # 20% for 60Hz + 10% padding
        else:
            max_flow_to_use = max_flow_data * 1.1  # Just 10% padding
    else:
        max_flow_to_use = float(max_flow)
    
    ax.set_xlim(left=float(min_flow), right=max_flow_to_use)
    
    # Set y-axis limits with padding
    y_padding = max_head_data * 0.1  # 10% padding
    if max_head is None:
        if '60Hz' in frequencies_to_plot:
            # Add extra space for 60Hz curves (44% higher head)
            max_head_to_use = max_head_data * 1.44 * 1.1  # 44% for 60Hz + 10% padding
        else:
            max_head_to_use = max_head_data * 1.1  # Just 10% padding
    else:
        max_head_to_use = float(max_head)
    
    ax.set_ylim(bottom=float(min_head), top=max_head_to_use)
    
    # Print actual axis limits for debugging
    if debug_mode:
        st.write(f"X-axis limits: {ax.get_xlim()}")
        st.write(f"Y-axis limits: {ax.get_ylim()}")
    
    # Add secondary x-axis for alternative flow units
    if flow_unit == "LPM":
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis
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
        ax_ft.set_ylabel(f'Head (ft)', fontsize, 12, fontweight='bold', labelpad=15)
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

def generate_pump_curve(df, frequency_option="Both", chart_style="Modern", show_system_curve=False, 
                       static_head=0.0, k_factor=0.0, refresh_counter=0, min_flow=0.0, max_flow=None,
                       min_head=0.0, max_head=None, show_grid=True, debug_mode=False):
    """Generate pump curves for data in the standard format (flow in first column, head in other columns)"""
    if debug_mode:
        st.write(debug_plot_data(df, "Standard Format Data"))
    
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
    flow_unit = flow_col.split('(')[-1].split(')')[0] if '(' in flow_col else "LPM"
    
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
    
    # Print debug info about frequencies
    if debug_mode:
        st.write(f"Frequencies present in data: {frequencies_present}")
        st.write(f"Frequencies selected to plot: {frequencies_to_plot}")
    
    # Get all head columns
    head_cols = [col for col in df.columns if 'Head' in col]
    
    # If no head columns found, check if this might be alternate format data
    if not head_cols and len(df.columns) > 1:
        # It might be Head-Flow format instead of Flow-Head
        st.warning("Data might be in Head-Flow format instead of Flow-Head format. Please use the appropriate function.")
        return generate_pump_curve_head_first(df, frequency_option, chart_style, show_system_curve, 
                                           static_head, k_factor, min_flow, max_flow, min_head, max_head, 
                                           show_grid, debug_mode)
    
    # Group columns by model and frequency
    model_data = {}
    
    # Debug head columns
    if debug_mode:
        st.write(f"Head columns found: {head_cols}")
    
    for col in head_cols:
        parts = col.split(' ')
        
        # Extract model name (everything before the frequency or "Head")
        model_name = None
        for i, part in enumerate(parts):
            if '50Hz' in part or '60Hz' in part or 'Head' in part:
                model_name = ' '.join(parts[:i])
                break
        
        # If still no model name, use first part as model name
        if not model_name:
            model_name = parts[0]
        
        # Find frequency in the column name
        freq = None
        for part in parts:
            if '50Hz' in part:
                freq = '50Hz'
                break
            elif '60Hz' in part:
                freq = '60Hz'
                break
        
        # If no frequency found, assume 50Hz
        if freq is None:
            freq = '50Hz'
            
        # Skip if frequency not selected to plot
        if freq not in frequencies_to_plot:
            continue
            
        # Get column unit
        head_unit = col.split('(')[-1].split(')')[0] if '(' in col else "m"
        
        # Create model entry if doesn't exist
        if model_name not in model_data:
            model_data[model_name] = {}
            
        # Store data for this model and frequency
        model_data[model_name][freq] = df[col].values
        
        # Debug model data
        if debug_mode:
            st.write(f"Processing column: {col}")
            st.write(f"  Model name: {model_name}")
            st.write(f"  Frequency: {freq}")
            st.write(f"  Head unit: {head_unit}")
            st.write(f"  Data sample: {df[col].head().tolist()}")
    
    # Get color cycle for plots - use different colors for different models
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Debug model data structure
    if debug_mode:
        st.write("Model data structure:")
        for model_name, freq_data in model_data.items():
            st.write(f"Model: {model_name}")
            for freq, data in freq_data.items():
                st.write(f"  {freq}: min={min(data)}, max={max(data)}, len={len(data)}")
    
    # Plot each pump model with distinct colors, with different line styles for frequencies
    model_names = list(model_data.keys())
    for i, model_name in enumerate(model_names):
        color = colors[i % len(colors)]
        
        # Plot different frequencies with different line styles and ensure visibility with zorder
        for j, freq in enumerate(frequencies_to_plot):
            if freq in model_data[model_name]:
                # Set line style based on frequency
                line_style = '-' if freq == '50Hz' else '--'
                
                # Plot curve for this model and frequency
                # Higher zorder ensures newer lines appear on top
                # Use thicker linewidth for better visibility
                zorder_value = 10 + i*2 + (0 if freq == '50Hz' else 1)
                ax.plot(df[flow_col], model_data[model_name][freq], 
                        linestyle=line_style, linewidth=3.0, 
                        label=f"{model_name} ({freq})", color=color,
                        zorder=zorder_value)
                
                # Debug plot info
                if debug_mode:
                    st.write(f"Plotting {model_name} at {freq} with color {color}, zorder {zorder_value}")
    
    # Add system curve if requested
    if show_system_curve:
        max_flow_val = df[flow_col].max() * 1.5 if max_flow is None else max_flow
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve with highest zorder to ensure visibility
        ax.plot(system_flows, system_heads, 'r-', linewidth=2.5, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)',
                zorder=30)  # Highest zorder to ensure visibility
        
        # Find and plot intersection points for all models and frequencies
        for i, model_name in enumerate(model_names):
            color = colors[i % len(colors)]
            
            for freq in frequencies_to_plot:
                if freq in model_data[model_name]:
                    # Get pump curve data
                    pump_heads = model_data[model_name][freq]
                    
                    # Interpolate pump curve for intersection
                    try:
                        pump_heads_interp = np.interp(system_flows, df[flow_col], pump_heads, 
                                                     left=np.nan, right=np.nan)
                        
                        # Find intersection point
                        diff = np.abs(pump_heads_interp - system_heads)
                        valid_idx = ~np.isnan(diff)
                        if np.any(valid_idx):
                            op_idx = np.argmin(diff[valid_idx])
                            op_flow = system_flows[valid_idx][op_idx]
                            op_head = pump_heads_interp[valid_idx][op_idx]
                            
                            # Debug intersection point
                            if debug_mode:
                                st.write(f"Found intersection for {model_name} at {freq}:")
                                st.write(f"  Flow: {op_flow:.2f}, Head: {op_head:.2f}")
                            
                            # Set marker style based on frequency
                            marker_style = 'o' if freq == '50Hz' else 's'
                            
                            # Plot operating point with high zorder
                            ax.plot(op_flow, op_head, marker_style, markersize=8, color=color, zorder=40)
                            
                            # Add operating point annotation
                            ax.annotate(f"{model_name} ({freq}): ({op_flow:.1f}, {op_head:.1f})",
                                       xy=(op_flow, op_head),
                                       xytext=(10, (5 if freq == '50Hz' else -15)),
                                       textcoords='offset points',
                                       color=color,
                                       fontweight='bold',
                                       zorder=45)
                    except Exception as e:
                        if debug_mode:
                            st.write(f"Error finding intersection for {model_name} at {freq}: {e}")
    
    # Set up the primary x and y axes
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12, fontweight='bold')
    
    # Use the head unit from the first head column, if available
    if head_cols:
        head_unit = head_cols[0].split('(')[-1].split(')')[0] if '(' in head_cols[0] else "m"
        ax.set_ylabel(f'Head ({head_unit})', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('Head (m)', fontsize=12, fontweight='bold')
    
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
    
    # Calculate maximum values for axes limits
    max_flow_data = df[flow_col].max()
    
    # Find maximum head across all models and frequencies
    max_head_val = 0
    for model_name in model_data:
        for freq in model_data[model_name]:
            max_head_val = max(max_head_val, np.max(model_data[model_name][freq]))
    
    # Debug axis range values
    if debug_mode:
        st.write(f"Max flow in data: {max_flow_data}")
        st.write(f"Max head in data: {max_head_val}")
    
    # Apply custom axis limits if provided
    # Add extra padding for better visualization
    x_padding = max_flow_data * 0.1  # 10% padding
    if max_flow is None:
        if '60Hz' in frequencies_to_plot:
            # Add extra space for 60Hz curves (20% higher flow)
            max_flow_to_use = max_flow_data * 1.2 * 1.1  # 20% for 60Hz + 10% padding
        else:
            max_flow_to_use = max_flow_data * 1.1  # Just 10% padding
    else:
        max_flow_to_use = float(max_flow)
        
    ax.set_xlim(left=float(min_flow), right=max_flow_to_use)
    
    # Set y-axis limits with padding
    y_padding = max_head_val * 0.1  # 10% padding
    if max_head is None:
        if '60Hz' in frequencies_to_plot:
            # Add extra space for 60Hz curves (44% higher head)
            max_head_to_use = max_head_val * 1.44 * 1.1  # 44% for 60Hz + 10% padding
        else:
            max_head_to_use = max_head_val * 1.1  # Just 10% padding
    else:
        max_head_to_use = float(max_head)
        
    ax.set_ylim(bottom=float(min_head), top=max_head_to_use)
    
    # Debug actual axis limits
    if debug_mode:
        st.write(f"X-axis limits set to: {ax.get_xlim()}")
        st.write(f"Y-axis limits set to: {ax.get_ylim()}")
    
    # Add secondary x-axis for alternative flow units
    if flow_unit == "LPM":
        # Add m³/h axis
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis
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
    if 'head_unit' in locals() and head_unit == "m":
        # Position the ft axis
        ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
        ax_ft.yaxis.set_major_locator(MaxNLocator(7))
        ax_ft.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax_ft.set_ylabel(f'Head (ft)', fontsize=12, fontweight='bold', labelpad=15)
    elif 'head_unit' in locals() and head_unit == "ft":
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

def download_button_for_plot(fig):
    """Create a download button for saving the plot as PNG"""
    # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Create download button
    btn = st.download_button(
        label="Download Plot (PNG)",
        data=buf,
        file_name="pump_curve_plot.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main()
