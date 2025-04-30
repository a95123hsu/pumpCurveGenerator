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
            'frequency': 50,
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
            frequency = st.selectbox(
                "Frequency (Hz)", 
                [50, 60], 
                index=[50, 60].index(st.session_state.chart_params['frequency']),
                key="frequency_select",
                on_change=lambda: (
                    setattr(
                        st.session_state, 'chart_params', 
                        {**st.session_state.chart_params, 'frequency': st.session_state.frequency_select}
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
        
        # Sidebar for input method selection
        input_method = st.radio(
            "Select Input Method",
            ["Upload CSV", "Manual Input"]
        )
        
        if input_method == "Upload CSV":
            df = handle_csv_upload()
        else:
            df = handle_manual_input()
        
        # Generate and display the pump curve if data is available
        if df is not None and not df.empty:
            st.session_state.current_df = df  # Store the current dataframe
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # For CSV uploads, automatically set chart_generated to True
                if input_method == "Upload CSV" and not st.session_state.chart_generated:
                    st.session_state.chart_generated = True
                
                # Generate curve using parameters from session state
                if st.session_state.chart_generated:
                    params = st.session_state.chart_params
                    try:
                        fig = generate_pump_curve(
                            df, 
                            params['frequency'], 
                            params['chart_style'], 
                            params['show_system_curve'], 
                            params['static_head'], 
                            params['k_factor'],
                            min_flow=params['min_flow'],
                            max_flow=params['max_flow'],
                            min_head=params['min_head'],
                            max_head=params['max_head'],
                            show_grid=params['show_grid']
                        )
                        st.pyplot(fig)
                        
                        # Add download button for the plot
                        download_button_for_plot(fig)
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
                else:
                    st.info("Click Generate Chart to create the pump curve.")
            
            with col2:
                st.subheader("Pump Curve Data")
                st.dataframe(df)
                
                # Add download button for the data
                download_button_for_data(df)
    
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
        
        ### Selecting the Right Pump
        
        When selecting a pump, consider:
        1. Required flow rate
        2. Required head
        3. System efficiency
        4. NPSH (Net Positive Suction Head)
        5. Power consumption
        """)

def handle_csv_upload():
    st.subheader("Upload CSV File")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Sample CSV template for download - Using head values as primary axis
    sample_data = pd.DataFrame({
        'Head (ft)': [4.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 49.8],
        'Model-A Flow (GPM)': [90.09, 89.96, 84.81, 77.43, 69.85, 61.89, 53.93, 41.81, 29.34, 18.52, 8.45],
        'Model-B Flow (GPM)': [105.50, 104.75, 95.22, 86.34, 78.43, 69.45, 60.76, 48.45, 35.21, 21.93, 11.24]
    })
    
    st.markdown("### Download Sample CSV Template")
    
    # Create template
    csv_standard = sample_data.to_csv(index=False)
    b64_standard = base64.b64encode(csv_standard.encode()).decode()
    href_standard = f'<a href="data:file/csv;base64,{b64_standard}" download="pump_curve_template.csv">Download CSV Template</a>'
    
    st.markdown(href_standard, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Transform the data into format needed for plotting (Head vs Flow)
            # Need to transpose it for the traditional pump curve format expected by the plotting function
            
            # Check if data is in Head-first, Flow-second format
            if 'Head' in df.columns[0]:
                st.info("Converting Head-Flow format to standard format for pump curve generation.")
                
                # Extract head values and units
                head_col = df.columns[0]
                head_unit = head_col.split('(')[-1].split(')')[0]
                head_values = df[head_col].values
                
                # Create new DataFrame in the format needed for plotting
                transformed_df = pd.DataFrame()
                
                # Find all flow columns and add them to the transformed DF
                flow_cols = [col for col in df.columns if 'Flow' in col]
                if not flow_cols:
                    st.warning("No flow columns detected. Please ensure your CSV has column names like 'Model-X Flow (GPM)'")
                    return None
                
                # Extract flow unit from first flow column
                flow_unit = flow_cols[0].split('(')[-1].split(')')[0]
                
                # For each model's flow, create a new column with the head values
                for flow_col in flow_cols:
                    model_name = flow_col.split(' ')[0]  # Extract model name
                    flow_values = df[flow_col].values
                    
                    # Add to new dataframe in the format needed for plotting
                    if len(transformed_df) == 0:
                        # First model, initialize with flow column
                        transformed_df[f'Flow ({flow_unit})'] = flow_values
                    
                    # Add head column for this model
                    transformed_df[f'{model_name} Head ({head_unit})'] = head_values
                
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

def handle_manual_input():
    st.subheader("Manual Data Input")
    
    # Initialize session state for manually input data if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    # Add a reset button outside the form
    if st.button("Reset Input Form", key=f"reset_button_{st.session_state.input_reset_key}"):
        st.session_state.input_reset_key += 1  # Incrementing forces form to re-render
        st.rerun()  # Force a rerun of the app
    
    # Create a form for manual input
    with st.form(f"manual_input_form_{st.session_state.input_reset_key}"):
        # Units selection
        col1, col2 = st.columns(2)
        with col1:
            flow_unit = st.selectbox("Flow Rate Unit", ["GPM", "LPM", "m³/h"], 
                                   key=f"flow_unit_{st.session_state.input_reset_key}")
        with col2:
            head_unit = st.selectbox("Head Unit", ["ft", "m"], 
                                   key=f"head_unit_{st.session_state.input_reset_key}")
        
        # Number of pump models
        num_models = st.number_input("Number of Pump Models", min_value=1, max_value=5, value=2,
                                   key=f"num_models_{st.session_state.input_reset_key}")
        
        # Model names
        model_names = []
        cols = st.columns(min(num_models, 5))
        for i, col in enumerate(cols):
            model_name = col.text_input(f"Model {i+1} Name", value=f"Model-{chr(65+i)}",
                                      key=f"model_name_{i}_{st.session_state.input_reset_key}")
            model_names.append(model_name)
        
        # Option to use template data
        use_template = st.checkbox("Use Template Data", value=True,
                                 key=f"use_template_{st.session_state.input_reset_key}")
        
        if use_template:
            # Use the data provided by the user - common head values for all models
            num_points = 11
            template_head = [4.8, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 49.8]
            
            # Flow values for the first model
            template_flow_model_A = [90.09, 89.96, 84.81, 77.43, 69.85, 61.89, 53.93, 41.81, 29.34, 18.52, 8.45]
            
            # Pre-calculate flow values for other models (increased capacity)
            template_flow_models = {}
            for i, model in enumerate(model_names):
                if i == 0:
                    template_flow_models[model] = template_flow_model_A
                else:
                    # Each subsequent model has slightly higher flow at same head points
                    multiplier = 1.0 + (0.15 * i)  # 15% increase for each model
                    template_flow_models[model] = [flow * multiplier for flow in template_flow_model_A]
        else:
            # Number of data points
            num_points = st.number_input("Number of Data Points", min_value=3, max_value=20, value=8,
                                      key=f"num_points_{st.session_state.input_reset_key}")
            
            # Generate default head values (common for all models)
            template_head = np.linspace(5.0, 50.0, num_points).tolist()
            
            # Generate flow values for each model
            template_flow_models = {}
            base_flow = 90.0  # Starting flow for first model
            
            for i, model in enumerate(model_names):
                # Generate decreasing flow as head increases
                max_flow = base_flow * (1.0 + 0.15 * i)  # Increase capacity for each model
                min_flow = max_flow * 0.1  # End at 10% of max flow
                flows = np.linspace(max_flow, min_flow, num_points).tolist()
                template_flow_models[model] = flows
        
        # Transform the data into standard "Flow vs. Head" format for plotting
        columns = [f'Flow ({flow_unit})']
        for name in model_names:
            columns.append(f"{name} Head ({head_unit})")
        
        # Start with empty dataframe
        transformed_data = pd.DataFrame()
        
        # For each model, create a separate row in the transformed dataframe
        for i, model in enumerate(model_names):
            # Create flow column for this model
            if i == 0:
                # First model sets up the flow column
                transformed_data[columns[0]] = template_flow_models[model]
            
            # All models share the same head values
            transformed_data[f"{model} Head ({head_unit})"] = template_head
        
        # Create an editable table with a unique key
        st.markdown("### Edit Pump Data Below")
        st.info("Note: Common head values across all models. Adjust flow values for each model at the same head point.")
        
        # Display the original head-flow data for editing
        head_flow_df = pd.DataFrame({
            f'Head ({head_unit})': template_head
        })
        
        # Add flow columns for each model
        for model in model_names:
            head_flow_df[f'{model} Flow ({flow_unit})'] = template_flow_models[model]
        
        edited_df = st.data_editor(head_flow_df, use_container_width=True, 
                                  num_rows="fixed", 
                                  height=min(500, 70 + 40*num_points),
                                  key=f"data_editor_{st.session_state.input_reset_key}")
        
        # Submit button
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Generate Pump Curve")
        with col2:
            # Add a refresh button inside the form
            refresh_data = st.form_submit_button("Refresh Form")
        
        if submitted:
            # Transform the edited data back into the format needed for plotting
            transformed_df = pd.DataFrame()
            
            # Extract data from the edited dataframe
            head_col = edited_df.columns[0]
            head_values = edited_df[head_col].values
            
            # Get flow columns
            flow_cols = [col for col in edited_df.columns if 'Flow' in col]
            
            # For each model, create a row in the transformed dataframe
            for i, flow_col in enumerate(flow_cols):
                model_name = flow_col.split(' ')[0]  # Extract model name
                flow_values = edited_df[flow_col].values
                
                # Add to new dataframe in the format needed for plotting
                if i == 0:
                    # First model, initialize with flow column
                    transformed_df[f'Flow ({flow_unit})'] = flow_values
                
                # Add head column for this model
                transformed_df[f'{model_name} Head ({head_unit})'] = head_values
            
            # Automatically generate the chart when data is submitted
            st.session_state.chart_generated = True
            return transformed_df
        elif refresh_data:
            # Just return the current data to update the form
            return None
    
    return None

def generate_pump_curve(df, frequency=50, chart_style="Modern", show_system_curve=False, 
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
    
    head_cols = df.columns[1:]
    head_unit = head_cols[0].split('(')[-1].split(')')[0]
    
    # Get color cycle for plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot each pump curve with distinct colors and line styles
    for i, column in enumerate(head_cols):
        model_name = column.split(' ')[0]
        color = colors[i % len(colors)]
        ax.plot(df[flow_col], df[column], linewidth=2.5, label=model_name, color=color)
        
        # Add model name label at the end of each curve
        last_idx = df[column].dropna().last_valid_index()
        if last_idx is not None:
            x_pos = df[flow_col].iloc[last_idx]
            y_pos = df[column].iloc[last_idx]
            # Add some padding to position the text
            x_padding = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
            ax.annotate(model_name, xy=(x_pos + x_padding, y_pos), 
                        color=color, fontweight='bold', va='center')
    
    # Add system curve if requested
    if show_system_curve:
        max_flow_val = df[flow_col].max() * 1.2 if max_flow is None else max_flow
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve with dashed line
        ax.plot(system_flows, system_heads, 'r--', linewidth=2, 
                label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)')
        
        # Find and plot intersection points
        for column in head_cols:
            model_name = column.split(' ')[0]
            # Interpolate pump curve
            pump_heads = np.interp(system_flows, df[flow_col], df[column], left=np.nan, right=np.nan)
            # Calculate difference
            diff = np.abs(pump_heads - system_heads)
            valid_idx = ~np.isnan(diff)
            if np.any(valid_idx):
                op_idx = np.argmin(diff[valid_idx])
                op_flow = system_flows[valid_idx][op_idx]
                op_head = pump_heads[valid_idx][op_idx]
                
                # Plot operating point
                ax.plot(op_flow, op_head, 'o', markersize=8, 
                        color=colors[list(head_cols).index(column) % len(colors)])
                
                # Add operating point annotation
                ax.annotate(f"({op_flow:.1f}, {op_head:.1f})",
                           xy=(op_flow, op_head),
                           xytext=(10, 10),
                           textcoords='offset points',
                           color=colors[list(head_cols).index(column) % len(colors)],
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
    # Set x-axis limits
    if max_flow is None:
        ax.set_xlim(left=float(min_flow))
    else:
        ax.set_xlim(left=float(min_flow), right=float(max_flow))
    
    # Set y-axis limits
    if max_head is None:
        ax.set_ylim(bottom=float(min_head))
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
    
    # Add frequency information
    plt.text(0.05, 0.95, f"{frequency}Hz", 
             transform=ax.transAxes, 
             fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add legend with model names
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
    plt.title('Pump Performance Curves', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def download_button_for_plot(fig):
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

def download_button_for_data(df):
    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    btn = st.download_button(
        label="Download Data (CSV)",
        data=csv,
        file_name="pump_curve_data.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
