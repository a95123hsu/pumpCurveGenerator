import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def main():
    st.set_page_config(page_title="Pump Curve Generator", layout="wide")
    
    st.title("Pump Curve Generator Tool")
    
    # Initialize session state for tracking changes
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
        st.session_state.last_config = {}
    
    # Initialize input reset key if it doesn't exist
    if 'input_reset_key' not in st.session_state:
        st.session_state.input_reset_key = 0
    
    st.markdown("""
    This tool allows you to generate pump performance curves similar to manufacturer specifications.
    You can either upload a CSV file with your pump data or manually input the data points.
    """)
    
    tab1, tab2 = st.tabs(["Create Pump Curves", "About Pump Curves"])
    
    with tab1:
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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Configuration options
                st.subheader("Chart Configuration")
                
                # Use container to avoid rerun when changing these options
                with st.container():
                    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
                    with col_a:
                        frequency = st.selectbox("Frequency (Hz)", [50, 60], index=0, key="frequency_select")
                    with col_b:
                        chart_style = st.selectbox("Chart Style", ["Modern", "Classic"], index=0, key="chart_style_select")
                    with col_c:
                        show_system_curve = st.checkbox("Show System Curve", value=False, key="system_curve_checkbox")
                    with col_d:
                        refresh_button = st.button("Refresh Chart", key="refresh_chart_button")
                
                # System curve parameters (only shown if show_system_curve is True)
                if show_system_curve:
                    with st.container():
                        col_d, col_e = st.columns(2)
                        with col_d:
                            static_head = st.number_input("Static Head (m)", min_value=0.0, value=2.0, step=0.5, key="static_head_input")
                        with col_e:
                            k_factor = st.number_input("Friction Factor (k)", min_value=0.00001, value=0.0001, 
                                                    format="%.6f", step=0.00001, key="k_factor_input")
                else:
                    static_head = 0
                    k_factor = 0
                
                # Only regenerate chart when refresh button is clicked
                if 'chart_params' not in st.session_state:
                    st.session_state.chart_params = {
                        'frequency': frequency,
                        'chart_style': chart_style,
                        'show_system_curve': show_system_curve,
                        'static_head': static_head,
                        'k_factor': k_factor,
                    }
                
                if refresh_button:
                    # Update chart parameters
                    st.session_state.chart_params = {
                        'frequency': frequency,
                        'chart_style': chart_style,
                        'show_system_curve': show_system_curve,
                        'static_head': static_head,
                        'k_factor': k_factor,
                    }
                
                # Generate curve with parameters from session state
                params = st.session_state.chart_params
                fig = generate_pump_curve(
                    df, 
                    params['frequency'], 
                    params['chart_style'], 
                    params['show_system_curve'], 
                    params['static_head'], 
                    params['k_factor']
                )
                st.pyplot(fig)
                
                # Add download button for the plot
                download_button_for_plot(fig)
            
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
    
    # Sample CSV template for download
    sample_data = pd.DataFrame({
        'Flow (LPM)': [0, 100, 200, 300, 400, 500, 600, 700, 800],
        'DS-05 Head (m)': [7.5, 7.3, 7.0, 6.5, 5.8, 5.0, 4.0, 2.5, 0.8],
        'DS-10 Head (m)': [8.5, 8.3, 8.0, 7.5, 7.0, 6.5, 5.5, 4.2, 2.5],
        'DS-20 Head (m)': [9.3, 9.1, 8.8, 8.5, 8.0, 7.5, 7.0, 6.0, 4.5],
        'DS-30 Head (m)': [9.8, 9.6, 9.4, 9.0, 8.5, 8.0, 7.5, 6.5, 5.0]
    })
    
    st.markdown("### Download Sample CSV Template")
    
    csv = sample_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="pump_curve_template.csv">Download Sample Template</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            if 'Flow' not in df.columns[0]:
                st.warning("First column should contain flow values (LPM, m³/h, or GPM)")
                
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
            flow_unit = st.selectbox("Flow Rate Unit", ["LPM", "m³/h", "GPM"], 
                                   key=f"flow_unit_{st.session_state.input_reset_key}")
        with col2:
            head_unit = st.selectbox("Head Unit", ["m", "ft"], 
                                   key=f"head_unit_{st.session_state.input_reset_key}")
        
        # Number of pump models
        num_models = st.number_input("Number of Pump Models", min_value=1, max_value=5, value=1,
                                   key=f"num_models_{st.session_state.input_reset_key}")
        
        # Model names
        model_names = []
        cols = st.columns(min(num_models, 5))
        for i, col in enumerate(cols):
            model_name = col.text_input(f"Model {i+1} Name", value=f"DS-{(i+1)*10}",
                                      key=f"model_name_{i}_{st.session_state.input_reset_key}")
            model_names.append(model_name)
        
        # Number of data points
        num_points = st.number_input("Number of Data Points", min_value=3, max_value=20, value=8,
                                   key=f"num_points_{st.session_state.input_reset_key}")
        
        # Create an empty dataframe for input
        columns = [f'Flow ({flow_unit})']
        for name in model_names:
            columns.append(f"{name} Head ({head_unit})")
        
        # Create default data
        data = {}
        data[columns[0]] = [i*100 for i in range(num_points)]
        
        for col in columns[1:]:
            # Create descending values for the head
            max_head = 10.0
            min_head = 4.0
            step = (max_head - min_head) / (num_points - 1)
            data[col] = [max_head - i*step for i in range(num_points)]
        
        df = pd.DataFrame(data)
        
        # Create an editable table with a unique key
        edited_df = st.data_editor(df, use_container_width=True, 
                                  num_rows="fixed", 
                                  height=min(400, 50 + 35*num_points),
                                  key=f"data_editor_{st.session_state.input_reset_key}")
        
        # Submit button
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Generate Curve")
        with col2:
            # Add a refresh button inside the form
            refresh_data = st.form_submit_button("Refresh Form")
        
        if submitted:
            return edited_df
        elif refresh_data:
            # Just return the current data to update the form
            return edited_df
    
    return None

def generate_pump_curve(df, frequency=50, chart_style="Modern", show_system_curve=False, 
                       static_head=0, k_factor=0, refresh_counter=0):
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
        max_flow = df[flow_col].max() * 1.2
        system_flows = np.linspace(0, max_flow, 100)
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
    
    # Set up the primary x and y axes
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Head ({head_unit})', fontsize=12, fontweight='bold')
    
    # Limit number of ticks to prevent overcrowding
    ax.xaxis.set_major_locator(MaxNLocator(7))
    ax.yaxis.set_major_locator(MaxNLocator(7))
    
    # Format tick labels to 1 decimal place
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    if chart_style == "Modern":
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Make sure axes start from 0
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    # Add secondary x-axis for alternative flow units if primary is LPM
    if flow_unit == "LPM":
        # Add m³/h axis at bottom with less spacing (closer to main axis)
        ax_m3h = ax.secondary_xaxis(-0.15, functions=(lambda x: x/60, lambda x: x*60))
        # Limit ticks to prevent overcrowding
        ax_m3h.xaxis.set_major_locator(MaxNLocator(7))
        # Format to 1 decimal place
        ax_m3h.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add GPM axis below with adjusted spacing
        ax_gpm = ax.secondary_xaxis(-0.30, functions=(lambda x: x*0.264172, lambda x: x/0.264172))
        # Limit ticks and format
        ax_gpm.xaxis.set_major_locator(MaxNLocator(7))
        ax_gpm.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_gpm.set_xlabel(f'Flow (GPM)', fontsize=12, fontweight='bold', labelpad=10)
    
    # Add secondary y-axis for alternative head units if primary is m
    if head_unit == "m":
        # Position the ft axis closer to the chart (1.05 instead of 1.18)
        ax_ft = ax.secondary_yaxis(1.05, functions=(lambda x: x*3.28084, lambda x: x/3.28084))
        # Limit ticks and format
        ax_ft.yaxis.set_major_locator(MaxNLocator(7))
        ax_ft.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        # Position the label farther from the axis
        ax_ft.set_ylabel(f'Head (ft)', fontsize=12, fontweight='bold', labelpad=15)
    
    # Add frequency information
    plt.text(0.05, 0.95, f"{frequency}Hz", 
             transform=ax.transAxes, 
             fontsize=14, 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Remove the legend since we're adding labels directly to the curves
    # ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    
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
