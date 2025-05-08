import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

def main():
    st.set_page_config(page_title="Pump Curve Plotter", layout="wide")
    
    st.title("Pump Curve Plotter")
    st.markdown("""
    This tool allows you to input head and flow values for different pump models and generates performance curves.
    Simply enter your data in the tables below and click 'Generate Curve'.
    """)
    
    # Initialize session state
    if 'chart_generated' not in st.session_state:
        st.session_state.chart_generated = False
    
    if 'num_models' not in st.session_state:
        st.session_state.num_models = 2
        
    if 'num_points' not in st.session_state:
        st.session_state.num_points = 5
        
    if 'model_data' not in st.session_state:
        st.session_state.model_data = {}
        
    if 'chart_params' not in st.session_state:
        st.session_state.chart_params = {
            'frequency_option': "50Hz Only",
            'chart_style': "Modern",
            'show_system_curve': False,
            'static_head': 2.0,
            'k_factor': 0.0001,
            'max_flow': None,
            'max_head': None,
            'min_flow': 0.0,
            'min_head': 0.0,
            'show_grid': True,
        }
    
    # Configuration section
    with st.expander("Chart Configuration", expanded=False):
        # Create columns for chart options
        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
        
        with col_a:
            frequency_option = st.selectbox(
                "Frequency Display", 
                ["50Hz Only", "60Hz Only", "Both"],
                index=["50Hz Only", "60Hz Only", "Both"].index(st.session_state.chart_params.get('frequency_option', "50Hz Only"))
            )
            st.session_state.chart_params['frequency_option'] = frequency_option
        
        with col_b:
            chart_style = st.selectbox(
                "Chart Style", 
                ["Modern", "Classic"], 
                index=["Modern", "Classic"].index(st.session_state.chart_params['chart_style'])
            )
            st.session_state.chart_params['chart_style'] = chart_style
        
        with col_c:
            show_system = st.checkbox(
                "Show System Curve", 
                value=st.session_state.chart_params['show_system_curve']
            )
            st.session_state.chart_params['show_system_curve'] = show_system
        
        with col_d:
            show_grid = st.checkbox(
                "Show Grid", 
                value=st.session_state.chart_params['show_grid']
            )
            st.session_state.chart_params['show_grid'] = show_grid
        
        # System curve parameters (only shown if show_system_curve is True)
        if show_system:
            col_e, col_f = st.columns(2)
            with col_e:
                static_head = st.number_input(
                    "Static Head (m)", 
                    min_value=0.0, 
                    value=st.session_state.chart_params['static_head'], 
                    step=0.5
                )
                st.session_state.chart_params['static_head'] = static_head
            
            with col_f:
                k_factor = st.number_input(
                    "Friction Factor (k)", 
                    min_value=0.00001, 
                    value=st.session_state.chart_params['k_factor'], 
                    format="%.6f", 
                    step=0.00001
                )
                st.session_state.chart_params['k_factor'] = k_factor
        
        # Add axis range controls
        st.subheader("Axis Range Settings")
        col_g, col_h, col_i, col_j = st.columns(4)
        
        with col_g:
            min_flow = st.number_input(
                "Min Flow", 
                min_value=0.0,
                value=float(st.session_state.chart_params['min_flow'] or 0.0), 
                step=10.0
            )
            st.session_state.chart_params['min_flow'] = min_flow
            
        with col_h:
            max_flow_value = st.session_state.chart_params['max_flow']
            max_flow_value = float(max_flow_value) if max_flow_value is not None else 0.0
            
            max_flow = st.number_input(
                "Max Flow (0 for auto)", 
                min_value=0.0,
                value=max_flow_value, 
                step=100.0
            )
            st.session_state.chart_params['max_flow'] = max_flow if max_flow > 0 else None
            
        with col_i:
            min_head = st.number_input(
                "Min Head", 
                min_value=0.0,
                value=float(st.session_state.chart_params['min_head'] or 0.0), 
                step=1.0
            )
            st.session_state.chart_params['min_head'] = min_head
            
        with col_j:
            max_head_value = st.session_state.chart_params['max_head']
            max_head_value = float(max_head_value) if max_head_value is not None else 0.0
            
            max_head = st.number_input(
                "Max Head (0 for auto)", 
                min_value=0.0,
                value=max_head_value, 
                step=1.0
            )
            st.session_state.chart_params['max_head'] = max_head if max_head > 0 else None
    
    # Data input section
    st.subheader("Pump Data Input")
    
    # Units selection
    col1, col2 = st.columns(2)
    with col1:
        flow_unit = st.selectbox("Flow Rate Unit", ["GPM", "LPM", "m³/h"], index=1)
    with col2:
        head_unit = st.selectbox("Head Unit", ["ft", "m"], index=1)
    
    # Number of models and data points
    col3, col4 = st.columns(2)
    with col3:
        num_models = st.number_input("Number of Pump Models", min_value=1, max_value=5, value=st.session_state.num_models)
        st.session_state.num_models = num_models
    with col4:
        num_points = st.number_input("Number of Data Points", min_value=3, max_value=20, value=st.session_state.num_points)
        st.session_state.num_points = num_points
    
    # Model names input
    model_names = []
    col_names = st.columns(num_models)
    for i in range(num_models):
        default_name = f"Model-{chr(65+i)}"
        with col_names[i]:
            model_name = st.text_input(f"Model {i+1} Name", value=default_name, key=f"model_name_{i}")
            model_names.append(model_name)
    
    # Create tabs for each model
    model_tabs = st.tabs(model_names)
    
    # Store all model data
    all_models_data = {}
    
    # Function to create sample data
    def create_sample_data(model_index, num_points):
        max_head = 50 + (model_index * 10)  # Different max head for each model
        
        # Create decreasing head values
        head_values = np.linspace(max_head, 0, num_points)
        
        # Create increasing flow values
        max_flow = 100 + (model_index * 20)  # Different max flow for each model
        flow_values = np.linspace(0, max_flow, num_points)
        
        return head_values, flow_values
    
    # Create data editors for each model
    for i, tab in enumerate(model_tabs):
        with tab:
            st.info(f"Enter the head and flow data points for {model_names[i]}. You can copy-paste from Excel.")
            
            # Create empty dataframe if it doesn't exist for this model
            model_key = f"model_{i}_data"
            if model_key not in st.session_state.model_data:
                # Generate sample data for this model
                head_values, flow_values = create_sample_data(i, num_points)
                
                # Create dataframe with sample data
                df = pd.DataFrame({
                    f'Head ({head_unit})': head_values,
                    f'Flow ({flow_unit})': flow_values
                })
                st.session_state.model_data[model_key] = df
            
            # Ensure dataframe has the correct number of rows
            current_df = st.session_state.model_data[model_key]
            if len(current_df) != num_points:
                # Resize dataframe to match new number of points
                new_head_values, new_flow_values = create_sample_data(i, num_points)
                
                # Create new dataframe with correct size
                new_df = pd.DataFrame({
                    f'Head ({head_unit})': new_head_values,
                    f'Flow ({flow_unit})': new_flow_values
                })
                st.session_state.model_data[model_key] = new_df
                current_df = new_df
            
            # Data editor for this model with callback to store changes
            editor_key = f"data_editor_{model_names[i]}"
            
            # Define a callback to save edits
            def save_edits():
                if editor_key in st.session_state:
                    st.session_state.model_data[model_key] = st.session_state[editor_key]
            
            # Create the data editor with on_change callback
            edited_df = st.data_editor(
                current_df,
                use_container_width=True,
                num_rows="fixed",
                key=editor_key,
                on_change=save_edits
            )
            
            # Add to all models data
            all_models_data[model_names[i]] = edited_df
    
    # Generate button
    if st.button("Generate Pump Curve", type="primary"):
        st.session_state.chart_generated = True
    
    # Generate and display the chart
    if st.session_state.chart_generated and all_models_data:
        try:
            fig = generate_pump_curve(
                all_models_data,
                model_names=model_names,
                flow_unit=flow_unit,
                head_unit=head_unit,
                frequency_option=st.session_state.chart_params['frequency_option'],
                chart_style=st.session_state.chart_params['chart_style'],
                show_system_curve=st.session_state.chart_params['show_system_curve'],
                static_head=st.session_state.chart_params['static_head'],
                k_factor=st.session_state.chart_params['k_factor'],
                min_flow=st.session_state.chart_params['min_flow'],
                max_flow=st.session_state.chart_params['max_flow'],
                min_head=st.session_state.chart_params['min_head'],
                max_head=st.session_state.chart_params['max_head'],
                show_grid=st.session_state.chart_params['show_grid']
            )
            
            st.pyplot(fig)
            
            # Download button
            download_button_for_plot(fig)
            
        except Exception as e:
            st.error(f"Error generating chart: {e}")
            st.exception(e)

def generate_pump_curve(model_data, model_names, flow_unit, head_unit, frequency_option="50Hz Only", 
                       chart_style="Modern", show_system_curve=False, static_head=0.0, k_factor=0.0, 
                       min_flow=0.0, max_flow=None, min_head=0.0, max_head=None, show_grid=True):
    """Generate pump curves from the model data dictionaries"""
    
    # Import scipy for better curve interpolation
    from scipy import interpolate
    
    # Create a larger figure
    if chart_style == "Modern":
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Increase figure margins to make room for axes labels
    plt.subplots_adjust(bottom=0.2, right=0.9)
    
    # Determine which frequencies to plot
    frequencies_to_plot = []
    if frequency_option == "50Hz Only" or frequency_option == "Both":
        frequencies_to_plot.append('50Hz')
    if frequency_option == "60Hz Only" or frequency_option == "Both":
        frequencies_to_plot.append('60Hz')
    
    # Get color cycle for plots
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Find maximum flow and head for setting axis limits
    max_flow_data = 0
    max_head_data = 0
    
    # Plot each pump model
    for i, model_name in enumerate(model_names):
        if model_name not in model_data:
            continue
            
        df = model_data[model_name]
        
        # Get column names
        head_col = df.columns[0]  # First column is head
        flow_col = df.columns[1]  # Second column is flow
        
        # Extract data
        head_values = df[head_col].values
        flow_values = df[flow_col].values
        
        # Sort by flow for proper curve fit
        sort_idx = np.argsort(flow_values)
        flow_values = flow_values[sort_idx]
        head_values = head_values[sort_idx]
        
        # Create spline interpolation for smoother curves
        # If we have enough points, use cubic spline, otherwise use linear interpolation
        if len(flow_values) >= 4 and len(np.unique(flow_values)) >= 4:
            try:
                # Use cubic spline interpolation for smooth curves
                spline = interpolate.CubicSpline(flow_values, head_values)
                
                # Create high-resolution points for smooth curve
                flow_smooth = np.linspace(min(flow_values), max(flow_values), num=100)
                head_smooth = spline(flow_smooth)
                
                # Use these smooth points for plotting
                plot_flow_values = flow_smooth
                plot_head_values = head_smooth
            except Exception:
                # Fallback to B-spline interpolation if cubic spline fails
                try:
                    # Create a B-spline representation of the curve
                    tck = interpolate.splrep(flow_values, head_values, s=0)
                    
                    # Create high-resolution points for smooth curve
                    flow_smooth = np.linspace(min(flow_values), max(flow_values), num=100)
                    head_smooth = interpolate.splev(flow_smooth, tck, der=0)
                    
                    # Use these smooth points for plotting
                    plot_flow_values = flow_smooth
                    plot_head_values = head_smooth
                except Exception:
                    # If all interpolation fails, use the original points
                    plot_flow_values = flow_values
                    plot_head_values = head_values
        else:
            # Not enough points for cubic spline, use linear interpolation
            try:
                # Create higher resolution points
                flow_smooth = np.linspace(min(flow_values), max(flow_values), num=100)
                head_smooth = np.interp(flow_smooth, flow_values, head_values)
                
                plot_flow_values = flow_smooth
                plot_head_values = head_smooth
            except Exception:
                # If interpolation fails, use original points
                plot_flow_values = flow_values
                plot_head_values = head_values
        
        # Update max values
        max_flow_data = max(max_flow_data, np.max(plot_flow_values))
        max_head_data = max(max_head_data, np.max(plot_head_values))
        
        # Get color for this model
        color = colors[i % len(colors)]
        
        # Plot 50Hz curve (base data)
        if '50Hz' in frequencies_to_plot:
            ax.plot(plot_flow_values, plot_head_values, 
                   linestyle='-', linewidth=3.0, 
                   label=f"{model_name} (50Hz)", color=color,
                   zorder=10+i)
            
            # Removed the scatter plot of original data points
        
        # Plot 60Hz curve if requested (20% higher flow, 44% higher head)
        if '60Hz' in frequencies_to_plot:
            flow_values_60hz = plot_flow_values * 1.2
            head_values_60hz = plot_head_values * 1.44
            
            ax.plot(flow_values_60hz, head_values_60hz, 
                   linestyle='--', linewidth=3.0, 
                   label=f"{model_name} (60Hz)", color=color,
                   zorder=20+i)
            
            # Update max values for 60Hz
            max_flow_data = max(max_flow_data, np.max(flow_values_60hz))
            max_head_data = max(max_head_data, np.max(head_values_60hz))
    
    # Add system curve if requested
    if show_system_curve:
        # Determine maximum flow value for system curve
        if max_flow is None:
            max_flow_val = max_flow_data * 1.5
        else:
            max_flow_val = max_flow
        
        # Create system curve
        system_flows = np.linspace(0, max_flow_val, 100)
        system_heads = static_head + k_factor * (system_flows ** 2)
        
        # Plot system curve
        ax.plot(system_flows, system_heads, 'r-', linewidth=2.5, 
               label=f'System Curve (H={static_head}+{k_factor:.6f}×Q²)',
               zorder=30)
        
        # Find and plot intersection points
        for i, model_name in enumerate(model_names):
            if model_name not in model_data:
                continue
                
            df = model_data[model_name]
            
            # Get data
            head_col = df.columns[0]
            flow_col = df.columns[1]
            head_values = df[head_col].values
            flow_values = df[flow_col].values
            
            color = colors[i % len(colors)]
            
            # For 50Hz data
            if '50Hz' in frequencies_to_plot:
                try:
                    from scipy.interpolate import interp1d
                    
                    # Sort data points by flow for interpolation
                    sorted_indices = np.argsort(flow_values)
                    sorted_flows = flow_values[sorted_indices]
                    sorted_heads = head_values[sorted_indices]
                    
                    if len(np.unique(sorted_flows)) >= 2:
                        pump_curve_func = interp1d(sorted_flows, sorted_heads, 
                                                 bounds_error=False, fill_value="extrapolate")
                        
                        from scipy.optimize import minimize_scalar
                        
                        diff_func = lambda q: abs(pump_curve_func(q) - (static_head + k_factor * q**2))
                        
                        result = minimize_scalar(diff_func, 
                                               bounds=(min(sorted_flows), max(sorted_flows)), 
                                               method='bounded')
                        
                        if result.success:
                            op_flow = result.x
                            op_head = static_head + k_factor * op_flow**2
                            
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
                    print(f"Error finding 50Hz intersection: {e}")
            
            # For 60Hz data
            if '60Hz' in frequencies_to_plot:
                flow_values_60hz = flow_values * 1.2
                head_values_60hz = head_values * 1.44
                
                try:
                    from scipy.interpolate import interp1d
                    
                    # Sort data points by flow for interpolation
                    sorted_indices = np.argsort(flow_values_60hz)
                    sorted_flows = flow_values_60hz[sorted_indices]
                    sorted_heads = head_values_60hz[sorted_indices]
                    
                    if len(np.unique(sorted_flows)) >= 2:
                        pump_curve_func = interp1d(sorted_flows, sorted_heads, 
                                                 bounds_error=False, fill_value="extrapolate")
                        
                        from scipy.optimize import minimize_scalar
                        
                        diff_func = lambda q: abs(pump_curve_func(q) - (static_head + k_factor * q**2))
                        
                        result = minimize_scalar(diff_func, 
                                               bounds=(min(sorted_flows), max(sorted_flows)), 
                                               method='bounded')
                        
                        if result.success:
                            op_flow = result.x
                            op_head = static_head + k_factor * op_flow**2
                            
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
                    print(f"Error finding 60Hz intersection: {e}")
    
    # Set up axis labels
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
