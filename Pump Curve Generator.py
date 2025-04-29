import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def main():
    st.set_page_config(page_title="Pump Curve Generator", layout="wide")
    
    st.title("Pump Curve Generator Tool")
    
    st.markdown("""
    This tool allows you to generate pump performance curves similar to manufacturer specifications.
    You can either:
    1. Upload a CSV file with your pump curve data
    2. Manually input the data points for each pump curve
    """)
    
    # Sidebar for input method selection
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["Upload CSV", "Manual Input"]
    )
    
    if input_method == "Upload CSV":
        df = handle_csv_upload()
    else:
        df = handle_manual_input()
    
    # Generate and display the pump curve if data is available
    if df is not None and not df.empty:
        fig = generate_pump_curve(df)
        st.pyplot(fig)
        
        # Add download button for the plot
        download_button_for_plot(fig)
        
        # Display the data table
        st.subheader("Pump Curve Data")
        st.dataframe(df)
        
        # Add download button for the data
        download_button_for_data(df)

def handle_csv_upload():
    st.sidebar.subheader("Upload CSV File")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    # Sample CSV template for download
    sample_data = pd.DataFrame({
        'Flow (LPM)': [0, 100, 200, 300, 400, 500, 600, 700, 800],
        'DS-05 Head (m)': [7.5, 7.3, 7.0, 6.5, 5.8, 5.0, 4.0, 2.5, 0.8],
        'DS-10 Head (m)': [8.5, 8.3, 8.0, 7.5, 7.0, 6.5, 5.5, 4.2, 2.5],
        'DS-20 Head (m)': [9.3, 9.1, 8.8, 8.5, 8.0, 7.5, 7.0, 6.0, 4.5],
        'DS-30 Head (m)': [9.8, 9.6, 9.4, 9.0, 8.5, 8.0, 7.5, 6.5, 5.0]
    })
    
    st.sidebar.markdown("### Download Sample CSV Template")
    
    csv = sample_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="pump_curve_template.csv">Download Sample Template</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            if 'Flow' not in df.columns[0]:
                st.sidebar.warning("First column should contain flow values (LPM, m³/h, or GPM)")
                
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
            return None
    
    return None

def handle_manual_input():
    st.sidebar.subheader("Manual Data Input")
    
    # Units selection
    flow_unit = st.sidebar.selectbox("Flow Rate Unit", ["LPM", "m³/h", "GPM"])
    head_unit = st.sidebar.selectbox("Head Unit", ["m", "ft"])
    
    # Number of pump models
    num_models = st.sidebar.number_input("Number of Pump Models", min_value=1, max_value=5, value=1)
    
    # Number of data points
    num_points = st.sidebar.number_input("Number of Data Points", min_value=3, max_value=20, value=8)
    
    # Create columns for input form
    st.sidebar.subheader("Enter Data Points")
    
    # Prepare dataframe
    columns = ['Flow ({})'.format(flow_unit)]
    for i in range(1, num_models + 1):
        model_name = st.sidebar.text_input(f"Model {i} Name", value=f"Model-{i}")
        columns.append(f"{model_name} Head ({head_unit})")
    
    data = {}
    for col in columns:
        data[col] = [0] * num_points
    
    df = pd.DataFrame(data)
    
    # Create a form for data input
    with st.sidebar.form("data_input_form"):
        for i in range(num_points):
            st.write(f"Point {i+1}")
            col1, *col_rest = st.columns(len(columns))
            
            flow_val = col1.number_input(f"Flow {i+1}", value=i*100, key=f"flow_{i}")
            df.loc[i, columns[0]] = flow_val
            
            for j, col in enumerate(col_rest):
                head_val = col.number_input(f"{columns[j+1].split(' ')[0]} {i+1}", 
                                           value=float(max(10 - i*0.8, 0.5)), 
                                           key=f"head_{j}_{i}")
                df.loc[i, columns[j+1]] = head_val
        
        submit_button = st.form_submit_button("Generate Curve")
    
    if submit_button:
        return df
    else:
        return None

def generate_pump_curve(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine flow unit and head unit from column names
    flow_col = df.columns[0]
    flow_unit = flow_col.split('(')[-1].split(')')[0]
    
    head_cols = df.columns[1:]
    head_unit = head_cols[0].split('(')[-1].split(')')[0]
    
    # Plot each pump curve
    for column in head_cols:
        model_name = column.split(' ')[0]
        ax.plot(df[flow_col], df[column], linewidth=2, label=model_name)
    
    # Set up the primary x and y axes
    ax.set_xlabel(f'Flow ({flow_unit})', fontsize=12)
    ax.set_ylabel(f'Head ({head_unit})', fontsize=12)
    ax.grid(True)
    
    # Add secondary x-axis for alternative flow units if primary is LPM
    if flow_unit == "LPM":
        # Add m³/h axis at bottom
        ax_m3h = ax.twiny()
        ax_m3h.set_xlim(ax.get_xlim())
        ax_m3h.set_xticks(ax.get_xticks())
        m3h_values = [round(x/60, 1) for x in ax.get_xticks()]
        ax_m3h.set_xticklabels(m3h_values)
        ax_m3h.set_xlabel(f'Flow (m³/h)', fontsize=12)
        ax_m3h.spines['bottom'].set_position(('outward', 40))
        
        # Add GPM axis at bottom
        ax_gpm = ax.twiny()
        ax_gpm.set_xlim(ax.get_xlim())
        ax_gpm.set_xticks(ax.get_xticks())
        gpm_values = [round(x*0.264172, 1) for x in ax.get_xticks()]
        ax_gpm.set_xticklabels(gpm_values)
        ax_gpm.set_xlabel(f'Flow (GPM)', fontsize=12)
        ax_gpm.spines['bottom'].set_position(('outward', 80))
    
    # Add secondary y-axis for alternative head units if primary is m
    if head_unit == "m":
        ax_ft = ax.twinx()
        ax_ft.set_ylim(ax.get_ylim())
        ax_ft.set_yticks(ax.get_yticks())
        ft_values = [round(x*3.28084, 1) for x in ax.get_yticks()]
        ax_ft.set_yticklabels(ft_values)
        ax_ft.set_ylabel(f'Head (ft)', fontsize=12)
    
    # Set legend
    ax.legend(loc='best', fontsize=10)
    
    plt.title('Pump Performance Curves', fontsize=16)
    plt.tight_layout()
    
    return fig

def download_button_for_plot(fig):
    # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Create download button
    btn = st.download_button(
        label="Download Pump Curve Plot (PNG)",
        data=buf,
        file_name="pump_curve_plot.png",
        mime="image/png"
    )

def download_button_for_data(df):
    # Convert dataframe to CSV
    csv = df.to_csv(index=False)
    
    # Create download button
    btn = st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="pump_curve_data.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
