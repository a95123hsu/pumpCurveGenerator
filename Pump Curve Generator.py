import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to plot the pump curve
def plot_pump_curve(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, df in data.groupby('Pump Model'):
        ax.plot(df['Flow'], df['Head'], label=label, marker='o')

    ax.set_title('Pump Curve')
    ax.set_xlabel('Flow (LPM)')
    ax.set_ylabel('Head (m)')
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)

# Function to upload and process CSV data
def upload_csv():
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview", df.head())
        return df
    return None

# Function to manually input data
def manual_input():
    st.subheader("Manual Input")
    num_rows = st.number_input("Number of data points", min_value=1, max_value=20, value=5)
    
    data = {
        'Flow': [],
        'Head': [],
        'Pump Model': []
    }
    
    for i in range(num_rows):
        with st.expander(f"Data Point {i + 1}"):
            flow = st.number_input(f"Flow (LPM) {i + 1}", value=100)
            head = st.number_input(f"Head (m) {i + 1}", value=5)
            pump_model = st.text_input(f"Pump Model {i + 1}", value=f"Model {i + 1}")
            
            data['Flow'].append(flow)
            data['Head'].append(head)
            data['Pump Model'].append(pump_model)
    
    df = pd.DataFrame(data)
    st.write("Data Preview", df.head())
    return df

# Main Streamlit app
def main():
    st.title("Pump Curve Generator")
    
    data_source = st.selectbox("Select Data Source", options=["Upload CSV", "Manual Input"])
    
    if data_source == "Upload CSV":
        df = upload_csv()
    else:
        df = manual_input()
    
    if df is not None:
        plot_pump_curve(df)

if __name__ == "__main__":
    main()
