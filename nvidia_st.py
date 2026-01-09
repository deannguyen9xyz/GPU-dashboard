import pynvml
import pandas as pd
import streamlit as st
from datetime import datetime

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="GPU Live Monitor",
    page_icon="🚀",
    layout="wide"
)

# 2. Initialize NVML once
@st.cache_resource
def get_gpu_handle():
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        return None

handle = get_gpu_handle()

# 3. Persistent Data Storage
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Temp', 'Memory', 'Load'])

# 4. The Dashboard Fragment
@st.fragment(run_every=5.0) 
def render_gpu_data():
    if handle:
        # --- DATA COLLECTION ---
        name = pynvml.nvmlDeviceGetName(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        now = datetime.now().strftime('%H:%M:%S')

        # Update Session Data
        new_entry = pd.DataFrame({'Time': [now], 'Temp': [temp], 'Memory': [mem], 'Load': [util]})
        st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True).iloc[-50:]
        
        # Calculate deltas for a "Pro" look
        prev_temp = st.session_state.history['Temp'].iloc[-2] if len(st.session_state.history) > 1 else temp

        # --- UI TOP SECTION ---
        st.title(f"📊 GPU: {name}")
        
        c1, c2, c3 = st.columns(3)
        # Metric with delta showing change since last refresh
        c1.metric("Temperature", f"{temp}°C", delta=f"{temp - prev_temp}°C", delta_color="inverse")
        c2.metric("Memory Used", f"{mem:.0f} MB")
        c3.metric("Utilization", f"{util}%")

        st.divider()

        # --- VISUAL CHARTS (Split into 3 for proper scaling) ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("🔥 Temperature & Load")
            # We can group Temp and Load because they share a similar scale (0-100)
            st.line_chart(st.session_state.history.set_index('Time')[['Temp', 'Load']], height=300)

        with chart_col2:
            st.subheader("💾 VRAM Usage (MB)")
            # VRAM gets its own chart because it's in the thousands
            st.area_chart(st.session_state.history.set_index('Time')['Memory'], color="#29B5E8", height=300)
            
    else:
        st.error("No NVIDIA GPU detected.")

# 5. Call the fragment
render_gpu_data()