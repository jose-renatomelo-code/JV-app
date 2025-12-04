import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import io

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="JV Analyser Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Parsing Logic
# -----------------------------------------------------------------------------
def parse_jv_file(uploaded_file):
    """
    Parses a single uploaded JV file.
    Returns a dictionary with metadata, dataframe, and parameters.
    """
    filename = uploaded_file.name
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    numeric_rows = []
    params = {}

    for ln in lines:
        # Normalize decimal separator
        ln_norm = ln.replace(',', '.')
        # Find numbers (including scientific notation)
        nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', ln_norm)

        # Extract V, J data points
        if len(nums) >= 2:
            try:
                v = float(nums[0])
                j = float(nums[1])
                numeric_rows.append((v, j))
            except Exception:
                pass

        # Extract parameters
        param_labels = {
            'Voc': r'Voc\s*\[?V\]?',
            'Jsc': r'Jsc\s*\[?mA\/cm',
            'FF': r'FF\s*\[%\]?',
            'Eff': r'(Eff\.|Eff|Efficiency)\s*\[%\]?'
        }
        for key, pat in param_labels.items():
            if re.search(pat, ln, flags=re.IGNORECASE):
                vals = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', ln_norm)
                if vals:
                    try:
                        params[key] = float(vals[-1])
                    except Exception:
                        params[key] = np.nan

    df = None
    if numeric_rows:
        df = pd.DataFrame(numeric_rows, columns=["V", "J"])
        df = df.sort_values(by="V", ignore_index=True)
        df["P"] = df["V"] * df["J"]

    # Clean params
    clean_params = {
        'Voc': float(params.get('Voc', np.nan)),
        'Jsc': float(params.get('Jsc', np.nan)),
        'FF': float(params.get('FF', np.nan)),
        'Eff': float(params.get('Eff', np.nan)),
    }

    return {'filename': filename, 'df': df, 'params': clean_params}

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("JV Analyser Pro")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload .txt files", 
        type=["txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state['uploaded_files'] = uploaded_files
    else: 
        uploaded_files = st.session_state.get('uploaded_files', [])
    
    st.markdown("### Filters")
    scan_direction = st.radio("Scan Direction", ["All", "FWD", "REV"], index=0)
    min_efficiency = st.number_input("Min Efficiency (%)", min_value=0.0, value=0.0, step=0.1)

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
if not uploaded_files:
    st.info("👋 Welcome! Please upload your JV .txt files in the sidebar to get started.")
    st.stop()

# Process files
data_list = []
for f in uploaded_files:
    parsed = parse_jv_file(f)
    if parsed['df'] is not None and not parsed['df'].empty:
        data_list.append(parsed)

if not data_list:
    st.error("No valid JV data found in uploaded files.")
    st.stop()

# Filter data
filtered_data = []
for d in data_list:
    name = d['filename']
    lname = name.lower()
    eff = d['params'].get('Eff', np.nan)
    
    # Efficiency filter
    if not np.isnan(eff) and eff < min_efficiency:
        continue
        
    # Direction filter
    if scan_direction == "FWD" and "fwd" not in lname:
        continue
    if scan_direction == "REV" and "rev" not in lname:
        continue
        
    filtered_data.append(d)

if not filtered_data:
    st.warning("No files match the current filters.")
    st.stop()

# -----------------------------------------------------------------------------
# Dashboard Layout
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "📑 Report Lab"])

with tab1:
    st.subheader("Interactive Curves")
    
    col1, col2 = st.columns(2)
    
    # Prepare data for plotting
    
    fig_jv = go.Figure()
    fig_pv = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, d in enumerate(filtered_data):
        df = d['df']
        name = d['filename']
        color = colors[i % len(colors)]
        
        fig_jv.add_trace(go.Scatter(x=df['V'], y=df['J'], mode='lines', name=name, line=dict(color=color)))
        fig_pv.add_trace(go.Scatter(x=df['V'], y=df['P'], mode='lines', name=name, line=dict(color=color)))

    # Layout updates
    fig_jv.update_layout(
        title="J-V Curves",
        xaxis_title="Voltage (V)",
        yaxis_title="Current Density (mA/cm²)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig_pv.update_layout(
        title="P-V Curves",
        xaxis_title="Voltage (V)",
        yaxis_title="Power Density (mW/cm²)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    with col1:
        st.plotly_chart(fig_jv, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pv, use_container_width=True)

    # Summary Metrics (Best Cell)
    best_cell = max(filtered_data, key=lambda x: x['params']['Eff'] if not np.isnan(x['params']['Eff']) else -1)
    st.markdown("### 🏆 Best Performing Cell")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Efficiency", f"{best_cell['params']['Eff']:.2f}%")
    m2.metric("Voc", f"{best_cell['params']['Voc']:.3f} V")
    m3.metric("Jsc", f"{best_cell['params']['Jsc']:.2f} mA/cm²")
    m4.metric("FF", f"{best_cell['params']['FF']:.1f}%")
    st.caption(f"File: {best_cell['filename']}")

with tab2:
    st.subheader("Statistical Report")
    
    # Create DataFrame of parameters
    params_list = []
    for d in filtered_data:
        p = d['params'].copy()
        p['Filename'] = d['filename']
        params_list.append(p)
    
    df_params = pd.DataFrame(params_list)
    
    # Reorder columns
    cols = ['Filename', 'Voc', 'Jsc', 'FF', 'Eff']
    df_params = df_params[cols]
    
    st.dataframe(df_params, use_container_width=True)
    
    st.markdown("### Statistics")
    stats = df_params[['Voc', 'Jsc', 'FF', 'Eff']].describe().T
    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("### Distributions")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown("**Boxplots**")
        param_to_plot = st.selectbox("Select Parameter", ['Voc', 'Jsc', 'FF', 'Eff'])
        fig_box = px.box(df_params, y=param_to_plot, points="all", title=f"{param_to_plot} Distribution")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with d_col2:
        st.markdown("**Correlations**")
        fig_corr = px.scatter_matrix(df_params, dimensions=['Voc', 'Jsc', 'FF', 'Eff'], title="Parameter Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Export
    csv = df_params.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report CSV",
        data=csv,
        file_name='jv_report.csv',
        mime='text/csv',
    )



