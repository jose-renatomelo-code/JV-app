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
        'HI': np.nan,
        'Rs': np.nan,
        'Rsh': np.nan
    }

    return {'filename': filename, 'df': df, 'params': clean_params}

def parse_maximus(uploaded_file):
    """
    Parses a simple text file with parameters.
    First line contains parameter names, subsequent lines contain values.
    Returns a dictionary with metadata and parameters.
    """
    filename = uploaded_file.name
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    
    clean_params = {
        'Voc_rev': np.nan,
        'Voc_fwd': np.nan,
        'Jsc_rev': np.nan,
        'Jsc_fwd': np.nan,
        'FF_rev': np.nan,
        'FF_fwd': np.nan,
        'PCE_rev': np.nan,
        'PCE_fwd': np.nan,
        'HI': np.nan,
        'Rs': np.nan,
        'Rsh': np.nan
    }
    
    if len(lines) < 2:
        return {'filename': filename, 'df': None, 'params': clean_params}
    
    # Parse header and values
    headers = [h.strip() for h in lines[0].split('\t')]
    values = [v.strip() for v in lines[2].split('\t')]
    
    # Map values to parameters
    param_map = {
        'Voc_jv_rev(V)': 'Voc_rev',
        'Voc_rev': 'Voc_rev',
        'Voc_jv_fwd(V)': 'Voc_fwd',
        'Voc_fwd': 'Voc_fwd',
        'Jsc_rev(mA/cm²)': 'Jsc_rev',
        'Jsc_rev': 'Jsc_rev',
        'Jsc_fwd(mA/cm²)': 'Jsc_fwd',
        'Jsc_fwd': 'Jsc_fwd',
        'FF_rev(%)': 'FF_rev',
        'FF_rev': 'FF_rev',
        'FF_fwd(%)': 'FF_fwd',
        'FF_fwd': 'FF_fwd',
        'PCE_jv_rev(%)': 'PCE_rev',
        'PCE_rev': 'PCE_rev',
        'PCE_jv_fwd(%)': 'PCE_fwd',
        'PCE_fwd': 'PCE_fwd',
        'HI(%)': 'HI',
        'HI': 'HI',
        'Rs(ohms)': 'Rs',
        'Rs': 'Rs',
        'Rsh(ohms)': 'Rsh',
        'Rsh': 'Rsh'
    }
    
    for header, value in zip(headers, values):
        header_clean = header.strip()
        if header_clean in param_map:
            try:
                clean_params[param_map[header_clean]] = float(value)
            except (ValueError, TypeError):
                pass
    
    return {'filename': filename, 'df': None, 'params': clean_params}

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("JV Analyser Pro")
    st.markdown("---")

    # inicializa chave do uploader e a lista persistente
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    # função de limpeza: incrementa a key, limpa a lista e força rerun
    def clean_all_uploads():
        st.session_state["uploaded_files"] = []
        st.session_state["uploader_key"] += 1
    
    # Escolha origem do txt para parsing
    txt_origin = st.radio("JV Software", ["Oninn", "Renato"], index=0)
    # file_uploader com key dinâmica baseada em uploader_key
    new_files = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    # se o usuário acabou de enviar arquivos, atualiza o estado (substitui)
    if new_files:
        st.session_state["uploaded_files"] = new_files

    # Pega a lista atual (pode ser [])
    uploaded_files = st.session_state["uploaded_files"]

    # Botão com on_click que chama a função (um clique basta)
    st.button("Clean Uploads", on_click=clean_all_uploads)

    # Filtros
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
    if txt_origin == "Oninn":
        parsed = parse_jv_file(f)
    elif txt_origin == "Renato": 
        parsed = parse_maximus(f)

    if parsed is not None:
        data_list.append(parsed)

if not data_list:
    st.error("No valid JV data found in uploaded files.")
    st.stop()

# Filter data
filtered_data = []
for d in data_list:
    name = d['filename']
    lname = name.lower()
    if txt_origin == 'oninn':
        eff = d['params'].get('Eff', np.nan)
    else:
        eff_rev = d['params'].get('PCE_rev', np.nan)
    
    # Efficiency filter
    if not np.isnan(eff_rev) and eff_rev < min_efficiency:
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
    
    if txt_origin == "Oninn":
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
    if txt_origin == "oninn": 
        best_cell = max(filtered_data, key=lambda x: x['params']['Eff'] if not np.isnan(x['params']['Eff']) else -1)
    else:
        best_cell = max(filtered_data, key=lambda x: x['params']['PCE_rev'] if not np.isnan(x['params']['PCE_rev']) else -1)

    st.markdown("### 🏆 Best Performing Cell")
    
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Efficiency Reverse", f"{best_cell['params']['PCE_rev']:.2f}%")
    m2.metric("Efficiency Forward", f"{best_cell['params']['PCE_fwd']:.2f}%")
    m3.metric("Voc Reverse", f"{best_cell['params']['Voc_rev']:.3f} V")
    m4.metric("Voc Forward", f"{best_cell['params']['Voc_fwd']:.3f} V")
    m5.metric("Jsc Reverse", f"{best_cell['params']['Jsc_rev']:.2f} mA/cm²")
    m6.metric("Jsc Forward", f"{best_cell['params']['Jsc_fwd']:.2f} mA/cm²")
    m7.metric("FF Reverse", f"{best_cell['params']['FF_rev']:.1f}%")
    m8.metric("FF Forward", f"{best_cell['params']['FF_fwd']:.1f}%")
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
    
    # Reorder columns (include extra parameters if available)
    if txt_origin == "oninn":
        cols = ['Filename', 'Voc', 'Jsc', 'FF', 'Eff']
    else: 
        cols = ["Filename", "Voc_rev", "Jsc_rev", "FF_rev", "PCE_rev"]
    extra_cols = ['HI', 'Rs', 'Rsh']
    available_cols = [col for col in extra_cols if col in df_params.columns]
    cols.extend(available_cols)
    df_params = df_params[cols]
    
    st.dataframe(df_params, use_container_width=True)
    
    st.markdown("### Statistics")
    if txt_origin == "oninn": 
        stat_cols = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd', 'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd']
    else: 
        stat_cols = ['Voc', 'Jsc', 'FF', 'Eff']
    extra_cols = ['HI', 'Rs', 'Rsh']
    available_extra = [col for col in extra_cols if col in df_params.columns]
    stat_cols.extend(available_extra)
    stats = df_params[stat_cols].describe().T
    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("### Distributions")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown("**Boxplots**")
        if txt_origin == "Renato": 
            plot_params = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd', 'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd', 'HI', 'Rs_rev', 'Rs_fwd', 'Rsh_rev']
        else: 
            plot_params = ['Voc', 'Jsc', 'FF', 'Eff']
        extra_cols = ['HI', 'Rs', 'Rsh']
        available_extra = [col for col in extra_cols if col in df_params.columns]
        plot_params.extend(available_extra)
        param_to_plot = st.selectbox("Select Parameter", plot_params)
        fig_box = px.box(df_params, y=param_to_plot, points="all", title=f"{param_to_plot} Distribution")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with d_col2:
        st.markdown("**Correlations**")
        if txt_origin == "Renato":
            corr_params = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd', 'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd', 
                        'HI', 'Rs', 'Rsh']
        else:
            corr_params = ['Voc', 'Jsc', 'FF', 'Eff']
        extra_cols = ['HI', 'Rs', 'Rsh']
        available_extra = [col for col in extra_cols if col in df_params.columns]
        corr_params.extend(available_extra)
        fig_corr = px.scatter_matrix(df_params, dimensions=corr_params, title="Parameter Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

    # Export
    csv = df_params.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Report CSV",
        data=csv,
        file_name='jv_report.csv',
        mime='text/csv',
    )
