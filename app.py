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
def calculate_jv_parameters(df):
    """
    Calculate JV parameters from V and J data.
    Assumes J is positive at V = 0 and decreases.
    """
    if df is None or len(df) < 5:
        return {}
    
    # Sort by V to be sure
    df_sorted = df.sort_values(by="V").reset_index(drop=True)
    v_arr = df_sorted["V"].values
    j_arr = df_sorted["J"].values
    
    # 1. Jsc: J at V = 0
    # Interpolate Jsc at V = 0.
    jsc = float(np.interp(0.0, v_arr, j_arr))
    
    # 2. Voc: V at J = 0
    # Find where J crosses 0.
    voc = np.nan
    for i in range(len(j_arr) - 1):
        if (j_arr[i] >= 0 and j_arr[i+1] <= 0) or (j_arr[i] <= 0 and j_arr[i+1] >= 0):
            j1, j2 = j_arr[i], j_arr[i+1]
            v1, v2 = v_arr[i], v_arr[i+1]
            if j1 != j2:
                voc = float(v1 + (0.0 - j1) * (v2 - v1) / (j2 - j1))
                break
    
    if np.isnan(voc):
        idx = np.argmin(np.abs(j_arr))
        voc = float(v_arr[idx])
        
    # 3. Fill Factor (FF) and Efficiency (Eff)
    p_multiplier = 1.0 if jsc >= 0 else -1.0
    p_vals = v_arr * j_arr * p_multiplier
    
    mpp_idx = np.argmax(p_vals)
    p_max = float(p_vals[mpp_idx])
    
    abs_jsc = abs(jsc)
    abs_voc = abs(voc)
    
    if abs_jsc > 0 and abs_voc > 0:
        ff = float((p_max / (abs_voc * abs_jsc)) * 100)
        # Eff = P_max % assuming 100 mW/cm2 illumination and V in volts, J in mA/cm2.
        eff = float(p_max)
    else:
        ff = np.nan
        eff = np.nan
        
    # 4. Rs: Series resistance (dV/dJ near Voc)
    rs = np.nan
    try:
        voc_mask = (v_arr >= 0.9 * voc) & (v_arr <= 1.1 * voc) if voc != 0 else np.array([False]*len(v_arr))
        if voc_mask.sum() >= 2:
            slope, _ = np.polyfit(j_arr[voc_mask], v_arr[voc_mask], 1)
            rs = float(abs(slope * 1000))
        else:
            closest_idxs = np.argsort(np.abs(v_arr - voc))[:3]
            slope, _ = np.polyfit(j_arr[closest_idxs], v_arr[closest_idxs], 1)
            rs = float(abs(slope * 1000))
    except Exception:
        pass
        
    # 5. Rsh: Shunt resistance (dV/dJ near V = 0)
    rsh = np.nan
    try:
        v0_mask = (v_arr >= -0.1) & (v_arr <= 0.1)
        if v0_mask.sum() >= 2:
            slope, _ = np.polyfit(j_arr[v0_mask], v_arr[v0_mask], 1)
            rsh = float(abs(slope * 1000))
        else:
            closest_idxs = np.argsort(np.abs(v_arr))[:3]
            slope, _ = np.polyfit(j_arr[closest_idxs], v_arr[closest_idxs], 1)
            rsh = float(abs(slope * 1000))
    except Exception:
        pass
        
    return {
        'Voc_calc': abs_voc,
        'Jsc_calc': abs_jsc,
        'FF_calc': ff,
        'Eff_calc': eff,
        'Rs_calc': rs,
        'Rsh_calc': rsh
    }

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

        # Check if the line is likely a data row (only numbers and delimiters)
        is_data_line = False
        if len(nums) >= 2:
            rem = re.sub(r'[\d\s.+\-eE,;\t]', '', ln)
            if len(rem) == 0:
                is_data_line = True

        # Extract V, J data points
        if is_data_line:
            try:
                v = float(nums[0])
                j = float(nums[1])
                numeric_rows.append((v, j))
            except Exception:
                pass

        # Extract parameters using precise regex patterns
        param_patterns = {
            'Voc': r'Voc\s*\[?V\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'Jsc': r'Jsc\s*\[?mA\/cm\^?²?2?\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'FF': r'FF\s*\[%\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'Eff': r'(?:Eff\.|Eff|Efficiency)\s*\[%\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'Rs': r'R\s*s\s*\[?(?:ohm|Ω)\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'Rsh': r'R\s*sh\s*\[?(?:ohm|Ω)\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            'HI': r'HI\s*\[%\]?[\s\:=~\-]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        }
        for key, pat in param_patterns.items():
            match = re.search(pat, ln_norm, flags=re.IGNORECASE)
            if match:
                try:
                    params[key] = float(match.group(1))
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
        'HI': float(params.get('HI', np.nan)),
        'Rs': float(params.get('Rs', np.nan)),
        'Rsh': float(params.get('Rsh', np.nan))
    }

    # Calculate fallback parameters if df is present and some parameters are NaN
    if df is not None and not df.empty:
        if (np.isnan(clean_params['Voc']) or np.isnan(clean_params['Jsc']) or 
            np.isnan(clean_params['FF']) or np.isnan(clean_params['Eff']) or 
            np.isnan(clean_params['Rs']) or np.isnan(clean_params['Rsh'])):
            
            calc = calculate_jv_parameters(df)
            if calc:
                if np.isnan(clean_params['Voc']):
                    clean_params['Voc'] = calc.get('Voc_calc', np.nan)
                if np.isnan(clean_params['Jsc']):
                    clean_params['Jsc'] = calc.get('Jsc_calc', np.nan)
                if np.isnan(clean_params['FF']):
                    clean_params['FF'] = calc.get('FF_calc', np.nan)
                if np.isnan(clean_params['Eff']):
                    clean_params['Eff'] = calc.get('Eff_calc', np.nan)
                if np.isnan(clean_params['Rs']):
                    clean_params['Rs'] = calc.get('Rs_calc', np.nan)
                if np.isnan(clean_params['Rsh']):
                    clean_params['Rsh'] = calc.get('Rsh_calc', np.nan)

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
    
    # Parse header and values with index safety
    headers = [h.strip() for h in lines[0].split('\t')]
    val_line_idx = 2 if len(lines) >= 3 else 1
    values = [v.strip() for v in lines[val_line_idx].split('\t')]
    
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

    # Efficiency filter
    if txt_origin == 'Oninn':
        eff = d['params'].get('Eff', np.nan)
        if not np.isnan(eff) and eff < min_efficiency:
            continue
    else:
        eff_rev = d['params'].get('PCE_rev', np.nan)
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
    def safe_fmt(val, fmt):
        return fmt.format(val) if not (isinstance(val, float) and np.isnan(val)) else "N/A"

    if txt_origin == "Oninn":
        best_cell = max(filtered_data, key=lambda x: x['params']['Eff'] if not np.isnan(x['params']['Eff']) else -1)
        st.markdown("### 🏆 Best Performing Cell")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Efficiency", safe_fmt(best_cell['params']['Eff'], "{:.2f}%"))
        m2.metric("Voc", safe_fmt(best_cell['params']['Voc'], "{:.3f} V"))
        m3.metric("Jsc", safe_fmt(best_cell['params']['Jsc'], "{:.2f} mA/cm²"))
        m4.metric("FF", safe_fmt(best_cell['params']['FF'], "{:.1f}%"))
        m5.metric("Rs", safe_fmt(best_cell['params']['Rs'], "{:.1f} Ω"))
        m6.metric("Rsh", safe_fmt(best_cell['params']['Rsh'], "{:.1f} Ω"))
    else:
        best_cell = max(filtered_data, key=lambda x: x['params']['PCE_rev'] if not np.isnan(x['params']['PCE_rev']) else -1)
        st.markdown("### 🏆 Best Performing Cell")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("PCE Rev", safe_fmt(best_cell['params']['PCE_rev'], "{:.2f}%"))
        m2.metric("Voc Rev", safe_fmt(best_cell['params']['Voc_rev'], "{:.3f} V"))
        m3.metric("Jsc Rev", safe_fmt(best_cell['params']['Jsc_rev'], "{:.2f} mA/cm²"))
        m4.metric("FF Rev", safe_fmt(best_cell['params']['FF_rev'], "{:.1f}%"))
        m5.metric("Rs", safe_fmt(best_cell['params']['Rs'], "{:.1f} Ω"))
        m6.metric("Rsh", safe_fmt(best_cell['params']['Rsh'], "{:.1f} Ω"))
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
    if txt_origin == "Oninn":
        base_cols = ['Filename', 'Voc', 'Jsc', 'FF', 'Eff']
        stat_base  = ['Voc', 'Jsc', 'FF', 'Eff']
    else:
        base_cols = ["Filename", "Voc_rev", "Voc_fwd", "Jsc_rev", "Jsc_fwd",
                     "FF_rev", "FF_fwd", "PCE_rev", "PCE_fwd"]
        stat_base  = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd',
                      'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd']
    extra_cols = ['HI', 'Rs', 'Rsh']
    # Only include columns that actually exist in the dataframe
    available_base = [c for c in base_cols if c in df_params.columns]
    available_stat = [c for c in stat_base  if c in df_params.columns]
    available_extra = [c for c in extra_cols if c in df_params.columns]
    df_params = df_params[available_base + available_extra]
    
    st.dataframe(df_params, use_container_width=True)
    
    st.markdown("### Statistics")
    stat_cols = available_stat + available_extra
    stats = df_params[stat_cols].describe().T
    st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("### Distributions")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown("**Boxplots**")
        if txt_origin == "Oninn":
            plot_base = ['Voc', 'Jsc', 'FF', 'Eff', 'Rs', 'Rsh']
        else:
            plot_base = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd',
                         'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd', 'HI', 'Rs', 'Rsh']
        # Only keep columns that exist in the dataframe
        plot_params = [c for c in plot_base if c in df_params.columns]
        param_to_plot = st.selectbox("Select Parameter", plot_params)
        fig_box = px.box(df_params, y=param_to_plot, points="all", title=f"{param_to_plot} Distribution")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with d_col2:
        st.markdown("**Correlations**")
        if txt_origin == "Oninn":
            corr_base = ['Voc', 'Jsc', 'FF', 'Eff', 'Rs', 'Rsh']
        else:
            corr_base = ['Voc_rev', 'Voc_fwd', 'Jsc_rev', 'Jsc_fwd',
                         'FF_rev', 'FF_fwd', 'PCE_rev', 'PCE_fwd', 'HI', 'Rs', 'Rsh']
        # Only keep columns that exist in the dataframe
        corr_params = [c for c in corr_base if c in df_params.columns]
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
