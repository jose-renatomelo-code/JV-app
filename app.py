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

    if len(v_arr) > 3 and not np.isnan(voc) and not np.isnan(jsc):
        try:
            # We need I in Amps for Ohms
            I_amps = (j_arr * 0.14) / 1000.0

            # Gradient dI/dV
            dI_dV = np.gradient(I_amps, v_arr)

            # R_s at Voc
            idx_voc = (np.abs(v_arr - voc)).argmin()
            if dI_dV[idx_voc] != 0:
                rs = abs(1.0 / dI_dV[idx_voc])

            # R_sh at Jsc
            idx_jsc = (np.abs(v_arr - 0.0)).argmin()
            if dI_dV[idx_jsc] != 0:
                rsh = abs(1.0 / dI_dV[idx_jsc])

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
    Parses a Maximus/Renato JV data file.
    Only reads voltage, avg_jcurrent, direction and Loop, and calculates
    PV parameters using calculate_jv_parameters for all loops and both directions (rev and fwd).
    """
    filename = uploaded_file.name
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    
    try:
        df_raw = pd.read_csv(io.StringIO(content), sep=None, engine='python')
    except Exception:
        df_raw = pd.read_csv(io.StringIO(content), sep='\t')
        
    if df_raw is None or df_raw.empty:
        return []

    # Identify only the required columns: voltage, avg_jcurrent, direction, loop
    cols = {c.strip(): c for c in df_raw.columns}
    col_v = next((cols[c] for c in cols if 'voltage' in c.lower() or c.lower() == 'v'), None)
    col_j = next((cols[c] for c in cols if 'avg_jcurrent' in c.lower() or 'current' in c.lower() or c.lower() == 'j'), None)
    col_dir = next((cols[c] for c in cols if 'direction' in c.lower() or 'dir' in c.lower()), None)
    col_loop = next((cols[c] for c in cols if 'loop' in c.lower()), None)

    if col_v is None or col_j is None:
        return []

    needed = [col_v, col_j]
    if col_dir:
        needed.append(col_dir)
    if col_loop:
        needed.append(col_loop)

    df_sub = df_raw[needed].copy()
    rename_map = {col_v: 'V', col_j: 'J'}
    if col_dir:
        rename_map[col_dir] = 'direction'
    if col_loop:
        rename_map[col_loop] = 'Loop'
    df_sub = df_sub.rename(columns=rename_map)

    df_sub['V'] = pd.to_numeric(df_sub['V'], errors='coerce')
    df_sub['J'] = pd.to_numeric(df_sub['J'], errors='coerce')
    df_sub = df_sub.dropna(subset=['V', 'J'])

    if 'direction' not in df_sub.columns:
        df_sub['direction'] = 'rev'
    if 'Loop' not in df_sub.columns:
        df_sub['Loop'] = 1

    unique_loops = df_sub['Loop'].dropna().unique()
    if len(unique_loops) == 0:
        unique_loops = [1]

    results = []
    for loop in unique_loops:
        loop_df = df_sub[df_sub['Loop'] == loop].copy()

        df_rev = loop_df[loop_df['direction'].astype(str).str.lower().str.contains('rev')].copy()
        df_fwd = loop_df[loop_df['direction'].astype(str).str.lower().str.contains('fwd')].copy()

        calc_rev = calculate_jv_parameters(df_rev) if not df_rev.empty else {}
        calc_fwd = calculate_jv_parameters(df_fwd) if not df_fwd.empty else {}

        pce_rev = calc_rev.get('Eff_calc', np.nan)
        pce_fwd = calc_fwd.get('Eff_calc', np.nan)

        hi = np.nan
        if not np.isnan(pce_rev) and not np.isnan(pce_fwd) and pce_fwd != 0:
            hi = float(((pce_rev - pce_fwd) / pce_fwd) * 100)

        clean_params = {
            'Loop': loop,
            'Voc_rev': calc_rev.get('Voc_calc', np.nan),
            'Voc_fwd': calc_fwd.get('Voc_calc', np.nan),
            'Jsc_rev': calc_rev.get('Jsc_calc', np.nan),
            'Jsc_fwd': calc_fwd.get('Jsc_calc', np.nan),
            'FF_rev': calc_rev.get('FF_calc', np.nan),
            'FF_fwd': calc_fwd.get('FF_calc', np.nan),
            'PCE_rev': pce_rev,
            'PCE_fwd': pce_fwd,
            'HI': hi,
            'Rs': calc_rev.get('Rs_calc', np.nan),
            'Rsh': calc_rev.get('Rsh_calc', np.nan),
            'Eff': pce_rev if not np.isnan(pce_rev) else pce_fwd,
            'Voc': calc_rev.get('Voc_calc', np.nan),
            'Jsc': calc_rev.get('Jsc_calc', np.nan),
            'FF': calc_rev.get('FF_calc', np.nan),
        }

        # Setup standard column names for curves
        loop_df['voltage(V)'] = loop_df['V']
        loop_df['avg_jcurrent(mA/cm²)'] = -loop_df['J']
        # Absolute power density (mW/cm²)
        p_multiplier = -1.0 if (loop_df['J'] < 0).any() else 1.0
        loop_df['P'] = loop_df['V'] * loop_df['J'] * p_multiplier
        loop_df['P(mw/cm²)'] = loop_df['P']

        item_name = f"{filename} (Loop {loop})" if len(unique_loops) > 1 else filename

        results.append({
            'filename': item_name,
            'df': loop_df,
            'loop': loop,
            'params': clean_params
        })

    return results


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
        if parsed is not None:
            data_list.append(parsed)
    elif txt_origin == "Renato": 
        parsed = parse_maximus(f)
        if parsed:
            if isinstance(parsed, list):
                data_list.extend(parsed)
            else:
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
        if scan_direction == "FWD":
            eff = d['params'].get('PCE_fwd', np.nan)
        elif scan_direction == "REV":
            eff = d['params'].get('PCE_rev', np.nan)
        else:
            eff_rev = d['params'].get('PCE_rev', np.nan)
            eff_fwd = d['params'].get('PCE_fwd', np.nan)
            eff = max(
                eff_rev if not np.isnan(eff_rev) else -1,
                eff_fwd if not np.isnan(eff_fwd) else -1
            )
        if not np.isnan(eff) and eff < min_efficiency:
            continue
        
    # Direction filter
    if txt_origin == "Oninn":
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
        if txt_origin == "Oninn":
            fig_jv.add_trace(go.Scatter(x=df['V'], y=df['J'], mode='lines', name=name, line=dict(color=color)))
            fig_pv.add_trace(go.Scatter(x=df['V'], y=df['P'], mode='lines', name=name, line=dict(color=color)))
        elif txt_origin == "Renato":
            df_rev = df[df['direction'].astype(str).str.lower().str.contains('rev')]
            df_fwd = df[df['direction'].astype(str).str.lower().str.contains('fwd')]

            if scan_direction in ["All", "REV"] and not df_rev.empty:
                label_rev = f"{name} (REV)" if scan_direction == "All" else name
                fig_jv.add_trace(go.Scatter(x=df_rev['voltage(V)'], y=df_rev['avg_jcurrent(mA/cm²)'], mode='lines', name=label_rev, line=dict(color=color)))
                fig_pv.add_trace(go.Scatter(x=df_rev['voltage(V)'], y=df_rev['P(mw/cm²)'], mode='lines', name=label_rev, line=dict(color=color)))

            if scan_direction in ["All", "FWD"] and not df_fwd.empty:
                label_fwd = f"{name} (FWD)" if scan_direction == "All" else name
                dash_style = 'dash' if scan_direction == "All" else 'solid'
                fig_jv.add_trace(go.Scatter(x=df_fwd['voltage(V)'], y=df_fwd['avg_jcurrent(mA/cm²)'], mode='lines', name=label_fwd, line=dict(color=color, dash=dash_style)))
                fig_pv.add_trace(go.Scatter(x=df_fwd['voltage(V)'], y=df_fwd['P(mw/cm²)'], mode='lines', name=label_fwd, line=dict(color=color, dash=dash_style)))
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
        base_cols = ["Filename", "Loop", "Voc_rev", "Voc_fwd", "Jsc_rev", "Jsc_fwd",
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
