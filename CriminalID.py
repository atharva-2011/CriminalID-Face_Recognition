import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
import time
import os
import base64
import json

# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

st.set_page_config(
    page_title="CriminalID · Face Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');

:root{
    --bg:#080b0f; --bg2:#0d1117; --bg3:#111820; --bdr:#1a2535;
    --red:#e63946; --org:#f4a261; --grn:#2ec4b6; --blu:#4361ee;
    --txt:#c8d6e5; --dim:#4a6070;
    --mono:'Share Tech Mono',monospace;
    --head:'Barlow Condensed',sans-serif;
    --body:'Rajdhani',sans-serif;
    --panel-w:300px;
}

html,body,[class*="css"]{background:var(--bg)!important;color:var(--txt)!important;font-family:var(--body)!important}
.stApp{background:var(--bg)!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 0 3rem 0!important;max-width:100%!important;margin-top:0!important}
[data-testid="stSidebar"]{display:none!important}

/* scanline */
.stApp::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.025) 2px,rgba(0,0,0,.025) 4px);pointer-events:none;z-index:100}

/* ══ FIXED CONFIG DRAWER ══════════════════════════════════════════════ */
#cfgDrawer{
    position:fixed;top:0;left:0;width:var(--panel-w);height:100vh;
    background:linear-gradient(180deg,#090d12,#060911);
    border-right:1px solid var(--bdr);
    overflow-y:auto;overflow-x:hidden;
    z-index:9000;
    transition:transform .3s cubic-bezier(.16,1,.3,1);
    padding:1rem .9rem 2rem;
    scrollbar-width:thin;
}
#cfgDrawer.closed{transform:translateX(calc(-1 * var(--panel-w)))}

/* Push main content when drawer open */
#mainWrap{
    transition:margin-left .3s cubic-bezier(.16,1,.3,1);
}
#mainWrap.shifted{margin-left:var(--panel-w)}

.cfg-title{font-family:var(--mono);font-size:.58rem;color:var(--red);letter-spacing:3px;
    border-left:3px solid var(--red);padding-left:.6rem;margin-bottom:1.1rem;display:block}
.cfg-lbl{font-family:var(--mono);font-size:.56rem;color:var(--dim);letter-spacing:2px;
    display:block;margin:.6rem 0 .2rem}
.cfg-status{border-radius:6px;padding:.4rem .7rem;font-family:var(--mono);font-size:.62rem;
    letter-spacing:.5px;margin:.3rem 0;display:block}
.cfg-ok{background:rgba(46,196,182,.08);border:1px solid rgba(46,196,182,.28);color:var(--grn)}
.cfg-err{background:rgba(230,57,70,.07);border:1px solid rgba(230,57,70,.35);color:var(--red)}
.cfg-warn{background:rgba(244,162,97,.08);border:1px solid rgba(244,162,97,.3);color:var(--org)}

/* ══ BANNER ══════════════════════════════════════════════════════════ */
.banner-wrap{
    display:flex;align-items:stretch;border-bottom:2px solid var(--red);
    background:linear-gradient(135deg,#080b0f 0%,#0c1a28 45%,#080b0f 100%);
    position:relative;overflow:hidden;
}
.banner-wrap::before{content:'';position:absolute;inset:0;
    background:repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(230,57,70,.02) 60px,rgba(230,57,70,.02) 61px);
    animation:scanx 6s linear infinite;pointer-events:none}
@keyframes scanx{to{background-position:120px 0}}

#hamburgerBtn{
    background:rgba(230,57,70,.12);border:1px solid rgba(230,57,70,.4);
    border-radius:5px;width:42px;height:42px;
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;flex-shrink:0;margin:auto .8rem;
    font-size:1.15rem;color:var(--red);
    transition:all .2s;z-index:10;
}
#hamburgerBtn:hover{background:rgba(230,57,70,.28);box-shadow:0 0 12px rgba(230,57,70,.3)}

.b-logo{font-size:2rem;animation:logop 3s ease-in-out infinite;z-index:1}
@keyframes logop{0%,100%{transform:scale(1)}50%{transform:scale(1.07);filter:drop-shadow(0 0 10px rgba(230,57,70,.7))}}
.b-title{font-family:var(--head)!important;font-size:2.3rem!important;font-weight:900!important;
    letter-spacing:6px!important;color:#fff!important;text-transform:uppercase;
    line-height:1!important;text-shadow:0 0 30px rgba(230,57,70,.35)}
.b-sub{font-family:var(--mono)!important;font-size:.58rem!important;color:var(--red)!important;
    letter-spacing:3px;text-transform:uppercase;animation:flkr 5s step-end infinite}
@keyframes flkr{0%,20%,22%,100%{opacity:1}21%{opacity:.2}}
.b-mid{display:flex;align-items:center;gap:1rem;padding:.75rem 1rem;flex:1;z-index:1}
.b-right{display:flex;flex-direction:column;align-items:flex-end;justify-content:center;
    gap:.35rem;padding:.75rem 1.5rem;z-index:1;margin-left:auto}
.b-badge{background:rgba(46,196,182,.1);border:1px solid var(--grn);border-radius:3px;
    padding:.2rem .7rem;font-family:var(--mono);font-size:.58rem;color:var(--grn);
    letter-spacing:2px;animation:blnk 1.8s step-end infinite}
@keyframes blnk{50%{opacity:.2}}
.b-time{font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:1px}
.ctlx,.cbrx{position:absolute;width:14px;height:14px}
.ctlx{top:7px;left:7px;border-top:2px solid var(--red);border-left:2px solid var(--red)}
.cbrx{bottom:7px;right:7px;border-bottom:2px solid var(--red);border-right:2px solid var(--red)}

/* ══ SECTION LABEL ═══════════════════════════════════════════════════ */
.slbl{display:block;font-family:var(--mono)!important;font-size:.58rem!important;
    letter-spacing:3px!important;color:var(--red)!important;text-transform:uppercase!important;
    border-left:3px solid var(--red);padding-left:.6rem;margin:.7rem 0 .8rem 0!important}

/* ══ PROCESSING BAR ══════════════════════════════════════════════════ */
.pbar{height:2px;background:linear-gradient(90deg,transparent,var(--red),transparent);
    background-size:200%;animation:shim .9s linear infinite;border-radius:2px;margin:.5rem 0}
@keyframes shim{0%{background-position:200%}100%{background-position:-200%}}

/* ══ RESULT PANELS ═══════════════════════════════════════════════════ */
.rf{background:linear-gradient(135deg,#120508,#1a0910);border:1px solid var(--red);
    border-radius:8px;padding:1.3rem;animation:slu .45s cubic-bezier(.16,1,.3,1);
    position:relative;overflow:hidden}
.rf::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;
    background:radial-gradient(circle at 25% 25%,rgba(230,57,70,.07),transparent 55%);pointer-events:none}
.ru{background:linear-gradient(135deg,#090e14,#0d1520);border:1px solid #1a2d3f;
    border-radius:8px;padding:1.3rem;animation:slu .45s cubic-bezier(.16,1,.3,1)}
@keyframes slu{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

/* ══ PROFILE ROWS ════════════════════════════════════════════════════ */
.prow{display:flex;align-items:center;gap:.8rem;padding:.45rem 0;
    border-bottom:1px solid rgba(26,37,53,.8);opacity:0;animation:fdr .3s ease forwards}
@keyframes fdr{to{opacity:1}}
.prow:nth-child(1){animation-delay:.05s}.prow:nth-child(2){animation-delay:.10s}
.prow:nth-child(3){animation-delay:.15s}.prow:nth-child(4){animation-delay:.20s}
.prow:nth-child(5){animation-delay:.25s}.prow:nth-child(6){animation-delay:.30s}
.prow:nth-child(7){animation-delay:.35s}.prow:nth-child(8){animation-delay:.40s}
.pk{font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:2px;
    text-transform:uppercase;min-width:105px;flex-shrink:0}
.pv{font-family:var(--body);font-size:.95rem;font-weight:600;color:var(--txt)}
.pv.cr{color:var(--red);font-size:1rem}
.pv.ds{font-size:.85rem;color:var(--dim);line-height:1.5}

/* ══ BADGES ══════════════════════════════════════════════════════════ */
.bw{background:rgba(230,57,70,.18);border:1px solid var(--red);color:var(--red)!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}
.ba{background:rgba(244,162,97,.18);border:1px solid var(--org);color:var(--org)!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}
.bi{background:rgba(46,196,182,.15);border:1px solid var(--grn);color:var(--grn)!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}

/* ══ CONFIDENCE BAR ══════════════════════════════════════════════════ */
.cbw{margin:.6rem 0}
.cbh{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.56rem;color:var(--dim);margin-bottom:4px}
.cbt{background:#0a1018;border-radius:4px;height:8px;overflow:hidden}
.cbf{height:8px;border-radius:4px;transition:width 1.2s cubic-bezier(.16,1,.3,1);position:relative}
.cbf::after{content:'';position:absolute;top:0;right:0;bottom:0;width:40px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.22));animation:csw 1.5s ease-in-out infinite}
@keyframes csw{0%,100%{opacity:0}50%{opacity:1}}

/* ══ ALERTS ══════════════════════════════════════════════════════════ */
.ad{background:rgba(230,57,70,.07);border:1px solid rgba(230,57,70,.35);border-radius:6px;
    padding:.75rem 1rem;font-family:var(--mono);font-size:.7rem;color:var(--red);letter-spacing:.7px}
.ai{background:rgba(67,97,238,.08);border:1px solid rgba(67,97,238,.3);border-radius:6px;
    padding:.75rem 1rem;font-family:var(--mono);font-size:.68rem;color:#7b9cff;letter-spacing:.4px}
.as{background:rgba(46,196,182,.08);border:1px solid rgba(46,196,182,.28);border-radius:6px;
    padding:.5rem .85rem;font-family:var(--mono);font-size:.66rem;color:var(--grn);letter-spacing:.7px}
.aw{background:rgba(244,162,97,.08);border:1px solid rgba(244,162,97,.3);border-radius:6px;
    padding:.75rem 1rem;font-family:var(--mono);font-size:.68rem;color:var(--org);letter-spacing:.4px}

/* ══ BUTTONS ═════════════════════════════════════════════════════════ */
.stButton>button{background:linear-gradient(135deg,#b91c1c,#e63946)!important;color:#fff!important;
    border:none!important;border-radius:4px!important;font-family:var(--head)!important;
    font-size:1rem!important;font-weight:700!important;letter-spacing:4px!important;
    text-transform:uppercase!important;padding:.65rem 2rem!important;width:100%!important;
    transition:all .25s!important;box-shadow:0 4px 16px rgba(230,57,70,.2)!important}
.stButton>button:hover{background:linear-gradient(135deg,#991b1b,#b91c1c)!important;
    transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(230,57,70,.45)!important}
.stButton>button:active{transform:translateY(0)!important}

/* ══ INPUTS ══════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"]{background:var(--bg3)!important;border:1px dashed var(--bdr)!important;
    border-radius:8px!important;transition:border-color .3s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(230,57,70,.3)!important}
[data-testid="stCameraInput"] video{border-radius:6px!important}
[data-testid="stCameraInput"] button{background:linear-gradient(135deg,#b91c1c,#e63946)!important;
    color:#fff!important;border:none!important;border-radius:4px!important;
    font-family:var(--head)!important;font-size:.9rem!important;font-weight:700!important;letter-spacing:3px!important}
.stTextInput input{background:var(--bg3)!important;border:1px solid var(--bdr)!important;
    color:var(--txt)!important;font-family:var(--mono)!important;font-size:.78rem!important;
    border-radius:4px!important;transition:border-color .2s!important}
.stTextInput input:focus{border-color:var(--red)!important;box-shadow:0 0 8px rgba(230,57,70,.12)!important}
.stTextInput label{font-family:var(--mono)!important;font-size:.58rem!important;
    color:var(--dim)!important;letter-spacing:2px!important}

/* ══ TABS ════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--bdr)!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--dim)!important;
    font-family:var(--head)!important;font-size:.88rem!important;letter-spacing:3px!important;
    font-weight:600!important;border-radius:0!important;padding:.75rem 1.4rem!important;
    border-bottom:2px solid transparent!important;transition:all .2s!important;cursor:pointer!important}
.stTabs [data-baseweb="tab"]:hover{color:var(--txt)!important}
.stTabs [aria-selected="true"]{background:transparent!important;color:var(--red)!important;
    border-bottom:2px solid var(--red)!important}

/* ══ SLIDER ══════════════════════════════════════════════════════════ */
.stSlider>div>div>div>div{background:var(--red)!important}

/* ══ IMAGES ══════════════════════════════════════════════════════════ */
[data-testid="stImage"] img{border-radius:6px!important;border:1px solid var(--bdr)!important;
    transition:box-shadow .3s!important}
[data-testid="stImage"] img:hover{box-shadow:0 0 20px rgba(230,57,70,.18)!important}

/* ══ METRIC BOXES ════════════════════════════════════════════════════ */
.mbox{background:var(--bg2);border:1px solid var(--bdr);border-radius:6px;padding:.9rem;
    text-align:center;position:relative;overflow:hidden;transition:transform .2s,border-color .2s;cursor:default}
.mbox:hover{transform:translateY(-2px)}
.mbox::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mval{font-family:var(--mono);font-size:1.7rem;display:block;line-height:1}
.mlbl{font-family:var(--mono);font-size:.52rem;color:var(--dim);letter-spacing:2px;
    text-transform:uppercase;margin-top:.25rem;display:block}

/* ══ DB TABLE — inside the combined modal+table iframe ═══════════════ */
.dbtw{overflow-y:auto;overflow-x:hidden;max-height:450px;border:1px solid #1a2535;
    border-radius:8px;margin-top:.8rem}
.dbt{width:100%;table-layout:fixed;border-collapse:collapse;
    font-family:'Rajdhani',sans-serif;font-size:.87rem;color:#c8d6e5}
.dbt thead{position:sticky;top:0;z-index:5;background:#08111a}
.dbt thead th{font-family:'Share Tech Mono',monospace;font-size:.53rem;letter-spacing:2px;
    color:#4a6070;text-transform:uppercase;padding:.62rem .8rem;
    border-bottom:1px solid #1a2535;text-align:left;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis}
.dbt thead th:nth-child(1){width:17%}.dbt thead th:nth-child(2){width:17%}
.dbt thead th:nth-child(3){width:13%}.dbt thead th:nth-child(4){width:6%}
.dbt thead th:nth-child(5){width:8%}.dbt thead th:nth-child(6){width:11%}
.dbt thead th:nth-child(7){width:12%}.dbt thead th:nth-child(8){width:16%}
.dbt tbody tr{border-bottom:1px solid rgba(26,37,53,.55);transition:background .15s;
    cursor:pointer;opacity:0;animation:rfad .3s ease forwards}
@keyframes rfad{to{opacity:1}}
.dbt tbody tr:hover{background:rgba(230,57,70,.07)!important}
.dbt tbody td{padding:.58rem .8rem;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;vertical-align:middle}
.dbt tbody td:first-child{color:#7b9cff;font-weight:700}
.vbtn{background:rgba(230,57,70,.1);border:1px solid rgba(230,57,70,.28);
    color:#e63946!important;padding:2px 10px;border-radius:3px;
    font-family:'Share Tech Mono',monospace;font-size:.58rem;letter-spacing:1px;
    cursor:pointer;transition:all .2s;display:inline-block;white-space:nowrap}
.vbtn:hover{background:rgba(230,57,70,.25);border-color:#e63946}

/* ══ CRIMINAL MODAL ══════════════════════════════════════════════════ */
#crimOverlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);
    z-index:99998;backdrop-filter:blur(5px)}
#crimOverlay.active{display:flex;align-items:center;justify-content:center;
    animation:fdo .2s ease}
@keyframes fdo{from{opacity:0}to{opacity:1}}
#crimBox{background:linear-gradient(145deg,#0d1117,#101a25);border:1px solid #e63946;
    border-radius:10px;width:min(680px,92vw);max-height:85vh;overflow-y:auto;
    z-index:99999;padding:2rem 2rem 1.5rem;position:relative;
    animation:mdin .35s cubic-bezier(.16,1,.3,1);
    box-shadow:0 0 60px rgba(230,57,70,.18),0 25px 50px rgba(0,0,0,.6)}
@keyframes mdin{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
#crimBox::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#e63946,rgba(230,57,70,.3),transparent);
    border-radius:10px 10px 0 0}
.mcls{position:absolute;top:1rem;right:1rem;width:28px;height:28px;
    background:rgba(230,57,70,.12);border:1px solid rgba(230,57,70,.3);
    border-radius:4px;cursor:pointer;display:flex;align-items:center;
    justify-content:center;font-size:.85rem;color:#e63946;transition:all .2s;
    font-family:'Share Tech Mono',monospace;line-height:1;user-select:none}
.mcls:hover{background:rgba(230,57,70,.28);border-color:#e63946}
.mtag{font-family:'Share Tech Mono',monospace;font-size:.54rem;color:#e63946;
    letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem}
.mname{font-family:'Barlow Condensed',sans-serif;font-size:2.2rem;font-weight:900;
    color:#fff;letter-spacing:3px;line-height:1;
    text-shadow:0 0 20px rgba(230,57,70,.3);margin-bottom:1rem}
.mfld{display:flex;align-items:flex-start;gap:.8rem;padding:.45rem 0;
    border-bottom:1px solid rgba(26,37,53,.8)}
.mkey{font-family:'Share Tech Mono',monospace;font-size:.54rem;color:#4a6070;
    letter-spacing:2px;text-transform:uppercase;min-width:115px;flex-shrink:0;padding-top:.18rem}
.mval2{font-family:'Rajdhani',sans-serif;font-size:.92rem;font-weight:600;
    color:#c8d6e5;line-height:1.45}
.mval2.mc{color:#e63946}
.mdesc{margin-top:1.1rem;background:rgba(8,11,15,.7);border:1px solid #1a2535;
    border-radius:6px;padding:.85rem 1rem;font-family:'Rajdhani',sans-serif;
    font-size:.86rem;color:#4a6070;line-height:1.7}
.mhr{border:none;border-top:1px solid #1a2535;margin:.75rem 0}
.mbw{background:rgba(230,57,70,.18);border:1px solid #e63946;color:#e63946!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;
    font-family:'Share Tech Mono',monospace}
.mba{background:rgba(244,162,97,.18);border:1px solid #f4a261;color:#f4a261!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;
    font-family:'Share Tech Mono',monospace}
.mbi{background:rgba(46,196,182,.15);border:1px solid #2ec4b6;color:#2ec4b6!important;
    padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;
    font-family:'Share Tech Mono',monospace}

/* ══ EMPTY STATE ══════════════════════════════════════════════════════ */
.emt{height:225px;display:flex;align-items:center;justify-content:center;
    flex-direction:column;gap:.75rem;background:var(--bg2);
    border:1px dashed var(--bdr);border-radius:8px;transition:border-color .3s}
.emt:hover{border-color:rgba(230,57,70,.2)}
.emi{font-size:2.1rem;opacity:.17;animation:flt 3s ease-in-out infinite}
@keyframes flt{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
.emtxt{font-family:var(--mono);font-size:.6rem;color:#1e2d3d;letter-spacing:2px;text-align:center}

/* ══ SCROLLBAR ═══════════════════════════════════════════════════════ */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(230,57,70,.4)}

hr{border:none!important;border-top:1px solid var(--bdr)!important;margin:.7rem 0!important}
.stSpinner>div{border-top-color:var(--red)!important}
.card{background:var(--bg2);border:1px solid var(--bdr);border-radius:6px;
    padding:1.1rem 1.3rem;position:relative}
.card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--red),transparent);border-radius:6px 6px 0 0}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found: {abs_path}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    last_exc = None
    try:
        import tf_keras
        return tf_keras.models.load_model(abs_path, compile=False)
    except ImportError:
        pass
    except Exception as e:
        last_exc = e
    try:
        return tf.keras.models.load_model(abs_path, compile=False)
    except Exception as e:
        last_exc = e
    try:
        return tf.keras.models.load_model(abs_path)
    except Exception as e:
        last_exc = e
    try:
        return tf.saved_model.load(abs_path)
    except Exception as e:
        last_exc = e
    raise RuntimeError(f"Could not load model. Last error: {last_exc}\nFIX: pip install tf-keras")


@st.cache_data(show_spinner=False)
def load_csv(path):
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"CSV not found: {abs_path}")
    return pd.read_csv(abs_path)


def get_model_info(mdl):
    try:
        out_shape = mdl.output_shape
        if isinstance(out_shape, list): out_shape = out_shape[-1]
        num_classes = out_shape[-1]
    except Exception:
        num_classes = "?"
    try:
        in_shape = mdl.input_shape
        if isinstance(in_shape, list): in_shape = in_shape[0]
        h = in_shape[1] if in_shape[1] not in (None, 3) else None
        w = in_shape[2] if in_shape[2] not in (None, 3) else None
        h = h or 224; w = w or 224
    except Exception:
        h, w = 224, 224
    return num_classes, int(h), int(w)


def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    for xml in ['haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml']:
        cc = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        for img_try in [gray_eq, gray]:
            faces = cc.detectMultiScale(img_try, scaleFactor=1.05, minNeighbors=3, minSize=(40,40))
            if len(faces) > 0:
                return faces
    return []


def draw_box(img, faces, label, color):
    out = img.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(out,(x,y-th-12),(x+tw+8,y),color,-1)
        cv2.putText(out,label,(x+4,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
    return out


try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mobilenet_preprocess
    _HAS_MOBILENET_PREPROCESS = True
except ImportError:
    _HAS_MOBILENET_PREPROCESS = False


def predict_face(model, crop, cnames):
    _, h, w = get_model_info(model)
    img = np.array(Image.fromarray(crop).resize((w, h))).astype(np.float32)
    if _HAS_MOBILENET_PREPROCESS:
        img = _mobilenet_preprocess(img)
    else:
        img = img / 255.0
    batch = np.expand_dims(img, 0)
    if hasattr(model, "predict"):
        preds = model.predict(batch, verbose=0)[0]
    else:
        preds = model(tf.constant(batch, dtype=tf.float32)).numpy()[0]
    idx  = int(np.argmax(preds))
    conf = float(preds[idx])
    name = cnames.get(idx, "Unknown")
    return name, conf


def sbadge(status):
    s = str(status).strip().lower()
    if 'wanted'   in s: return '<span class="bw">⚠ WANTED</span>'
    if 'arrested' in s: return '<span class="ba">⚡ ARRESTED</span>'
    if 'imprison' in s: return '<span class="bi">🔒 IMPRISONED</span>'
    return f'<span style="color:#8899aa;font-family:var(--mono);font-size:.7rem">{status}</span>'


def confbar(conf, thr):
    pct   = int(conf*100)
    col   = "#e63946" if conf>=thr else "#3a5060"
    glow  = "rgba(230,57,70,.35)" if conf>=thr else "transparent"
    label = "POSITIVE MATCH" if conf>=thr else "BELOW THRESHOLD"
    return f"""<div class="cbw">
<div class="cbh"><span>MATCH CONFIDENCE</span>
  <span style="color:{'#e63946' if conf>=thr else '#3a5060'}">{pct}%</span></div>
<div class="cbt"><div class="cbf" style="width:{pct}%;background:linear-gradient(90deg,{col}cc,{col});box-shadow:0 0 10px {glow}"></div></div>
<div style="font-family:var(--mono);font-size:.55rem;color:var(--dim);margin-top:3px">
  THRESHOLD: {int(thr*100)}% &nbsp;·&nbsp; {label}</div>
</div>"""


def render_profile(name, conf, thr, df):
    disp  = name.replace('_',' ')
    match = df[df['name'].str.strip()==name] if df is not None else pd.DataFrame()
    if conf >= thr:
        st.markdown('<div class="rf">', unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:.54rem;color:var(--red);letter-spacing:3px;margin-bottom:.5rem;animation:flkr 5s step-end infinite">
  ⚠ &nbsp;MATCH FOUND — CRIMINAL IDENTIFIED</div>
<div style="font-family:var(--head);font-size:2rem;font-weight:900;color:#fff;letter-spacing:3px;line-height:1;text-shadow:0 0 20px rgba(230,57,70,.4)">
  {disp.upper()}</div>""", unsafe_allow_html=True)
        st.markdown(confbar(conf, thr), unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        if not match.empty:
            row = match.iloc[0]
            html = ""
            for col in df.columns:
                if col == 'name': continue
                val = str(row[col]); key = col.replace('_',' ').upper()
                if   col.lower() == 'status':     v = sbadge(val)
                elif col.lower() == 'crime':       v = f'<span class="pv cr">{val}</span>'
                elif col.lower() == 'description': v = f'<span class="pv ds">{val}</span>'
                else:                              v = f'<span class="pv">{val}</span>'
                html += f'<div class="prow"><span class="pk">{key}</span>{v}</div>'
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('<div class="ai" style="margin-top:.9rem;font-size:.62rem">💡 &nbsp;Open <strong>DATABASE</strong> tab to view full records</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ai" style="margin-top:.8rem">No additional profile data in database.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ru">', unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:.54rem;color:var(--dim);letter-spacing:3px;margin-bottom:.5rem">
  🔍 &nbsp;IDENTITY UNKNOWN — NO MATCH FOUND</div>
<div style="font-family:var(--head);font-size:2rem;font-weight:900;color:#2a3a4a;letter-spacing:3px">
  UNIDENTIFIED SUBJECT</div>""", unsafe_allow_html=True)
        st.markdown(confbar(conf, thr), unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="ad">⚠ &nbsp;CONFIDENCE TOO LOW FOR POSITIVE ID<br>
<span style="font-size:.67rem;opacity:.7">Best guess: {disp} ({int(conf*100)}%) — below {int(thr*100)}% threshold</span></div>
<div style="margin-top:.9rem;font-family:var(--mono);font-size:.6rem;color:var(--dim);line-height:2">
RECOMMENDATIONS:<br>
› &nbsp;Ensure face is clearly visible and front-facing<br>
› &nbsp;Improve lighting conditions<br>
› &nbsp;Try a higher resolution photo<br>
› &nbsp;Adjust confidence threshold in settings panel
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def row_to_b64(row_dict):
    clean = {k: str(v) for k, v in row_dict.items()}
    return base64.b64encode(json.dumps(clean).encode()).decode()


def build_database_block(df, total):
    """
    Returns ONE combined HTML string: modal overlay + script + table.
    Everything in ONE st.markdown call = same iframe = JS works.
    """
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        b64    = row_to_b64(row.to_dict())
        delay  = f"{i*0.03:.2f}s"
        fname  = str(row.get('full_name', row.get('name',''))).replace('_',' ')
        crime  = str(row.get('crime','—'))
        status = str(row.get('status','—'))
        age    = str(row.get('age','—'))
        gender = str(row.get('gender','—'))
        nat    = str(row.get('nationality','—'))
        last   = str(row.get('last_seen','—'))
        s = status.lower()
        if 'wanted'   in s: sbdg = '<span class="mbw">⚠ WANTED</span>'
        elif 'arrested' in s: sbdg = '<span class="mba">⚡ ARRESTED</span>'
        elif 'imprison' in s: sbdg = '<span class="mbi">🔒 IMPRISONED</span>'
        else: sbdg = f'<span style="color:#8899aa;font-family:Share Tech Mono,monospace;font-size:.68rem">{status}</span>'
        rows_html += f'<tr data-b64="{b64}" style="animation-delay:{delay}"><td title="{fname}">{fname}</td><td title="{crime}" style="color:#e63946;font-weight:600">{crime}</td><td>{sbdg}</td><td style="color:#4a6070">{age}</td><td style="color:#4a6070">{gender}</td><td style="color:#4a6070" title="{nat}">{nat}</td><td style="color:#4a6070" title="{last}">{last}</td><td><span class="vbtn">👁 VIEW</span></td></tr>'

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#080b0f;font-family:'Rajdhani',sans-serif;color:#c8d6e5;overflow-x:hidden}}
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Barlow+Condensed:wght@700;900&display=swap');

/* TABLE */
.dbtw{{overflow-y:auto;overflow-x:hidden;border:1px solid #1a2535;border-radius:8px;margin-top:.5rem}}
table{{width:100%;table-layout:fixed;border-collapse:collapse;font-size:.87rem}}
thead{{position:sticky;top:0;z-index:5;background:#08111a}}
thead th{{font-family:'Share Tech Mono',monospace;font-size:.53rem;letter-spacing:2px;color:#4a6070;
  text-transform:uppercase;padding:.62rem .8rem;border-bottom:1px solid #1a2535;
  text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
th:nth-child(1){{width:17%}}th:nth-child(2){{width:17%}}th:nth-child(3){{width:13%}}
th:nth-child(4){{width:6%}}th:nth-child(5){{width:8%}}th:nth-child(6){{width:11%}}
th:nth-child(7){{width:12%}}th:nth-child(8){{width:16%}}
tbody tr{{border-bottom:1px solid rgba(26,37,53,.55);transition:background .15s;cursor:pointer}}
tbody tr:hover{{background:rgba(230,57,70,.07)}}
tbody td{{padding:.58rem .8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:middle;color:#c8d6e5}}
tbody td:first-child{{color:#7b9cff;font-weight:700}}
.vbtn{{background:rgba(230,57,70,.1);border:1px solid rgba(230,57,70,.28);color:#e63946;
  padding:2px 10px;border-radius:3px;font-family:'Share Tech Mono',monospace;
  font-size:.58rem;cursor:pointer;transition:all .2s;display:inline-block}}
.vbtn:hover{{background:rgba(230,57,70,.25)}}
.bw{{background:rgba(230,57,70,.18);border:1px solid #e63946;color:#e63946;
  padding:2px 8px;border-radius:3px;font-size:.65rem;font-family:'Share Tech Mono',monospace}}
.ba{{background:rgba(244,162,97,.18);border:1px solid #f4a261;color:#f4a261;
  padding:2px 8px;border-radius:3px;font-size:.65rem;font-family:'Share Tech Mono',monospace}}
.bi{{background:rgba(46,196,182,.15);border:1px solid #2ec4b6;color:#2ec4b6;
  padding:2px 8px;border-radius:3px;font-size:.65rem;font-family:'Share Tech Mono',monospace}}

/* MODAL — fixed to viewport via parent window communication */
#overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
  z-index:9999;align-items:center;justify-content:center;backdrop-filter:blur(4px)}}
#overlay.show{{display:flex}}
#box{{background:linear-gradient(145deg,#0d1117,#101a25);border:1px solid #e63946;
  border-radius:10px;width:min(660px,90vw);max-height:80vh;overflow-y:auto;
  padding:2rem;position:relative;
  box-shadow:0 0 60px rgba(230,57,70,.2),0 25px 50px rgba(0,0,0,.7);
  animation:mdin .3s cubic-bezier(.16,1,.3,1)}}
@keyframes mdin{{from{{opacity:0;transform:translateY(15px)}}to{{opacity:1;transform:translateY(0)}}}}
#box::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#e63946,rgba(230,57,70,.3),transparent);border-radius:10px 10px 0 0}}
.cls{{position:absolute;top:1rem;right:1rem;width:28px;height:28px;
  background:rgba(230,57,70,.12);border:1px solid rgba(230,57,70,.3);border-radius:4px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  font-size:.85rem;color:#e63946;font-family:'Share Tech Mono',monospace;user-select:none}}
.cls:hover{{background:rgba(230,57,70,.28)}}
.mtag{{font-family:'Share Tech Mono',monospace;font-size:.54rem;color:#e63946;
  letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem}}
.mname{{font-family:'Barlow Condensed',sans-serif;font-size:2.2rem;font-weight:900;
  color:#fff;letter-spacing:3px;line-height:1;text-shadow:0 0 20px rgba(230,57,70,.3);
  margin-bottom:1rem}}
.mhr{{border:none;border-top:1px solid #1a2535;margin:.75rem 0}}
.mfld{{display:flex;align-items:flex-start;gap:.8rem;padding:.45rem 0;
  border-bottom:1px solid rgba(26,37,53,.8)}}
.mkey{{font-family:'Share Tech Mono',monospace;font-size:.54rem;color:#4a6070;
  letter-spacing:2px;text-transform:uppercase;min-width:115px;flex-shrink:0;padding-top:.18rem}}
.mv{{font-family:'Rajdhani',sans-serif;font-size:.92rem;font-weight:600;color:#c8d6e5;line-height:1.45}}
.mv.mc{{color:#e63946}}
.mdesc{{margin-top:1rem;background:rgba(8,11,15,.7);border:1px solid #1a2535;
  border-radius:6px;padding:.85rem 1rem;font-size:.86rem;color:#4a6070;line-height:1.7}}
.footer{{font-family:'Share Tech Mono',monospace;font-size:.55rem;color:#4a6070;
  text-align:right;margin-top:.5rem}}
</style>
</head>
<body>

<!-- MODAL -->
<div id="overlay">
  <div id="box">
    <div class="cls" id="cls">✕</div>
    <div id="content"></div>
  </div>
</div>

<!-- TABLE -->
<div class="dbtw">
<table>
<thead><tr>
  <th>NAME</th><th>CRIME</th><th>STATUS</th>
  <th>AGE</th><th>GENDER</th><th>NATIONALITY</th>
  <th>LAST SEEN</th><th>PROFILE</th>
</tr></thead>
<tbody id="tbody">{rows_html}</tbody>
</table>
</div>
<div class="footer">SHOWING {len(df)} OF {total} RECORDS &nbsp;·&nbsp; CLICK ANY ROW OR 👁 VIEW</div>

<script>
var overlay=document.getElementById('overlay');
var content=document.getElementById('content');

function esc(s){{var d=document.createElement('div');d.textContent=s||'—';return d.innerHTML;}}
function badge(s){{
  s=(s||'').toLowerCase();
  if(s.indexOf('wanted')>=0)  return '<span class="bw">⚠ WANTED</span>';
  if(s.indexOf('arrested')>=0)return '<span class="ba">⚡ ARRESTED</span>';
  if(s.indexOf('imprison')>=0)return '<span class="bi">🔒 IMPRISONED</span>';
  return '<span style="color:#8899aa">'+esc(s)+'</span>';
}}

function openModal(b64){{
  var data=JSON.parse(atob(b64));
  var html='<div class="mtag">⚠ CRIMINAL RECORD — CONFIDENTIAL</div>';
  html+='<div class="mname">'+esc(data.full_name||data.name||'UNKNOWN').toUpperCase()+'</div>';
  html+='<hr class="mhr">';
  [['FULL NAME',data.full_name||data.name||'—',''],
   ['CRIME',data.crime||'—','mc'],
   ['STATUS',data.status||'—','status'],
   ['AGE',data.age||'—',''],
   ['GENDER',data.gender||'—',''],
   ['NATIONALITY',data.nationality||'—',''],
   ['LAST SEEN',data.last_seen||'—','']
  ].forEach(function(f){{
    var v = f[2]==='status' ? badge(f[1]) :
            f[2]==='mc'     ? '<span class="mv mc">'+esc(f[1])+'</span>' :
                              '<span class="mv">'+esc(f[1])+'</span>';
    html+='<div class="mfld"><span class="mkey">'+f[0]+'</span>'+v+'</div>';
  }});
  if(data.description&&data.description!=='—'){{
    html+='<div class="mdesc"><small style="font-family:monospace;font-size:.5rem;color:#4a6070;letter-spacing:2px;display:block;margin-bottom:.3rem">CASE NOTES</small>'+esc(data.description)+'</div>';
  }}
  content.innerHTML=html;
  overlay.classList.add('show');
  document.body.style.overflow='hidden';
}}

function closeModal(){{
  overlay.classList.remove('show');
  document.body.style.overflow='';
}}

document.getElementById('cls').onclick=closeModal;
overlay.onclick=function(e){{if(e.target===overlay)closeModal();}};
document.addEventListener('keydown',function(e){{if(e.key==='Escape')closeModal();}});

// Event delegation on tbody
document.getElementById('tbody').addEventListener('click',function(e){{
  var tr=e.target.closest('tr[data-b64]');
  if(tr) openModal(tr.getAttribute('data-b64'));
}});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
#  BANNER — pure HTML div, no Streamlit columns for this row
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div id="mainWrap" class="{'shifted' if st.session_state.sidebar_open else ''}">
<div class="banner-wrap">
  <div class="ctlx"></div><div class="cbrx"></div>
  <div id="hamburgerBtn" onclick="toggleDrawer()">☰</div>
  <div class="b-mid">
    <div class="b-logo">🔍</div>
    <div>
      <div class="b-title">CriminalID</div>
      <div class="b-sub">Automated Face Recognition &amp; Identification System</div>
    </div>
  </div>
  <div class="b-right">
    <div class="b-badge">● SYSTEM ACTIVE</div>
    <div class="b-time">{time.strftime('%Y-%m-%d  %H:%M:%S')}</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  CONFIG DRAWER — fixed position HTML panel (not a Streamlit column)
# ══════════════════════════════════════════════════════════════════════════
# Build status strings first (Python side), inject into fixed HTML drawer
_script_dir    = os.path.dirname(os.path.abspath(__file__))

# Default values
model_loaded    = False
model           = None
info_df         = None
class_names_map = {}
threshold       = 0.80
model_path      = "criminal_recognition_model.keras"
csv_path        = "criminals_info.csv"
cn_path         = "class_names.txt"

# ── Use st.sidebar for the interactive controls (hidden visually but
#    functional — Streamlit widgets need a container) ──
# Actually we use a collapsed expander trick: put controls in sidebar
# which is hidden, but since sidebar is collapsed we use session_state
# to store path values. Better: use st.columns with width 0 trick.
# CORRECT APPROACH: Use a narrow hidden column for widget state,
# render the visual drawer via HTML only.

# Since Streamlit requires widgets to be in the render tree,
# we put them in a visually hidden div using CSS:
st.markdown("""
<style>
/* Hide the widget column visually but keep it in DOM for state */
div[data-testid="stHorizontalBlock"]:has(#widget-anchor){display:none!important}
</style>
<div id="widget-anchor"></div>
""", unsafe_allow_html=True)

# Put all interactive widgets in the main flow but visually hidden
with st.expander("⚙ CONFIG", expanded=st.session_state.sidebar_open):
    def slbl(t):
        st.markdown(f'<span style="font-family:var(--mono);font-size:.56rem;color:var(--dim);letter-spacing:2px">{t}</span>', unsafe_allow_html=True)

    slbl("MODEL PATH")
    model_path = st.text_input("mp", value="criminal_recognition_model.keras", label_visibility="collapsed")
    slbl("DATABASE CSV")
    csv_path   = st.text_input("cp", value="criminals_info.csv", label_visibility="collapsed")
    slbl("CLASS NAMES FILE")
    cn_path    = st.text_input("cn", value="class_names.txt", label_visibility="collapsed")
    slbl("CONFIDENCE THRESHOLD")
    threshold  = st.slider("thr", 0.50, 0.99, 0.80, 0.01, label_visibility="collapsed")

    resolved_model = model_path if os.path.isabs(model_path) else os.path.join(_script_dir, model_path)
    resolved_csv   = csv_path   if os.path.isabs(csv_path)   else os.path.join(_script_dir, csv_path)
    resolved_cn    = cn_path    if os.path.isabs(cn_path)    else os.path.join(_script_dir, cn_path)

    if os.path.exists(resolved_model):
        try:
            model = load_model(resolved_model)
            num_classes, in_h, in_w = get_model_info(model)
            model_loaded = True
            model_status = f'<span class="cfg-status cfg-ok">✓ MODEL · {num_classes} cls · {in_h}×{in_w}</span>'
        except Exception as e:
            err_msg = str(e)[:120].replace('<','&lt;').replace('>','&gt;')
            is_mm   = any(k in str(e) for k in ["input_layer","functional_","incompatible"])
            hint    = ' · pip install tf-keras' if is_mm else ''
            model_status = f'<span class="cfg-status cfg-err">✗ MODEL ERROR: {err_msg}{hint}</span>'
    else:
        model_status = f'<span class="cfg-status cfg-err">✗ MODEL NOT FOUND: {os.path.basename(resolved_model)}</span>'

    if os.path.exists(resolved_csv):
        try:
            info_df = load_csv(resolved_csv)
            csv_status = f'<span class="cfg-status cfg-ok">✓ DATABASE · {len(info_df)} records</span>'
        except Exception as e:
            csv_status = f'<span class="cfg-status cfg-err">✗ CSV ERROR: {str(e)[:60]}</span>'
    else:
        csv_status = '<span class="cfg-status cfg-err">✗ CSV NOT FOUND</span>'

    if os.path.exists(resolved_cn):
        with open(resolved_cn) as f:
            names = [l.strip() for l in f if l.strip()]
        class_names_map = {i: n for i, n in enumerate(names)}
        cn_status = f'<span class="cfg-status cfg-ok">✓ CLASSES · {len(class_names_map)}</span>'
    elif model_loaded and info_df is not None:
        names = list(info_df['name'].str.strip())
        class_names_map = {i: n for i, n in enumerate(names)}
        cn_status = '<span class="cfg-status cfg-warn">⚠ Using CSV order as classes</span>'
    else:
        cn_status = '<span class="cfg-status cfg-err">✗ CLASS NAMES NOT FOUND</span>'

    col_r, col_c = st.columns(2)
    with col_r:
        if st.button("🔄 RELOAD", key="reload_mdl"):
            load_model.clear(); load_csv.clear(); st.rerun()
    with col_c:
        if st.button("🗑 CLEAR", key="clr"):
            for k in ['up_res','cam_res']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

# ── Inject the fixed visual drawer with status from Python ──
st.markdown(f"""
<div id="cfgDrawer" class="{'closed' if not st.session_state.sidebar_open else ''}">
  <div style="margin-bottom:1.2rem">
    <span class="cfg-title">⚙ SYSTEM CONFIGURATION</span>
  </div>
  <span class="cfg-lbl">MODEL PATH</span>
  <div style="font-family:var(--mono);font-size:.65rem;color:var(--txt);
       background:var(--bg3);border:1px solid var(--bdr);border-radius:4px;
       padding:.35rem .6rem;margin-bottom:.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
    {model_path}
  </div>
  <span class="cfg-lbl">DATABASE CSV</span>
  <div style="font-family:var(--mono);font-size:.65rem;color:var(--txt);
       background:var(--bg3);border:1px solid var(--bdr);border-radius:4px;
       padding:.35rem .6rem;margin-bottom:.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
    {csv_path}
  </div>
  <span class="cfg-lbl">CLASS NAMES FILE</span>
  <div style="font-family:var(--mono);font-size:.65rem;color:var(--txt);
       background:var(--bg3);border:1px solid var(--bdr);border-radius:4px;
       padding:.35rem .6rem;margin-bottom:.4rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
    {cn_path}
  </div>
  <span class="cfg-lbl">CONFIDENCE THRESHOLD</span>
  <div style="font-family:var(--mono);font-size:.8rem;color:var(--red);
       text-align:center;padding:.3rem 0;font-weight:700">
    {threshold:.2f} &nbsp;·&nbsp; MIN {int(threshold*100)}% FOR POSITIVE ID
  </div>
  <hr style="border-top:1px solid var(--bdr);margin:.7rem 0">
  {model_status}
  {csv_status}
  {cn_status}
  <hr style="border-top:1px solid var(--bdr);margin:.7rem 0">
  <div style="font-family:var(--mono);font-size:.48rem;color:#172030;
       text-align:center;line-height:2.2;margin-top:.5rem">
    CRIMINALID v2.0 · FACE RECOGNITION SYSTEM<br>TensorFlow · OpenCV · Streamlit
  </div>
  <div style="margin-top:.6rem;font-family:var(--mono);font-size:.52rem;color:var(--dim);
       text-align:center">
    Use the ⚙ CONFIG expander below to edit paths &amp; threshold
  </div>
</div>
<script>
// Script runs AFTER cfgDrawer exists in DOM
(function(){{
  var _open = {'true' if st.session_state.sidebar_open else 'false'};
  var _drawer, _wrap;
  function _get(){{
    _drawer = document.getElementById('cfgDrawer');
    _wrap   = document.getElementById('mainWrap');
  }}
  _get();
  window.toggleDrawer = function(){{
    _get();
    _open = !_open;
    if(_drawer) _drawer.classList.toggle('closed', !_open);
    if(_wrap)   _wrap.classList.toggle('shifted', _open);
  }};
  // Hamburger button onclick may fire before this script runs — attach listener too
  var hbtn = document.getElementById('hamburgerBtn');
  if(hbtn) hbtn.onclick = window.toggleDrawer;
}})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div style="padding:0 1.2rem">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "  📁  UPLOAD IMAGE  ",
    "  📷  WEBCAM  ",
    "  📊  DATABASE  "
])

# ── TAB 1 — UPLOAD ──
with tab1:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    L, R = st.columns([1,1], gap="large")

    with L:
        st.markdown('<span class="slbl">INPUT — UPLOAD SUSPECT IMAGE</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop image here", type=["jpg","jpeg","png","webp"], key="upl")

        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(img_pil)

            if st.button("⚡  RUN IDENTIFICATION", key="run1"):
                if not model_loaded:
                    st.markdown('<div class="ad">⚠ &nbsp;MODEL NOT LOADED — open ⚙ CONFIG panel</div>', unsafe_allow_html=True)
                elif not class_names_map:
                    st.markdown('<div class="ad">⚠ &nbsp;CLASS NAMES MISSING</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("Analyzing image..."):
                        st.markdown('<div class="pbar"></div>', unsafe_allow_html=True)
                        time.sleep(0.3)
                        faces = detect_face(img_rgb)
                    if len(faces) == 0:
                        st.markdown('<div class="ad">❌ &nbsp;NO FACE DETECTED<br><span style="font-size:.68rem;opacity:.7">Ensure face is clearly visible and well-lit.</span></div>', unsafe_allow_html=True)
                        st.session_state['up_res'] = (img_rgb, None, 0.0)
                    else:
                        x,y,w,h = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)[0]
                        pad = int(0.15*w)
                        x1=max(0,x-pad); y1=max(0,y-pad)
                        x2=min(img_rgb.shape[1],x+w+pad); y2=min(img_rgb.shape[0],y+h+pad)
                        crop = img_rgb[y1:y2,x1:x2]
                        name, conf = predict_face(model, crop, class_names_map)
                        c   = (230,57,70) if conf>=threshold else (74,96,112)
                        lbl = f"{name.replace('_',' ')}  {int(conf*100)}%" if conf>=threshold else f"Unknown  {int(conf*100)}%"
                        st.session_state['up_res'] = (draw_box(img_rgb,[(x,y,w,h)],lbl,c), name, conf)

            if 'up_res' in st.session_state:
                st.image(st.session_state['up_res'][0], width=250)
            else:
                st.image(img_rgb, width=250)
        else:
            st.markdown('<div class="emt"><div class="emi">📷</div><div class="emtxt">AWAITING INPUT IMAGE<br><span style="font-size:.52rem;opacity:.45">JPG · JPEG · PNG · WEBP</span></div></div>', unsafe_allow_html=True)

    with R:
        st.markdown('<span class="slbl">OUTPUT — IDENTIFICATION RESULT</span>', unsafe_allow_html=True)
        if 'up_res' in st.session_state:
            _, name, conf = st.session_state['up_res']
            if name:
                render_profile(name, conf, threshold, info_df)
            else:
                st.markdown('<div class="ad">❌ &nbsp;No face detected — cannot identify</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="emt"><div class="emi">🧾</div><div class="emtxt">RESULT WILL APPEAR HERE<br><span style="font-size:.52rem;opacity:.45">Upload an image and click Run</span></div></div>', unsafe_allow_html=True)

# ── TAB 2 — WEBCAM ──
with tab2:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ai" style="margin-bottom:1rem">📷 &nbsp;Click <strong>Allow</strong> when your browser asks for camera access &nbsp;·&nbsp; Then click <strong>Take Photo</strong> to identify automatically</div>', unsafe_allow_html=True)

    L2, R2 = st.columns([1,1], gap="large")
    with L2:
        st.markdown('<span class="slbl">INPUT — LIVE WEBCAM CAPTURE</span>', unsafe_allow_html=True)
        cam = st.camera_input("cam", label_visibility="collapsed")

        if cam:
            img_pil = Image.open(cam).convert("RGB")
            img_rgb = np.array(img_pil)
            if not model_loaded:
                st.markdown('<div class="ad">⚠ &nbsp;MODEL NOT LOADED — open ⚙ CONFIG panel</div>', unsafe_allow_html=True)
            elif not class_names_map:
                st.markdown('<div class="ad">⚠ &nbsp;CLASS NAMES MISSING</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing..."):
                    st.markdown('<div class="pbar"></div>', unsafe_allow_html=True)
                    time.sleep(0.2)
                    faces = detect_face(img_rgb)
                if len(faces) == 0:
                    st.markdown('<div class="ad">❌ &nbsp;NO FACE DETECTED — try different angle or lighting</div>', unsafe_allow_html=True)
                    st.session_state['cam_res'] = None
                else:
                    x,y,w,h = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)[0]
                    pad = int(0.15*w)
                    x1=max(0,x-pad); y1=max(0,y-pad)
                    x2=min(img_rgb.shape[1],x+w+pad); y2=min(img_rgb.shape[0],y+h+pad)
                    crop = img_rgb[y1:y2,x1:x2]
                    name, conf = predict_face(model, crop, class_names_map)
                    st.markdown(f"<div style='font-family:monospace;color:#7b9cff;font-size:.7rem'>RAW CONFIDENCE: {round(conf,4)}</div>", unsafe_allow_html=True)
                    c   = (230,57,70) if conf>=threshold else (74,96,112)
                    lbl = f"{name.replace('_',' ')}  {int(conf*100)}%" if conf>=threshold else f"Unknown  {int(conf*100)}%"
                    st.image(draw_box(img_rgb,[(x,y,w,h)],lbl,c), use_container_width=True)
                    st.session_state['cam_res'] = (name, conf)

    with R2:
        st.markdown('<span class="slbl">OUTPUT — IDENTIFICATION RESULT</span>', unsafe_allow_html=True)
        if st.session_state.get('cam_res'):
            name, conf = st.session_state['cam_res']
            render_profile(name, conf, threshold, info_df)
        else:
            st.markdown('<div class="emt"><div class="emi">🎯</div><div class="emtxt">AWAITING WEBCAM CAPTURE<br><span style="font-size:.52rem;opacity:.45">Take a photo to run identification</span></div></div>', unsafe_allow_html=True)

# ── TAB 3 — DATABASE ──
with tab3:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<span class="slbl">CRIMINAL DATABASE — ALL RECORDS</span>', unsafe_allow_html=True)

    if info_df is not None:
        total      = len(info_df)
        wanted     = len(info_df[info_df['status'].str.lower().str.contains('wanted',   na=False)]) if 'status' in info_df.columns else 0
        arrested   = len(info_df[info_df['status'].str.lower().str.contains('arrested', na=False)]) if 'status' in info_df.columns else 0
        imprisoned = len(info_df[info_df['status'].str.lower().str.contains('imprison', na=False)]) if 'status' in info_df.columns else 0

        c1,c2,c3,c4 = st.columns(4)
        for col_ui, val, lbl, col_hex in [
            (c1, total,      "TOTAL RECORDS", "#4361ee"),
            (c2, wanted,     "WANTED",        "#e63946"),
            (c3, arrested,   "ARRESTED",      "#f4a261"),
            (c4, imprisoned, "IMPRISONED",    "#2ec4b6"),
        ]:
            col_ui.markdown(f"""
<div class="mbox" style="border-color:{col_hex}22">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{col_hex}"></div>
  <span class="mval" style="color:{col_hex}">{val}</span>
  <span class="mlbl">{lbl}</span>
</div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
        search = st.text_input("🔎  SEARCH RECORDS", placeholder="Filter by name, crime, status...", key="dbs", label_visibility="visible")
        filtered = info_df
        if search:
            mask = info_df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
            filtered = info_df[mask]

        # ── KEY: components.html = true iframe, JS works 100%, modal works ──
        import streamlit.components.v1 as components
        row_count = len(filtered)
        iframe_h  = min(600, max(300, 120 + row_count * 42))
        components.html(build_database_block(filtered, total), height=iframe_h, scrolling=False)

    else:
        st.markdown('<div class="ad">⚠ &nbsp;DATABASE NOT LOADED<br><span style="font-size:.68rem;opacity:.7">Open ⚙ CONFIG panel and set the CSV path.</span></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card" style="margin-top:1.2rem">
  <div style="font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:2px;margin-bottom:.8rem">REQUIRED CSV FORMAT</div>
  <div style="font-family:var(--mono);font-size:.68rem;color:#3a5060;line-height:2.2;background:#080b0f;padding:1rem;border-radius:4px">
    name, full_name, crime, status, age, gender, nationality, last_seen, description<br>
    John_Doe, John Doe, Armed Robbery, Wanted, 34, Male, Indian, Mumbai 2024, Dangerous armed robber
  </div>
  <div style="font-family:var(--mono);font-size:.58rem;color:#2a3a4a;margin-top:.7rem">
    ⚠ The 'name' column must exactly match your dataset folder names
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
