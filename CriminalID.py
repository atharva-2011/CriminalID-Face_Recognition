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

st.set_page_config(
    page_title="CriminalID · Face Recognition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');

:root {
    --bg:     #080b0f;
    --bg2:    #0d1117;
    --bg3:    #111820;
    --bdr:    #1a2535;
    --red:    #e63946;
    --org:    #f4a261;
    --grn:    #2ec4b6;
    --blu:    #4361ee;
    --txt:    #c8d6e5;
    --dim:    #4a6070;
    --mono:   'Share Tech Mono', monospace;
    --head:   'Barlow Condensed', sans-serif;
    --body:   'Rajdhani', sans-serif;
}

/* ── Global ── */
html,body,[class*="css"]{background:var(--bg)!important;color:var(--txt)!important;font-family:var(--body)!important}
.stApp{background:var(--bg)!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 1.5rem 3rem!important;max-width:100%!important}

/* scanline */
.stApp::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.025) 2px,rgba(0,0,0,.025) 4px);pointer-events:none;z-index:9999}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#090d12,#060911)!important;border-right:1px solid var(--bdr)!important}
[data-testid="stSidebar"] *{color:var(--txt)!important}
[data-testid="stSidebar"] input{background:var(--bg3)!important;border:1px solid var(--bdr)!important;color:var(--txt)!important;font-family:var(--mono)!important;font-size:.72rem!important;border-radius:4px!important;cursor:text!important}
[data-testid="stSidebar"] input:focus{border-color:var(--red)!important;outline:none!important}
[data-testid="stSidebar"] .stSlider{cursor:pointer!important}
[data-testid="stSidebar"] .stSlider *{cursor:pointer!important}

/* Fix Streamlit overriding button cursor */
.stButton > button{cursor:pointer!important}
.stButton > button *{cursor:pointer!important}

/* Fix file uploader cursor */
[data-testid="stFileUploader"] label{cursor:pointer!important}
[data-testid="stFileUploader"] section{cursor:pointer!important}

/* Fix tab cursor */
.stTabs [data-baseweb="tab"]{cursor:pointer!important}

/* ── Banner ── */
.banner{background:linear-gradient(135deg,#080b0f 0%,#0c1a28 45%,#080b0f 100%);border-bottom:2px solid var(--red);padding:1.1rem 2rem;display:flex;align-items:center;gap:1.2rem;position:relative;overflow:hidden}
.banner::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(230,57,70,.022) 60px,rgba(230,57,70,.022) 61px);animation:scanx 6s linear infinite}
@keyframes scanx{to{background-position:120px 0}}
.b-logo{font-size:2.1rem;z-index:1;animation:logop 3s ease-in-out infinite}
@keyframes logop{0%,100%{transform:scale(1)}50%{transform:scale(1.07);filter:drop-shadow(0 0 10px rgba(230,57,70,.7))}}
.b-title{font-family:var(--head)!important;font-size:2.5rem!important;font-weight:900!important;letter-spacing:6px!important;color:#fff!important;text-transform:uppercase;line-height:1!important;z-index:1;text-shadow:0 0 30px rgba(230,57,70,.35)}
.b-sub{font-family:var(--mono)!important;font-size:.6rem!important;color:var(--red)!important;letter-spacing:3px;text-transform:uppercase;z-index:1;animation:flkr 5s step-end infinite}
@keyframes flkr{0%,20%,22%,100%{opacity:1}21%{opacity:.2}}
.b-right{margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:.4rem;z-index:1}
.b-badge{background:rgba(46,196,182,.1);border:1px solid var(--grn);border-radius:3px;padding:.2rem .7rem;font-family:var(--mono);font-size:.58rem;color:var(--grn);letter-spacing:2px;animation:blnk 1.8s step-end infinite}
@keyframes blnk{50%{opacity:.2}}
.b-time{font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:1px}
.ctlx,.cbrx{position:absolute;width:14px;height:14px;z-index:2}
.ctlx{top:7px;left:7px;border-top:2px solid var(--red);border-left:2px solid var(--red)}
.cbrx{bottom:7px;right:7px;border-bottom:2px solid var(--red);border-right:2px solid var(--red)}

/* ── Section label ── */
.slbl{display:block;font-family:var(--mono)!important;font-size:.58rem!important;letter-spacing:3px!important;color:var(--red)!important;text-transform:uppercase!important;border-left:3px solid var(--red);padding-left:.6rem;margin:.7rem 0 .8rem 0!important}

/* ── Cards ── */
.card{background:var(--bg2);border:1px solid var(--bdr);border-radius:6px;padding:1.1rem 1.3rem;position:relative;transition:border-color .3s}
.card:hover{border-color:rgba(230,57,70,.25)}
.card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--red),transparent);border-radius:6px 6px 0 0}

/* ── Processing bar ── */
.pbar{height:2px;background:linear-gradient(90deg,transparent,var(--red),transparent);background-size:200%;animation:shim .9s linear infinite;border-radius:2px;margin:.5rem 0}
@keyframes shim{0%{background-position:200%}100%{background-position:-200%}}

/* ── Result panels ── */
.rf{background:linear-gradient(135deg,#120508,#1a0910);border:1px solid var(--red);border-radius:8px;padding:1.3rem;animation:slu .45s cubic-bezier(.16,1,.3,1);position:relative;overflow:hidden}
.rf::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle at 25% 25%,rgba(230,57,70,.07),transparent 55%);pointer-events:none}
.ru{background:linear-gradient(135deg,#090e14,#0d1520);border:1px solid #1a2d3f;border-radius:8px;padding:1.3rem;animation:slu .45s cubic-bezier(.16,1,.3,1)}
@keyframes slu{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}

/* ── Profile rows ── */
.prow{display:flex;align-items:center;gap:.8rem;padding:.45rem 0;border-bottom:1px solid rgba(26,37,53,.8);opacity:0;animation:fdr .3s ease forwards}
@keyframes fdr{to{opacity:1}}
.prow:nth-child(1){animation-delay:.05s}.prow:nth-child(2){animation-delay:.10s}
.prow:nth-child(3){animation-delay:.15s}.prow:nth-child(4){animation-delay:.20s}
.prow:nth-child(5){animation-delay:.25s}.prow:nth-child(6){animation-delay:.30s}
.prow:nth-child(7){animation-delay:.35s}.prow:nth-child(8){animation-delay:.40s}
.pk{font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase;min-width:105px;flex-shrink:0}
.pv{font-family:var(--body);font-size:.95rem;font-weight:600;color:var(--txt)}
.pv.cr{color:var(--red);font-size:1rem}
.pv.ds{font-size:.85rem;color:var(--dim);line-height:1.5}

/* Badges */
.bw{background:rgba(230,57,70,.18);border:1px solid var(--red);color:var(--red)!important;padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}
.ba{background:rgba(244,162,97,.18);border:1px solid var(--org);color:var(--org)!important;padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}
.bi{background:rgba(46,196,182,.15);border:1px solid var(--grn);color:var(--grn)!important;padding:2px 9px;border-radius:3px;font-size:.67rem;letter-spacing:1px;font-family:var(--mono)}

/* ── Confidence bar ── */
.cbw{margin:.6rem 0}
.cbh{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.56rem;color:var(--dim);margin-bottom:4px}
.cbt{background:#0a1018;border-radius:4px;height:8px;overflow:hidden}
.cbf{height:8px;border-radius:4px;transition:width 1.2s cubic-bezier(.16,1,.3,1);position:relative}
.cbf::after{content:'';position:absolute;top:0;right:0;bottom:0;width:40px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.22));animation:csw 1.5s ease-in-out infinite}
@keyframes csw{0%,100%{opacity:0}50%{opacity:1}}

/* ── Alerts ── */
.ad{background:rgba(230,57,70,.07);border:1px solid rgba(230,57,70,.35);border-radius:6px;padding:.75rem 1rem;font-family:var(--mono);font-size:.7rem;color:var(--red);letter-spacing:.7px}
.ai{background:rgba(67,97,238,.08);border:1px solid rgba(67,97,238,.3);border-radius:6px;padding:.75rem 1rem;font-family:var(--mono);font-size:.68rem;color:#7b9cff;letter-spacing:.4px}
.as{background:rgba(46,196,182,.08);border:1px solid rgba(46,196,182,.28);border-radius:6px;padding:.5rem .85rem;font-family:var(--mono);font-size:.66rem;color:var(--grn);letter-spacing:.7px}
.aw{background:rgba(244,162,97,.08);border:1px solid rgba(244,162,97,.3);border-radius:6px;padding:.75rem 1rem;font-family:var(--mono);font-size:.68rem;color:var(--org);letter-spacing:.4px}

/* ── Buttons ── */
.stButton>button{background:linear-gradient(135deg,#b91c1c,#e63946)!important;color:#fff!important;border:none!important;border-radius:4px!important;font-family:var(--head)!important;font-size:1rem!important;font-weight:700!important;letter-spacing:4px!important;text-transform:uppercase!important;padding:.65rem 2rem!important;width:100%!important;transition:all .25s!important;box-shadow:0 4px 16px rgba(230,57,70,.2)!important}
.stButton>button:hover{background:linear-gradient(135deg,#991b1b,#b91c1c)!important;transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(230,57,70,.45)!important}
.stButton>button:active{transform:translateY(0)!important}

/* ── Inputs ── */
[data-testid="stFileUploader"]{background:var(--bg3)!important;border:1px dashed var(--bdr)!important;border-radius:8px!important;transition:border-color .3s!important}
[data-testid="stFileUploader"]:hover{border-color:rgba(230,57,70,.3)!important}
[data-testid="stCameraInput"] video{border-radius:6px!important}
[data-testid="stCameraInput"] button{background:linear-gradient(135deg,#b91c1c,#e63946)!important;color:#fff!important;border:none!important;border-radius:4px!important;font-family:var(--head)!important;font-size:.9rem!important;font-weight:700!important;letter-spacing:3px!important}
.stTextInput input{background:var(--bg3)!important;border:1px solid var(--bdr)!important;color:var(--txt)!important;font-family:var(--mono)!important;font-size:.78rem!important;border-radius:4px!important;transition:border-color .2s!important}
.stTextInput input:focus{border-color:var(--red)!important;box-shadow:0 0 8px rgba(230,57,70,.12)!important}
.stTextInput label{font-family:var(--mono)!important;font-size:.58rem!important;color:var(--dim)!important;letter-spacing:2px!important}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--bdr)!important;gap:0!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--dim)!important;font-family:var(--head)!important;font-size:.88rem!important;letter-spacing:3px!important;font-weight:600!important;border-radius:0!important;padding:.75rem 1.4rem!important;border-bottom:2px solid transparent!important;transition:all .2s!important}
.stTabs [data-baseweb="tab"]:hover{color:var(--txt)!important}
.stTabs [aria-selected="true"]{background:transparent!important;color:var(--red)!important;border-bottom:2px solid var(--red)!important}

/* ── Slider ── */
.stSlider>div>div>div>div{background:var(--red)!important}

/* ── Images ── */
[data-testid="stImage"] img{border-radius:6px!important;border:1px solid var(--bdr)!important;transition:box-shadow .3s!important}
[data-testid="stImage"] img:hover{box-shadow:0 0 20px rgba(230,57,70,.18)!important}

/* ── Metric boxes ── */
.mbox{background:var(--bg2);border:1px solid var(--bdr);border-radius:6px;padding:.9rem;text-align:center;position:relative;overflow:hidden;transition:transform .2s,border-color .2s;cursor:default}
.mbox:hover{transform:translateY(-2px)}
.mbox::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.mval{font-family:var(--mono);font-size:1.7rem;display:block;line-height:1}
.mlbl{font-family:var(--mono);font-size:.52rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-top:.25rem;display:block}

/* ── DB TABLE ── */
.dbtw{overflow-y:auto;overflow-x:hidden;max-height:450px;border:1px solid var(--bdr);border-radius:8px;margin-top:.8rem}
.dbt{width:100%;table-layout:fixed;border-collapse:collapse;font-family:var(--body);font-size:.87rem}
.dbt thead{position:sticky;top:0;z-index:5;background:#08111a}
.dbt thead th{font-family:var(--mono);font-size:.53rem;letter-spacing:2px;color:var(--dim);text-transform:uppercase;padding:.62rem .8rem;border-bottom:1px solid var(--bdr);text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dbt thead th:nth-child(1){width:17%}.dbt thead th:nth-child(2){width:17%}.dbt thead th:nth-child(3){width:13%}
.dbt thead th:nth-child(4){width:6%}.dbt thead th:nth-child(5){width:8%}.dbt thead th:nth-child(6){width:11%}
.dbt thead th:nth-child(7){width:12%}.dbt thead th:nth-child(8){width:16%}
.dbt tbody tr{border-bottom:1px solid rgba(26,37,53,.55);transition:background .15s;cursor:pointer;opacity:0;animation:rfad .3s ease forwards}
@keyframes rfad{to{opacity:1}}
.dbt tbody tr:hover{background:rgba(230,57,70,.06)!important}
.dbt tbody td{padding:.58rem .8rem;color:var(--txt);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:middle}
.dbt tbody td:first-child{color:#7b9cff;font-weight:700}
.vbtn{background:rgba(230,57,70,.1);border:1px solid rgba(230,57,70,.28);color:var(--red)!important;padding:2px 10px;border-radius:3px;font-family:var(--mono);font-size:.58rem;letter-spacing:1px;cursor:pointer;transition:all .2s;display:inline-block;white-space:nowrap}
.vbtn:hover{background:rgba(230,57,70,.25);border-color:var(--red)}

/* ── MODAL ── */
#crimOverlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);z-index:99998;backdrop-filter:blur(5px);animation:fdo .2s ease}
@keyframes fdo{from{opacity:0}to{opacity:1}}
#crimBox{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:linear-gradient(145deg,#0d1117,#101a25);border:1px solid var(--red);border-radius:10px;width:min(700px,92vw);max-height:87vh;overflow-y:auto;z-index:99999;padding:2rem 2rem 1.5rem;animation:mdin .35s cubic-bezier(.16,1,.3,1);box-shadow:0 0 60px rgba(230,57,70,.18),0 25px 50px rgba(0,0,0,.6)}
@keyframes mdin{from{opacity:0;transform:translate(-50%,-44%)}to{opacity:1;transform:translate(-50%,-50%)}}
#crimBox::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--red),rgba(230,57,70,.3),transparent);border-radius:10px 10px 0 0}
.mcls{position:absolute;top:1rem;right:1rem;width:28px;height:28px;background:rgba(230,57,70,.12);border:1px solid rgba(230,57,70,.3);border-radius:4px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.85rem;color:var(--red);transition:all .2s;font-family:var(--mono);line-height:1;user-select:none}
.mcls:hover{background:rgba(230,57,70,.28);border-color:var(--red)}
.mtag{font-family:var(--mono);font-size:.54rem;color:var(--red);letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem}
.mname{font-family:var(--head);font-size:2.2rem;font-weight:900;color:#fff;letter-spacing:3px;line-height:1;text-shadow:0 0 20px rgba(230,57,70,.3);margin-bottom:1rem}
.mfld{display:flex;align-items:flex-start;gap:.8rem;padding:.45rem 0;border-bottom:1px solid rgba(26,37,53,.8)}
.mkey{font-family:var(--mono);font-size:.54rem;color:var(--dim);letter-spacing:2px;text-transform:uppercase;min-width:115px;flex-shrink:0;padding-top:.18rem}
.mval{font-family:var(--body);font-size:.92rem;font-weight:600;color:var(--txt);line-height:1.45}
.mval.mc{color:var(--red)}
.mdesc{margin-top:1.1rem;background:rgba(8,11,15,.7);border:1px solid var(--bdr);border-radius:6px;padding:.85rem 1rem;font-family:var(--body);font-size:.86rem;color:var(--dim);line-height:1.7}
.mhr{border:none;border-top:1px solid var(--bdr);margin:.75rem 0}

/* ── Empty state ── */
.emt{height:225px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:.75rem;background:var(--bg2);border:1px dashed var(--bdr);border-radius:8px;transition:border-color .3s}
.emt:hover{border-color:rgba(230,57,70,.2)}
.emi{font-size:2.1rem;opacity:.17;animation:flt 3s ease-in-out infinite}
@keyframes flt{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
.emtxt{font-family:var(--mono);font-size:.6rem;color:#1e2d3d;letter-spacing:2px;text-align:center}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--bdr);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(230,57,70,.4)}

hr{border:none!important;border-top:1px solid var(--bdr)!important;margin:.7rem 0!important}
.stSpinner>div{border-top-color:var(--red)!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  MODAL HTML + JS  (injected once, lives in DOM throughout session)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="crimOverlay" onclick="if(event.target===this)closeModal()">
  <div id="crimBox">
    <div class="mcls" onclick="closeModal()">✕</div>
    <div id="crimContent"></div>
  </div>
</div>

<script>
function openModal(b64) {
    var raw  = atob(b64);
    var data = JSON.parse(raw);

    function sb(s){
        s = (s||'').toLowerCase();
        if(s.indexOf('wanted')   >=0) return '<span class="bw">⚠ WANTED</span>';
        if(s.indexOf('arrested') >=0) return '<span class="ba">⚡ ARRESTED</span>';
        if(s.indexOf('imprison') >=0) return '<span class="bi">🔒 IMPRISONED</span>';
        return '<span style="color:#8899aa;font-family:var(--mono);font-size:.72rem">'+s+'</span>';
    }

    function esc(s){ var d=document.createElement('div');d.textContent=s;return d.innerHTML; }

    var nameDisp = esc(data.full_name || data.name || '—');
    var fields = [
        ['FULL NAME',   data.full_name   || data.name || '—', ''],
        ['CRIME',       data.crime       || '—',              'mc'],
        ['STATUS',      data.status      || '—',              'status'],
        ['AGE',         data.age         || '—',              ''],
        ['GENDER',      data.gender      || '—',              ''],
        ['NATIONALITY', data.nationality || '—',              ''],
        ['LAST SEEN',   data.last_seen   || '—',              ''],
    ];
    var html = '<div class="mtag">⚠ &nbsp;CRIMINAL RECORD — CONFIDENTIAL</div>';
    html += '<div class="mname">'+nameDisp.toUpperCase()+'</div>';
    html += '<hr class="mhr">';
    fields.forEach(function(f){
        var v;
        if(f[2]==='status') v = sb(f[1]);
        else if(f[2]==='mc') v = '<span class="mval mc">'+esc(f[1])+'</span>';
        else v = '<span class="mval">'+esc(f[1])+'</span>';
        html += '<div class="mfld"><span class="mkey">'+f[0]+'</span>'+v+'</div>';
    });
    if(data.description && data.description!=='—'){
        html += '<div class="mdesc"><span style="font-family:var(--mono);font-size:.53rem;color:var(--dim);letter-spacing:2px;display:block;margin-bottom:.4rem">CASE NOTES</span>'+esc(data.description)+'</div>';
    }
    document.getElementById('crimContent').innerHTML = html;
    document.getElementById('crimOverlay').style.display = 'block';
    document.body.style.overflow = 'hidden';
}
function closeModal(){
    document.getElementById('crimOverlay').style.display='none';
    document.body.style.overflow='';
}
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    for xml in ['haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_profileface.xml']:
        cc    = cv2.CascadeClassifier(cv2.data.haarcascades + xml)
        faces = cc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50,50))
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

def predict_face(model, crop, cnames):
    # Auto-detect input size from model
    try:
        h = model.input_shape[1] or 96
        w = model.input_shape[2] or 96
    except Exception:
        h, w = 96, 96

    img = np.array(Image.fromarray(crop).resize((w, h))).astype(np.float32)

    # MobileNetV2 uses preprocess_input (scales to [-1,1]), NOT /255.0
    # This matches exactly how the training notebook preprocessed images
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
        img = mobilenet_preprocess(img)
    except Exception:
        img = img / 255.0   # fallback for custom CNN models

    preds = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    idx = int(np.argmax(preds))
    conf  = float(preds[idx])
    name  = cnames.get(idx,"Unknown")
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
            row  = match.iloc[0]
            html = ""
            for col in df.columns:
                if col == 'name': continue
                val = str(row[col])
                key = col.replace('_',' ').upper()
                if   col.lower() == 'status':      v = sbadge(val)
                elif col.lower() == 'crime':        v = f'<span class="pv cr">{val}</span>'
                elif col.lower() == 'description':  v = f'<span class="pv ds">{val}</span>'
                else:                               v = f'<span class="pv">{val}</span>'
                html += f'<div class="prow"><span class="pk">{key}</span>{v}</div>'
            st.markdown(html, unsafe_allow_html=True)
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
› &nbsp;Adjust confidence threshold in sidebar
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def row_to_b64(row_dict):
    """Encode row data as base64 JSON for safe onclick attribute."""
    clean = {k: str(v) for k, v in row_dict.items()}
    return base64.b64encode(json.dumps(clean).encode()).decode()


def build_table(df):
    """Returns the full HTML table — must be passed to st.markdown with unsafe_allow_html=True."""
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        b64      = row_to_b64(row.to_dict())
        delay    = f"{i*0.03:.2f}s"
        fname    = str(row.get('full_name', row.get('name',''))).replace('_',' ')
        crime    = str(row.get('crime','—'))
        stat     = sbadge(str(row.get('status','—')))
        age      = str(row.get('age','—'))
        gender   = str(row.get('gender','—'))
        nat      = str(row.get('nationality','—'))
        last     = str(row.get('last_seen','—'))
        rows_html += f"""
<tr style="animation-delay:{delay}" onclick="openModal('{b64}')">
  <td title="{fname}">{fname}</td>
  <td title="{crime}" style="color:var(--red);font-weight:600">{crime}</td>
  <td>{stat}</td>
  <td style="color:var(--dim)">{age}</td>
  <td style="color:var(--dim)">{gender}</td>
  <td style="color:var(--dim)" title="{nat}">{nat}</td>
  <td style="color:var(--dim)" title="{last}">{last}</td>
  <td><span class="vbtn">👁 VIEW</span></td>
</tr>"""

    return f"""
<div class="dbtw">
<table class="dbt">
<thead>
<tr>
  <th>NAME</th><th>CRIME</th><th>STATUS</th>
  <th>AGE</th><th>GENDER</th><th>NATIONALITY</th>
  <th>LAST SEEN</th><th>PROFILE</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="font-family:var(--mono);font-size:.6rem;color:var(--red);
     letter-spacing:3px;margin-bottom:1.2rem;
     border-left:3px solid var(--red);padding-left:.6rem">
  SYSTEM CONFIGURATION
</div>""", unsafe_allow_html=True)

    def slbl(t):
        st.markdown(f'<span style="font-family:var(--mono);font-size:.58rem;color:var(--dim);letter-spacing:2px">{t}</span>',
                    unsafe_allow_html=True)

    slbl("MODEL PATH")
    model_path = st.text_input("mp", value="criminal_recognition_model.keras", label_visibility="collapsed")
    slbl("DATABASE CSV")
    csv_path   = st.text_input("cp", value="criminals_info.csv",label_visibility="collapsed")
    slbl("CLASS NAMES FILE")
    cn_path    = st.text_input("cn", value="class_names.txt",label_visibility="collapsed")
    slbl("CONFIDENCE THRESHOLD")
    threshold  = st.slider("thr", 0.50, 0.99, 0.80, 0.01,label_visibility="collapsed")
    st.markdown(f'<div style="font-family:var(--mono);font-size:.58rem;color:var(--dim);text-align:center;margin-top:-.3rem">MIN {int(threshold*100)}% FOR POSITIVE ID</div>',
                unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    # ── Load resources ──
    model_loaded    = False
    model           = None
    info_df         = None
    class_names_map = {}

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            nc = model.output_shape[-1]
            model_loaded = True
            st.markdown(f'<div class="as">✓ &nbsp;MODEL LOADED &nbsp;·&nbsp; {nc} classes</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="ad">✗ MODEL ERROR<br><span style="font-size:.62rem">{str(e)[:55]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ad">✗ MODEL NOT FOUND<br><span style="font-size:.62rem;opacity:.7">{model_path}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.3rem"></div>', unsafe_allow_html=True)

    if os.path.exists(csv_path):
        try:
            info_df = load_csv(csv_path)
            st.markdown(f'<div class="as">✓ &nbsp;DATABASE &nbsp;·&nbsp; {len(info_df)} records</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="ad">✗ CSV ERROR<br><span style="font-size:.62rem">{str(e)[:55]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ad">✗ CSV NOT FOUND<br><span style="font-size:.62rem;opacity:.7">{csv_path}</span></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.3rem"></div>', unsafe_allow_html=True)

    if os.path.exists(cn_path):
        with open(cn_path) as f:
            names = [l.strip() for l in f if l.strip()]
        class_names_map = {i: n for i, n in enumerate(names)}
        st.markdown(f'<div class="as">✓ &nbsp;CLASS NAMES &nbsp;·&nbsp; {len(class_names_map)}</div>', unsafe_allow_html=True)
    elif model_loaded and info_df is not None:
        names = list(info_df['name'].str.strip())
        class_names_map = {i: n for i, n in enumerate(names)}
        st.markdown('<div class="aw">⚠ Using CSV order as class names</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ad">✗ CLASS NAMES NOT FOUND</div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:var(--mono);font-size:.5rem;color:#172030;text-align:center;line-height:2.2">
  CRIMINALID v2.0<br>FACE RECOGNITION SYSTEM<br>TensorFlow · OpenCV · Streamlit
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="banner">
  <div class="ctlx"></div><div class="cbrx"></div>
  <div class="b-logo">🔍</div>
  <div>
    <div class="b-title">CriminalID</div>
    <div class="b-sub">Automated Face Recognition &amp; Identification System</div>
  </div>
  <div class="b-right">
    <div class="b-badge">● SYSTEM ACTIVE</div>
    <div class="b-time">{time.strftime('%Y-%m-%d  %H:%M:%S')}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "  📁  UPLOAD IMAGE  ",
    "  📷  WEBCAM  ",
    "  📊  DATABASE  "
])


# ────────────────────────────────────────────
#  TAB 1 — UPLOAD
# ────────────────────────────────────────────
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
                    st.markdown('<div class="ad">⚠ &nbsp;MODEL NOT LOADED — configure in sidebar</div>', unsafe_allow_html=True)
                elif not class_names_map:
                    st.markdown('<div class="ad">⚠ &nbsp;CLASS NAMES MISSING — configure in sidebar</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("Analyzing image..."):
                        st.markdown('<div class="pbar"></div>', unsafe_allow_html=True)
                        time.sleep(0.3)
                        faces = detect_face(img_rgb)
                    if len(faces) == 0:
                        st.markdown('<div class="ad">❌ &nbsp;NO FACE DETECTED<br><span style="font-size:.68rem;opacity:.7">Ensure subject\'s face is clearly visible and well-lit.</span></div>', unsafe_allow_html=True)
                        st.session_state['up_res'] = (img_rgb, None, 0.0)
                    else:
                        x,y,w,h = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)[0]
                        pad = int(0.15*w)

                        x1 = max(0,x - pad)
                        y1 = max(0,y - pad)
                        x2 = min(img_rgb.shape[1], x + w + pad)
                        y2 = min(img_rgb.shape[0], y + h + pad)
                        crop = img_rgb[y1:y2,x1:x2]
                        name, conf = predict_face(model, crop, class_names_map)
                        c = (230,57,70) if conf>=threshold else (74,96,112)
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


# ────────────────────────────────────────────
#  TAB 2 — WEBCAM
# ────────────────────────────────────────────
with tab2:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ai" style="margin-bottom:1rem">📷 &nbsp;Click <strong>Allow</strong> when your browser asks for camera access &nbsp;·&nbsp; Then click <strong>Take Photo</strong> to capture and identify automatically</div>', unsafe_allow_html=True)

    L2, R2 = st.columns([1,1], gap="large")
    with L2:
        st.markdown('<span class="slbl">INPUT — LIVE WEBCAM CAPTURE</span>', unsafe_allow_html=True)
        cam = st.camera_input("cam", label_visibility="collapsed")

        if cam:
            img_pil = Image.open(cam).convert("RGB")
            img_rgb = np.array(img_pil)
            if not model_loaded:
                st.markdown('<div class="ad">⚠ &nbsp;MODEL NOT LOADED</div>', unsafe_allow_html=True)
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

                    x1 = max(0,x - pad)
                    y1 = max(0,y - pad)
                    x2 = min(img_rgb.shape[1], x + w + pad)
                    y2 = min(img_rgb.shape[0], y + h + pad)
                    crop = img_rgb[y1:y2,x1:x2]
                    name, conf = predict_face(model, crop, class_names_map)
                    st.markdown(f"<div style='font-family:monospace;color:#7b9cff'>RAW CONFIDENCE: {round(conf,4)}</div>", unsafe_allow_html=True)
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


# ────────────────────────────────────────────
#  TAB 3 — DATABASE
# ────────────────────────────────────────────
with tab3:
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.markdown('<span class="slbl">CRIMINAL DATABASE — ALL RECORDS</span>', unsafe_allow_html=True)

    if info_df is not None:
        total      = len(info_df)
        wanted     = len(info_df[info_df['status'].str.lower().str.contains('wanted',   na=False)]) if 'status' in info_df.columns else 0
        arrested   = len(info_df[info_df['status'].str.lower().str.contains('arrested', na=False)]) if 'status' in info_df.columns else 0
        imprisoned = len(info_df[info_df['status'].str.lower().str.contains('imprison', na=False)]) if 'status' in info_df.columns else 0

        # Stat boxes
        c1,c2,c3,c4 = st.columns(4)
        for col_ui, val, lbl, col_hex in [
            (c1, total,      "TOTAL RECORDS", "#4361ee"),
            (c2, wanted,     "WANTED",         "#e63946"),
            (c3, arrested,   "ARRESTED",       "#f4a261"),
            (c4, imprisoned, "IMPRISONED",     "#2ec4b6"),
        ]:
            col_ui.markdown(f"""
<div class="mbox" style="border-color:{col_hex}22">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{col_hex}"></div>
  <span class="mval" style="color:{col_hex}">{val}</span>
  <span class="mlbl">{lbl}</span>
</div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

        search = st.text_input("🔎  SEARCH RECORDS",
                               placeholder="Filter by name, crime, status...",
                               key="dbs", label_visibility="visible")
        filtered = info_df
        if search:
            mask     = info_df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
            filtered = info_df[mask]

        # ── KEY FIX: pass HTML to st.markdown with unsafe_allow_html=True ──
        st.markdown(build_table(filtered), unsafe_allow_html=True)

        st.markdown(f"""
<div style="font-family:var(--mono);font-size:.55rem;color:var(--dim);text-align:right;margin-top:.5rem">
  SHOWING {len(filtered)} OF {total} RECORDS &nbsp;·&nbsp; CLICK ANY ROW OR 👁 VIEW TO OPEN FULL PROFILE
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="ad">⚠ &nbsp;DATABASE NOT LOADED<br><span style="font-size:.68rem;opacity:.7">Set the CSV path in the sidebar and ensure the file exists.</span></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="card" style="margin-top:1.2rem">
  <div style="font-family:var(--mono);font-size:.55rem;color:var(--dim);letter-spacing:2px;margin-bottom:.8rem">REQUIRED CSV FORMAT</div>
  <div style="font-family:var(--mono);font-size:.68rem;color:#3a5060;line-height:2.2;background:#080b0f;padding:1rem;border-radius:4px">
    name, full_name, crime, status, age, gender, nationality, last_seen, description<br>
    John_Doe, John Doe, Armed Robbery, Wanted, 34, Male, Indian, Mumbai 2024, Dangerous armed robber<br>
    Jane_Smith, Jane Smith, Fraud, Arrested, 28, Female, Indian, Pune 2023, White collar fraud
  </div>
  <div style="font-family:var(--mono);font-size:.58rem;color:#2a3a4a;margin-top:.7rem">
    ⚠ The 'name' column must exactly match your dataset folder names
  </div>
</div>""", unsafe_allow_html=True)
