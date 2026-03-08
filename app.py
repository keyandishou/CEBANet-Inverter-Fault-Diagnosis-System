import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

# ==========================================
# 模型架构引入
# ==========================================
try:
    from model20 import CNN_LSTM_Attention
except ImportError:
    st.error("Error: 'model20.py' not found in the current directory.")

# ==========================================
# 系统 UI 与缓存配置
# ==========================================
st.set_page_config(page_title="CEBANet Fault Diagnosis", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .github-link {
        display: inline-block;
        padding: 8px 15px;
        background-color: #24292e;
        color: white !important;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 20px;
    }
    .github-link:hover {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Attention(num_classes=19).to(device)
    try:
        model.load_state_dict(torch.load("CNN7.pth", map_location=device))
    except Exception as e:
        st.sidebar.error(f"Weights load failed: {e}")
    model.eval() 
    return model, device

def process_raw_data(file_obj):
    xy = np.loadtxt(file_obj, delimiter=',', dtype=np.float32)
    first_col = xy[:, 0]
    if np.all(np.mod(first_col, 1) == 0) and np.max(first_col) <= 18:
        y_data = xy[:, 0]
        x_data = xy[:, 1:]
    else:
        y_data = xy[:, -1]
        x_data = xy[:, :-1]
    return torch.from_numpy(x_data), torch.from_numpy(y_data).long()

FAULT_DICT = {
    0: "Normal (正常)", 
    1: "Phase A - T1 Open Fault", 2: "Phase A - T2 Open Fault", 
    3: "Phase A - T3 Open Fault", 4: "Phase A - T4 Open Fault",
    18: "Phase C - T4 Open Fault"
}
for i in range(19):
    if i not in FAULT_DICT:
        FAULT_DICT[i] = f"Fault Class {i}"

# ==========================================
# 侧边栏 (Sidebar)
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center;'>CEBANet</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <a href="https://github.com/keyandishou" target="_blank" class="github-link">
        🔗 GitHub Repository
    </a>
    """, unsafe_allow_html=True
)

st.sidebar.header("Data Import")
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
st.sidebar.caption("Auto-detects label position (first/last column). Supports 19 fault classes.")

# ==========================================
# 主界面 (Main UI)
# ==========================================
st.title("⚡ CEBANet: Inverter Fault Diagnosis System")
st.markdown("A lightweight 1D-CNN-BiLSTM-Attention framework for 19-class open-circuit fault diagnosis in T-type three-level inverters.")

model, device = load_model()

if uploaded_file is not None:
    x_tensor, y_tensor = process_raw_data(uploaded_file)
    
    tab1, tab2 = st.tabs(["Single Inference (单样本推理)", "Batch Evaluation (批量验证)"])

    with tab1:
        st.write(f"**Loaded Samples**: `{len(x_tensor)}`")
        idx = st.slider("Select Sample Index:", 0, len(x_tensor)-1, 0)
        
        feat = x_tensor[idx].numpy()
        if len(feat) == 360:
            st.subheader("Signal Waveform")
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.plot(feat[0:120], label="Phase A", color='#e74c3c', linewidth=1.2)
            ax.plot(feat[120:240], label="Phase B", color='#2ecc71', linewidth=1.2)
            ax.plot(feat[240:360], label="Phase C", color='#3498db', linewidth=1.2)
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig)

        if st.button("Run Inference", type="primary"):
            input_data = x_tensor[idx].unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                start_time = time.time()
                output = model(input_data)
                prob = F.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)
                latency = (time.time() - start_time) * 1000
            
            p_idx = pred.item()
            t_idx = y_tensor[idx].item()
            
            st.markdown("### Inference Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ground Truth", f"Class {t_idx}")
            c2.metric("Prediction", f"Class {p_idx}", delta="Match" if p_idx == t_idx else "Mismatch", delta_color="normal" if p_idx == t_idx else "inverse")
            c3.metric("Confidence", f"{conf.item()*100:.2f}%")
            c4.metric("Latency", f"{latency:.2f} ms")
            
            st.info(f"**Status:** {FAULT_DICT.get(p_idx, f'Class {p_idx}')}")

    with tab2:
        st.markdown("Evaluate the entire uploaded dataset using standard PyTorch `DataLoader` (batch_size=8).")
        if st.button("Run Batch Evaluation"):
            dataset = TensorDataset(x_tensor, y_tensor)
            val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
            
            correct, total = 0, 0
            with st.spinner("Evaluating..."):
                start_time = time.time()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        inputs = inputs.unsqueeze(1)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                acc = (correct / total) * 100
                eval_time = time.time() - start_time
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Samples", total)
                c2.metric("Correct", correct)
                c3.metric("Accuracy", f"{acc:.2f}%")
                c4.metric("Total Time", f"{eval_time:.2f} s")
else:
    st.info("Please upload a CSV dataset from the sidebar to start.")