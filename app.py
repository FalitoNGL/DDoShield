import os
import sys
import pkgutil
import importlib.util

# Monkeypatch pkgutil for compatibility with Python 3.14+
if not hasattr(pkgutil, 'get_loader'):
    def _get_loader(module_or_name):
        try:
            spec = importlib.util.find_spec(module_or_name)
            return spec.loader if spec else None
        except Exception:
            return None
    pkgutil.get_loader = _get_loader

if not hasattr(pkgutil, 'find_loader'):
    def _find_loader(fullname, path=None):
        try:
            spec = importlib.util.find_spec(fullname, path)
            return spec.loader if spec else None
        except Exception:
            return None
    pkgutil.find_loader = _find_loader

import ast

# Monkeypatch ast for compatibility with Python 3.14+ (removed ast.Str, ast.Num, etc.)
if not hasattr(ast, 'Str'):
    class StrNode(ast.Constant):
        def __init__(self, s=None, **kwargs):
            if s is not None:
                super().__init__(value=s, **kwargs)
            elif 'value' in kwargs:
                super().__init__(**kwargs)
            elif 's' in kwargs:
                super().__init__(value=kwargs.pop('s'), **kwargs)
            else:
                super().__init__(value="", **kwargs)
        @property
        def s(self):
            return self.value
        @s.setter
        def s(self, val):
            self.value = val
    ast.Str = StrNode

if not hasattr(ast, 'Num'):
    class NumNode(ast.Constant):
        def __init__(self, n=None, **kwargs):
            if n is not None:
                super().__init__(value=n, **kwargs)
            elif 'value' in kwargs:
                super().__init__(**kwargs)
            elif 'n' in kwargs:
                super().__init__(value=kwargs.pop('n'), **kwargs)
            else:
                super().__init__(value=0, **kwargs)
        @property
        def n(self):
            return self.value
        @n.setter
        def n(self, val):
            self.value = val
    ast.Num = NumNode

if not hasattr(ast, 'Bytes'):
    class BytesNode(ast.Constant):
        def __init__(self, s=None, **kwargs):
            if s is not None:
                super().__init__(value=s, **kwargs)
            elif 'value' in kwargs:
                super().__init__(**kwargs)
            elif 's' in kwargs:
                super().__init__(value=kwargs.pop('s'), **kwargs)
            else:
                super().__init__(value=b"", **kwargs)
        @property
        def s(self):
            return self.value
        @s.setter
        def s(self, val):
            self.value = val
    ast.Bytes = BytesNode

import time
import joblib
import psutil
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Models
print("[DDoShield] Loading models...")
CURRENT_DIR = Path(os.path.dirname(__file__))
MODEL_DIR = CURRENT_DIR / "models"

try:
    imputer = joblib.load(MODEL_DIR / "imputer.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    selector = joblib.load(MODEL_DIR / "selector_kopt.pkl")
    model = joblib.load(MODEL_DIR / "xgb_model_kopt.pkl")
    print("[DDoShield] Models loaded successfully from local models/ folder!")
except Exception as e:
    print(f"[DDoShield] WARNING: Failed to load models from local folder: {str(e)}")
    print("[DDoShield] Attempting fallback to parent output/ directory...")
    try:
        parent_output = CURRENT_DIR.parent / "output"
        imputer = joblib.load(parent_output / "imputer.pkl")
        scaler = joblib.load(parent_output / "scaler.pkl")
        selector = joblib.load(parent_output / "selector_kopt.pkl")
        model = joblib.load(parent_output / "xgb_model_kopt.pkl")
        print("[DDoShield] Fallback to parent output/ folder successful!")
    except Exception as ex:
        print(f"[DDoShield] ERROR: Models cannot be loaded: {str(ex)}")
        imputer, scaler, selector, model = None, None, None, None

# Global state to store the last analyzed batch for XAI explanation
LAST_ANALYZED_DF = None
LAST_ANALYZED_META = None

# Feature list in exact training order (78 features)
FEATURE_COLUMNS = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
    'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
    ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
    'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean',
    ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean',
    ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
    ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
    ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
    'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
    ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count',
    ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
    ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
    ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
    ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
    ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward',
    'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
    ' Idle Std', ' Idle Max', ' Idle Min'
]

def parse_packets(packets):
    """
    Extracts flows and features from a Scapy packet list.
    """
    from scapy.all import IP, TCP, UDP, ICMP
    flows = {}
    
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        
        sport, dport = 0, 0
        if pkt.haslayer(TCP):
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        elif pkt.haslayer(ICMP):
            sport = 0
            dport = 0
        else:
            continue
            
        flow_key = (src_ip, dst_ip, sport, dport, proto)
        rev_key = (dst_ip, src_ip, dport, sport, proto)
        
        is_forward = True
        if flow_key in flows:
            key = flow_key
        elif rev_key in flows:
            key = rev_key
            is_forward = False
        else:
            key = flow_key
            flows[key] = {
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'sport': sport,
                'dport': dport,
                'proto': proto,
                'fwd_packets': [],
                'bwd_packets': [],
                'fwd_times': [],
                'bwd_times': [],
                'start_time': float(pkt.time),
                'last_time': float(pkt.time),
            }
            
        flow = flows[key]
        flow['last_time'] = float(pkt.time)
        pkt_len = len(pkt)
        
        if is_forward:
            flow['fwd_packets'].append(pkt_len)
            flow['fwd_times'].append(float(pkt.time))
        else:
            flow['bwd_packets'].append(pkt_len)
            flow['bwd_times'].append(float(pkt.time))
            
    flow_features_list = []
    flow_meta_list = []
    
    for key, f in flows.items():
        fwd_p_count = len(f['fwd_packets'])
        bwd_p_count = len(f['bwd_packets'])
        total_fwd_len = sum(f['fwd_packets'])
        total_bwd_len = sum(f['bwd_packets'])
        
        if fwd_p_count == 0 and bwd_p_count == 0:
            continue
            
        fwd_len_max = max(f['fwd_packets']) if fwd_p_count > 0 else 0
        fwd_len_min = min(f['fwd_packets']) if fwd_p_count > 0 else 0
        fwd_len_mean = np.mean(f['fwd_packets']) if fwd_p_count > 0 else 0
        fwd_len_std = np.std(f['fwd_packets']) if fwd_p_count > 1 else 0
        
        bwd_len_max = max(f['bwd_packets']) if bwd_p_count > 0 else 0
        bwd_len_min = min(f['bwd_packets']) if bwd_p_count > 0 else 0
        bwd_len_mean = np.mean(f['bwd_packets']) if bwd_p_count > 0 else 0
        bwd_len_std = np.std(f['bwd_packets']) if bwd_p_count > 1 else 0
        
        duration = float(f['last_time'] - f['start_time'])
        duration_us = duration * 1000000
        
        fwd_iat = np.diff(f['fwd_times']) if fwd_p_count > 1 else [0]
        bwd_iat = np.diff(f['bwd_times']) if bwd_p_count > 1 else [0]
        
        fwd_iat_total = sum(fwd_iat) * 1000000
        fwd_iat_mean = np.mean(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0
        fwd_iat_std = np.std(fwd_iat) * 1000000 if len(fwd_iat) > 1 else 0
        fwd_iat_max = max(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0
        fwd_iat_min = min(fwd_iat) * 1000000 if len(fwd_iat) > 0 else 0
        
        bwd_iat_total = sum(bwd_iat) * 1000000
        bwd_iat_mean = np.mean(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0
        bwd_iat_std = np.std(bwd_iat) * 1000000 if len(bwd_iat) > 1 else 0
        bwd_iat_max = max(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0
        bwd_iat_min = min(bwd_iat) * 1000000 if len(bwd_iat) > 0 else 0
        
        all_packets = f['fwd_packets'] + f['bwd_packets']
        pkt_len_mean = np.mean(all_packets) if len(all_packets) > 0 else 0
        pkt_len_std = np.std(all_packets) if len(all_packets) > 1 else 0
        pkt_len_var = np.var(all_packets) if len(all_packets) > 1 else 0
        
        avg_pkt_size = (total_fwd_len + total_bwd_len) / (fwd_p_count + bwd_p_count) if (fwd_p_count + bwd_p_count) > 0 else 0
        
        meta = {
            'src_ip': f['src_ip'],
            'dst_ip': f['dst_ip'],
            'sport': int(f['sport']),
            'dport': int(f['dport']),
            'proto': 'TCP' if f['proto'] == 6 else ('UDP' if f['proto'] == 17 else ('ICMP' if f['proto'] == 1 else str(f['proto']))),
            'flow_duration_s': round(duration, 4),
            'packet_count': fwd_p_count + bwd_p_count,
            'byte_count': total_fwd_len + total_bwd_len
        }
        
        feats = {
            ' Destination Port': f['dport'],
            ' Flow Duration': duration_us,
            ' Total Fwd Packets': fwd_p_count,
            ' Total Backward Packets': bwd_p_count,
            'Total Length of Fwd Packets': total_fwd_len,
            ' Total Length of Bwd Packets': total_bwd_len,
            ' Fwd Packet Length Max': fwd_len_max,
            ' Fwd Packet Length Min': fwd_len_min,
            ' Fwd Packet Length Mean': fwd_len_mean,
            ' Fwd Packet Length Std': fwd_len_std,
            'Bwd Packet Length Max': bwd_len_max,
            ' Bwd Packet Length Min': bwd_len_min,
            ' Bwd Packet Length Mean': bwd_len_mean,
            ' Bwd Packet Length Std': bwd_len_std,
            'Flow Bytes/s': (total_fwd_len + total_bwd_len) / duration if duration > 0 else 0,
            ' Flow Packets/s': (fwd_p_count + bwd_p_count) / duration if duration > 0 else 0,
            ' Flow IAT Mean': np.mean(np.diff(f['fwd_times'] + f['bwd_times'])) * 1000000 if (fwd_p_count + bwd_p_count) > 1 else 0,
            ' Flow IAT Std': np.std(np.diff(f['fwd_times'] + f['bwd_times'])) * 1000000 if (fwd_p_count + bwd_p_count) > 2 else 0,
            ' Flow IAT Max': max(np.diff(f['fwd_times'] + f['bwd_times'])) * 1000000 if (fwd_p_count + bwd_p_count) > 1 else 0,
            ' Flow IAT Min': min(np.diff(f['fwd_times'] + f['bwd_times'])) * 1000000 if (fwd_p_count + bwd_p_count) > 1 else 0,
            'Fwd IAT Total': fwd_iat_total,
            ' Fwd IAT Mean': fwd_iat_mean,
            ' Fwd IAT Std': fwd_iat_std,
            ' Fwd IAT Max': fwd_iat_max,
            ' Fwd IAT Min': fwd_iat_min,
            'Bwd IAT Total': bwd_iat_total,
            ' Bwd IAT Mean': bwd_iat_mean,
            ' Bwd IAT Std': bwd_iat_std,
            ' Bwd IAT Max': bwd_iat_max,
            ' Bwd IAT Min': bwd_iat_min,
            'Fwd PSH Flags': 0,
            ' Bwd PSH Flags': 0,
            ' Fwd URG Flags': 0,
            ' Bwd URG Flags': 0,
            ' Fwd Header Length': fwd_p_count * 20,
            ' Bwd Header Length': bwd_p_count * 20,
            'Fwd Packets/s': fwd_p_count / duration if duration > 0 else 0,
            ' Bwd Packets/s': bwd_p_count / duration if duration > 0 else 0,
            ' Min Packet Length': min(all_packets) if len(all_packets) > 0 else 0,
            ' Max Packet Length': max(all_packets) if len(all_packets) > 0 else 0,
            ' Packet Length Mean': pkt_len_mean,
            ' Packet Length Std': pkt_len_std,
            ' Packet Length Variance': pkt_len_var,
            'FIN Flag Count': 0,
            ' SYN Flag Count': 0,
            ' RST Flag Count': 0,
            ' PSH Flag Count': 0,
            ' ACK Flag Count': 0,
            ' URG Flag Count': 0,
            ' CWE Flag Count': 0,
            ' ECE Flag Count': 0,
            ' Down/Up Ratio': bwd_p_count / fwd_p_count if fwd_p_count > 0 else 0,
            ' Average Packet Size': avg_pkt_size,
            ' Avg Fwd Segment Size': fwd_len_mean,
            ' Avg Bwd Segment Size': bwd_len_mean,
            ' Fwd Header Length.1': fwd_p_count * 20,
            'Fwd Avg Bytes/Bulk': 0,
            ' Fwd Avg Packets/Bulk': 0,
            ' Fwd Avg Bulk Rate': 0,
            ' Bwd Avg Bytes/Bulk': 0,
            ' Bwd Avg Packets/Bulk': 0,
            'Bwd Avg Bulk Rate': 0,
            'Subflow Fwd Packets': fwd_p_count,
            ' Subflow Fwd Bytes': total_fwd_len,
            ' Subflow Bwd Packets': bwd_p_count,
            ' Subflow Bwd Bytes': total_bwd_len,
            'Init_Win_bytes_forward': 8192,
            ' Init_Win_bytes_backward': 8192,
            ' act_data_pkt_fwd': max(0, fwd_p_count - 1),
            ' min_seg_size_forward': 20,
            'Active Mean': 0,
            ' Active Std': 0,
            ' Active Max': 0,
            ' Active Min': 0,
            'Idle Mean': 0,
            ' Idle Std': 0,
            ' Idle Max': 0,
            ' Idle Min': 0
        }
        flow_features_list.append(feats)
        flow_meta_list.append(meta)
        
    if not flow_features_list:
        return None, []
        
    df_features = pd.DataFrame(flow_features_list)
    return df_features, flow_meta_list

def parse_pcap_with_scapy(filepath):
    """
    Parses PCAP file using Scapy by reading and passing to parse_packets.
    """
    from scapy.all import rdpcap
    print(f"Parsing PCAP file: {filepath}...")
    try:
        packets = rdpcap(filepath)
    except Exception as e:
        print(f"Scapy failed to read PCAP: {str(e)}")
        return None, []
    return parse_packets(packets)

def predict_flows(df_features):
    """
    Runs prediction pipeline on dataframe of features.
    """
    if model is None or scaler is None or selector is None or imputer is None:
        # Fallback simulated predictions if models aren't loaded
        print("[DDoShield] Running simulated prediction (Models not loaded)")
        # Classify as DDoS if dest port is 80 and packets > 10
        sim_preds = []
        for _, row in df_features.iterrows():
            if row.get(' Destination Port', 0) in [80, 443, 8080] and row.get(' Total Fwd Packets', 0) > 15:
                sim_preds.append(1)  # DDoS
            else:
                sim_preds.append(0)  # Benign
        return np.array(sim_preds)
        
    # Standardize column order
    df_ordered = df_features[FEATURE_COLUMNS].copy()
    
    # 1. Impute
    X_imputed = imputer.transform(df_ordered.values)
    # 2. Scale
    X_scaled = scaler.transform(X_imputed)
    # 3. Select K=30
    X_k30 = selector.transform(X_scaled)
    # 4. Predict
    preds = model.predict(X_k30)
    return preds

def check_heuristics(meta):
    # Rule 1: DNS Amplification (UDP port 53, high packet count)
    if meta['proto'] == 'UDP' and (meta['sport'] == 53 or meta['dport'] == 53):
        if meta['packet_count'] > 50:
            return True, "DDoS (DNS Amp)"
            
    # Rule 2: NTP Amplification (UDP port 123)
    if meta['proto'] == 'UDP' and (meta['sport'] == 123 or meta['dport'] == 123):
        if meta['packet_count'] > 50:
            return True, "DDoS (NTP Amp)"
            
    # Rule 3: SSDP/UPnP Amplification (UDP port 1900)
    if meta['proto'] == 'UDP' and (meta['sport'] == 1900 or meta['dport'] == 1900):
        if meta['packet_count'] > 50:
            return True, "DDoS (SSDP Amp)"
            
    # Rule 4: Memcached Amplification (UDP port 11211)
    if meta['proto'] == 'UDP' and (meta['sport'] == 11211 or meta['dport'] == 11211):
        if meta['packet_count'] > 30:
            return True, "DDoS (Memcached Amp)"

    # Rule 5: Generic Volumetric UDP flood
    if meta['proto'] == 'UDP' and meta['packet_count'] > 300:
        return True, "DDoS (UDP Flood)"
        
    # Rule 6: Generic Volumetric TCP Flood / Rate based
    if meta['flow_duration_s'] > 0:
        rate = meta['packet_count'] / meta['flow_duration_s']
        if rate > 200 and meta['packet_count'] > 50:
            return True, f"DDoS ({meta['proto']} Flood)"
    else:
        if meta['packet_count'] > 100:
            return True, f"DDoS ({meta['proto']} Flood)"
            
    return False, None

def aggregate_and_detect_ddos(flow_meta, predictions):
    # 1. Aggregate traffic by Destination IP
    dst_aggregates = {}
    for i, meta in enumerate(flow_meta):
        dst = meta['dst_ip']
        if dst not in dst_aggregates:
            dst_aggregates[dst] = {
                'total_packets': 0,
                'total_bytes': 0,
                'unique_sources': set(),
                'unique_sports': set(),
                'unique_dports': set(),
                'protocols': set(),
                'flow_count': 0
            }
        agg = dst_aggregates[dst]
        agg['total_packets'] += meta['packet_count']
        agg['total_bytes'] += meta['byte_count']
        agg['unique_sources'].add(meta['src_ip'])
        agg['unique_sports'].add(meta['sport'])
        agg['unique_dports'].add(meta['dport'])
        agg['protocols'].add(meta['proto'])
        agg['flow_count'] += 1

    # 2. Identify target IPs under distributed/volumetric attack
    under_attack_dsts = {}
    for dst, agg in dst_aggregates.items():
        is_under_attack = False
        attack_reason = "BENIGN"
        
        # 2a. Distributed SYN Flood / Port Scan (many flows, TCP)
        if agg['flow_count'] > 50 and (len(agg['unique_sources']) > 10 or len(agg['unique_sports']) > 20):
            if 'TCP' in agg['protocols']:
                is_under_attack = True
                attack_reason = "DDoS (Distributed TCP Flood)"
            elif 'UDP' in agg['protocols']:
                is_under_attack = True
                attack_reason = "DDoS (Distributed UDP Flood)"
                
        # 2b. ICMP Flood (large volume of ICMP)
        if 'ICMP' in agg['protocols'] and agg['total_packets'] > 50:
            is_under_attack = True
            attack_reason = "DDoS (ICMP Flood)"
            
        # 2c. Reflection Amplification (DNS, NTP, SNMP, SSDP)
        if 'UDP' in agg['protocols'] and agg['total_packets'] > 50:
            reflective_ports = {53, 123, 161, 1900}
            if len(agg['unique_sports'] & reflective_ports) > 0:
                is_under_attack = True
                if 53 in agg['unique_sports']:
                    attack_reason = "DDoS (DNS Amplification)"
                elif 123 in agg['unique_sports']:
                    attack_reason = "DDoS (NTP Amplification)"
                elif 161 in agg['unique_sports']:
                    attack_reason = "DDoS (SNMP Amplification)"
                elif 1900 in agg['unique_sports']:
                    attack_reason = "DDoS (SSDP Amplification)"
                else:
                    attack_reason = "DDoS (UDP Amplification)"
                    
        # 2d. Generic Volumetric UDP flood (many flows or packets, UDP)
        if 'UDP' in agg['protocols'] and (agg['flow_count'] > 50 or agg['total_packets'] > 300) and (len(agg['unique_sources']) > 10 or len(agg['unique_sports']) > 20 or len(agg['unique_dports']) > 20):
            is_under_attack = True
            attack_reason = "DDoS (UDP Flood)"

        if is_under_attack:
            under_attack_dsts[dst] = attack_reason

    # 3. Apply labels and compute metrics
    flow_results = []
    benign_count = 0
    ddos_count = 0
    
    for i, meta in enumerate(flow_meta):
        ml_pred = predictions[i]
        is_attack, attack_type = check_heuristics(meta)
        dst = meta['dst_ip']
        
        if dst in under_attack_dsts:
            pred_label = under_attack_dsts[dst]
            ddos_count += 1
        elif is_attack:
            pred_label = attack_type
            ddos_count += 1
        elif ml_pred == 1:
            pred_label = "DDoS (XGBoost)"
            ddos_count += 1
        else:
            pred_label = "BENIGN"
            benign_count += 1
            
        meta['prediction'] = pred_label
        flow_results.append(meta)
        
    return flow_results, benign_count, ddos_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    t0 = time.perf_counter()
    
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()
    df_features = None
    flow_meta = []
    
    if ext == '.csv':
        try:
            df_csv = pd.read_csv(filepath, low_memory=False)
            if ' Label' in df_csv.columns:
                df_csv = df_csv.drop(columns=[' Label'])
            elif 'Label' in df_csv.columns:
                df_csv = df_csv.drop(columns=['Label'])
                
            df_csv.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            flow_meta = []
            for i, row in df_csv.iterrows():
                flow_meta.append({
                    'src_ip': row.get(' Source IP', row.get('Source IP', '192.168.1.50')),
                    'dst_ip': row.get(' Destination IP', row.get('Destination IP', '10.0.0.1')),
                    'sport': int(row.get(' Source Port', row.get('Source Port', 0))),
                    'dport': int(row.get(' Destination Port', row.get('Destination Port', 80))),
                    'proto': str(row.get(' Protocol', row.get('Protocol', 'TCP'))),
                    'flow_duration_s': round(float(row.get(' Flow Duration', row.get('Flow Duration', 0))) / 1000000.0, 4),
                    'packet_count': int(row.get(' Total Fwd Packets', 0) + row.get(' Total Backward Packets', 0)),
                    'byte_count': int(row.get('Total Length of Fwd Packets', 0) + row.get(' Total Length of Bwd Packets', 0))
                })
            df_features = df_csv
        except Exception as e:
            return jsonify({'error': f'Failed to process CSV file: {str(e)}'}), 400
            
    elif ext in ['.pcap', '.pcapng']:
        try:
            df_features, flow_meta = parse_pcap_with_scapy(filepath)
            if df_features is None:
                return jsonify({'error': 'No IP/TCP/UDP packets found in PCAP.'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to process PCAP file: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Unsupported file format. Please upload .pcap or .csv'}), 400
        
    # Run predictions
    try:
        predictions = predict_flows(df_features)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    
    # Merge predictions with heuristics and distributed/volumetric aggregation
    flow_results, benign_count, ddos_count = aggregate_and_detect_ddos(flow_meta, predictions)
    total_flows = len(flow_meta)
        
    # Calculate threat index
    threat_percent = round((ddos_count / total_flows) * 100, 2) if total_flows > 0 else 0
    threat_level = "CRITICAL" if threat_percent > 50 else ("WARNING" if threat_percent > 10 else "SAFE")
    
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    
    response = {
        'filename': filename,
        'file_size_kb': round(os.path.getsize(filepath) / 1024, 2),
        'total_flows': total_flows,
        'benign_count': benign_count,
        'ddos_count': ddos_count,
        'threat_percent': threat_percent,
        'threat_level': threat_level,
        'processing_time_ms': round(latency_ms, 2),
        'ms_per_flow': round(latency_ms / total_flows, 4) if total_flows > 0 else 0,
        'throughput_flows_s': round(total_flows / (t1 - t0), 2) if (t1 - t0) > 0 else 0,
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'flows': flow_results[:200]  # Return top 200 flows for display in table
    }
    
    # Clean up file
    try:
        os.remove(filepath)
    except:
        pass
        
    global LAST_ANALYZED_DF, LAST_ANALYZED_META
    LAST_ANALYZED_DF = df_features
    LAST_ANALYZED_META = flow_results
    
    return jsonify(response)
        
@app.route('/api/live_sniff', methods=['POST'])
def live_sniff():
    """
    Sniffs traffic on the default interface, extracts flows, and runs detection.
    """
    from scapy.all import sniff
    
    # Get sniffing duration from request, default is 5 seconds
    data = request.json or {}
    duration = int(data.get('duration', 5))
    if duration > 30:  # safety limit to prevent resource locks
        duration = 30
        
    print(f"[DDoShield] Starting live sniffing for {duration} seconds...")
    t0 = time.perf_counter()
    
    try:
        # Sniff packets from network interface
        packets = sniff(timeout=duration)
        t_sniff = time.perf_counter()
        packet_count = len(packets)
        print(f"[DDoShield] Sniffed {packet_count} packets.")
        
        if packet_count == 0:
            return jsonify({
                'error': 'Tidak ada paket yang terendus (sniffed) dalam durasi tersebut. Pastikan ada aktivitas jaringan.'
            }), 400
            
        # Parse packets into flow features and metadata
        df_features, flow_meta = parse_packets(packets)
        if df_features is None or len(df_features) == 0:
            return jsonify({
                'error': 'Gagal mengekstrak flow IP valid dari paket yang terendus.'
            }), 400
            
        # Run predictions
        predictions = predict_flows(df_features)
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        # Merge predictions with heuristics and distributed/volumetric aggregation
        flow_results, benign_count, ddos_count = aggregate_and_detect_ddos(flow_meta, predictions)
        total_flows = len(flow_meta)
            
        # Calculate threat index
        threat_percent = round((ddos_count / total_flows) * 100, 2) if total_flows > 0 else 0
        threat_level = "CRITICAL" if threat_percent > 50 else ("WARNING" if threat_percent > 10 else "SAFE")
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        # For live sniffing, the actual detection latency is processing time (t1 - t_sniff)
        detection_latency_ms = (t1 - t_sniff) * 1000 if (t1 - t_sniff) > 0 else 0
        response = {
            'filename': f'Live_Capture_{duration}s',
            'file_size_kb': 0.0,
            'total_flows': total_flows,
            'benign_count': benign_count,
            'ddos_count': ddos_count,
            'threat_percent': threat_percent,
            'threat_level': threat_level,
            'processing_time_ms': round(latency_ms, 2),
            'ms_per_flow': round(detection_latency_ms / total_flows, 4) if total_flows > 0 else 0,
            'throughput_flows_s': round(total_flows / (t1 - t_sniff), 2) if (t1 - t_sniff) > 0 else 0,
            'cpu_usage': cpu_usage,
            'ram_usage': ram_usage,
            'flows': flow_results[:200]  # Return top 200 flows for display in table
        }
        
        global LAST_ANALYZED_DF, LAST_ANALYZED_META
        LAST_ANALYZED_DF = df_features
        LAST_ANALYZED_META = flow_results
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[DDoShield] Sniffing error: {str(e)}")
        return jsonify({
            'error': f'Gagal mengaktifkan live sniffing: {str(e)}. Pastikan Npcap/Wireshark terpasang dan Anda memiliki hak administrator.'
        }), 500

@app.route('/api/explain/<int:flow_idx>', methods=['GET'])
def explain(flow_idx):
    global LAST_ANALYZED_DF, LAST_ANALYZED_META
    
    if LAST_ANALYZED_DF is None or LAST_ANALYZED_META is None:
        return jsonify({'error': 'Tidak ada data aliran yang aktif untuk dijelaskan.'}), 400
        
    if flow_idx < 0 or flow_idx >= len(LAST_ANALYZED_META):
        return jsonify({'error': 'Indeks aliran tidak valid.'}), 400
        
    meta = LAST_ANALYZED_META[flow_idx]
    prediction = meta['prediction']
    
    # Friendly names for the features
    FEATURE_NAME_MAP = {
        ' Destination Port': 'Port Tujuan (Destination Port)',
        ' Flow Duration': 'Durasi Aliran (Flow Duration)',
        ' Total Fwd Packets': 'Total Paket Forward (Fwd Packets)',
        ' Total Backward Packets': 'Total Paket Backward (Bwd Packets)',
        'Total Length of Fwd Packets': 'Total Ukuran Paket Forward (Fwd Bytes)',
        ' Total Length of Bwd Packets': 'Total Ukuran Paket Backward (Bwd Bytes)',
        ' Fwd Packet Length Max': 'Ukuran Paket Forward Maksimum (Fwd Pkt Max)',
        ' Fwd Packet Length Min': 'Ukuran Paket Forward Minimum (Fwd Pkt Min)',
        ' Fwd Packet Length Mean': 'Rata-rata Ukuran Paket Forward (Fwd Pkt Mean)',
        ' Fwd Packet Length Std': 'Standar Deviasi Ukuran Paket Forward',
        'Bwd Packet Length Max': 'Ukuran Paket Backward Maksimum',
        ' Bwd Packet Length Min': 'Ukuran Paket Backward Minimum',
        ' Bwd Packet Length Mean': 'Rata-rata Ukuran Paket Backward',
        ' Bwd Packet Length Std': 'Standar Deviasi Ukuran Paket Backward',
        'Flow Bytes/s': 'Laju Transfer Data (Flow Bytes/s)',
        ' Flow Packets/s': 'Laju Pengiriman Paket (Flow Packets/s)',
        ' Flow IAT Mean': 'Rata-rata Jeda Waktu Antar Paket (Flow IAT Mean)',
        ' Flow IAT Std': 'Standar Deviasi Jeda Waktu Antar Paket',
        ' Flow IAT Max': 'Jeda Waktu Antar Paket Maksimum',
        ' Flow IAT Min': 'Jeda Waktu Antar Paket Minimum',
        'Fwd IAT Total': 'Total Jeda Waktu Paket Forward',
        ' Fwd IAT Mean': 'Rata-rata Jeda Waktu Paket Forward',
        ' Fwd IAT Std': 'Standar Deviasi Jeda Waktu Paket Forward',
        ' Fwd IAT Max': 'Jeda Waktu Paket Forward Maksimum',
        ' Fwd IAT Min': 'Jeda Waktu Paket Forward Minimum',
        'Bwd IAT Total': 'Total Jeda Waktu Paket Backward',
        ' Bwd IAT Mean': 'Rata-rata Jeda Waktu Paket Backward',
        ' Bwd IAT Std': 'Standar Deviasi Jeda Waktu Paket Backward',
        ' Bwd IAT Max': 'Jeda Waktu Paket Backward Maksimum',
        ' Bwd IAT Min': 'Jeda Waktu Paket Backward Minimum',
        'Fwd PSH Flags': 'Bendera PSH Forward',
        ' Bwd PSH Flags': 'Bendera PSH Backward',
        ' Fwd Header Length': 'Panjang Header Forward',
        ' Bwd Header Length': 'Panjang Header Backward',
        'Fwd Packets/s': 'Laju Paket Forward per Detik',
        ' Bwd Packets/s': 'Laju Paket Backward per Detik',
        ' Min Packet Length': 'Ukuran Paket Minimum',
        ' Max Packet Length': 'Ukuran Paket Maksimum',
        ' Packet Length Mean': 'Rata-rata Ukuran Paket Keseluruhan',
        ' Packet Length Std': 'Standar Deviasi Ukuran Paket Keseluruhan',
        ' Packet Length Variance': 'Variansi Ukuran Paket Keseluruhan',
        'FIN Flag Count': 'Jumlah Bendera FIN',
        ' SYN Flag Count': 'Jumlah Bendera SYN',
        ' RST Flag Count': 'Jumlah Bendera RST',
        ' PSH Flag Count': 'Jumlah Bendera PSH',
        ' ACK Flag Count': 'Jumlah Bendera ACK',
        ' URG Flag Count': 'Jumlah Bendera URG',
        ' CWE Flag Count': 'Jumlah Bendera CWE',
        ' ECE Flag Count': 'Jumlah Bendera ECE',
        ' Down/Up Ratio': 'Rasio Down/Up Paket',
        ' Average Packet Size': 'Ukuran Rata-rata Paket (Avg Packet Size)',
        ' Avg Fwd Segment Size': 'Rata-rata Ukuran Segmen Forward',
        ' Avg Bwd Segment Size': 'Rata-rata Ukuran Segmen Backward',
        'Subflow Fwd Packets': 'Total Paket Subflow Forward',
        ' Subflow Fwd Bytes': 'Total Byte Subflow Forward',
        ' Subflow Bwd Packets': 'Total Paket Subflow Backward',
        ' Subflow Bwd Bytes': 'Total Byte Subflow Backward',
        'Init_Win_bytes_forward': 'Ukuran TCP Window Awal Forward (Init Win Fwd)',
        ' Init_Win_bytes_backward': 'Ukuran TCP Window Awal Backward (Init Win Bwd)',
        ' act_data_pkt_fwd': 'Jumlah Paket Data Forward dengan Payload',
        ' min_seg_size_forward': 'Ukuran Segmen Forward Minimum',
        'Active Mean': 'Rata-rata Waktu Aktif Aliran',
        ' Active Std': 'Standar Deviasi Waktu Aktif Aliran',
        ' Active Max': 'Waktu Aktif Aliran Maksimum',
        ' Active Min': 'Waktu Aktif Aliran Minimum',
        'Idle Mean': 'Rata-rata Jeda Diam Jaringan',
        ' Idle Std': 'Standar Deviasi Jeda Diam Jaringan',
        ' Idle Max': 'Jeda Diam Jaringan Maksimum',
        ' Idle Min': 'Waktu Diam Jaringan Minimum'
    }
    
    # 1. Check if prediction is Heuristics-based
    if "Amp" in prediction or "Flood" in prediction:
        explanation_desc = ""
        if "DNS" in prediction:
            explanation_desc = (
                "Aliran data ini diklasifikasikan sebagai serangan DDoS (DNS Amplification) berdasarkan modul Heuristik DDoShield. "
                "Sistem mendeteksi aktivitas penelusuran DNS (Port 53) berbasis UDP yang sangat agresif dengan total paket melebihi "
                "ambang batas aman (> 50 paket) dalam durasi singkat. Serangan jenis ini menyalahgunakan server DNS terbuka untuk "
                "membanjiri kapasitas bandwidth jaringan kampus."
            )
        elif "NTP" in prediction:
            explanation_desc = (
                "Aliran data ini diklasifikasikan sebagai serangan DDoS (NTP Amplification) berdasarkan modul Heuristik DDoShield. "
                "Ditemukan trafik sinkronisasi waktu NTP (Port 123) yang tidak wajar dengan volume paket tinggi (> 50 paket). "
                "Penyerang memanfaatkan protokol NTP untuk melipatgandakan ukuran paket tanggapan demi melumpuhkan server target."
            )
        elif "SSDP" in prediction:
            explanation_desc = (
                "Aliran data ini diklasifikasikan sebagai serangan DDoS (SSDP Amplification) berdasarkan modul Heuristik DDoShield. "
                "Terdapat aktivitas pencarian perangkat SSDP/UPnP (Port 1900) dengan jumlah paket > 50. Protokol ini sering "
                "disalahgunakan untuk merekrut perangkat IoT terinfeksi guna mengirimkan trafik sampah berukuran besar."
            )
        elif "Memcached" in prediction:
            explanation_desc = (
                "Aliran data ini diklasifikasikan sebagai serangan DDoS (Memcached Amplification) berdasarkan modul Heuristik DDoShield. "
                "Terdeteksi trafik pada Port 11211 dengan volume paket > 30. Serangan ini sangat berbahaya karena faktor amplifikasi "
                "Memcached bisa melipatgandakan trafik hingga puluhan ribu kali lipat."
            )
        else:
            explanation_desc = (
                f"Aliran data ini diklasifikasikan sebagai serangan {prediction} oleh modul Heuristik DDoShield karena "
                f"laju pengiriman paket (throughput) dan total akumulasi paket yang berseliweran melebihi batas batas ambang volume aman "
                f"untuk protokol {meta['proto']}."
            )
            
        return jsonify({
            'method': 'Heuristic Rules Engine',
            'prediction': prediction,
            'explanation_text': explanation_desc,
            'contributions': [
                {
                    'feature': 'Protokol',
                    'value': meta['proto'],
                    'contrib_score': 1.0,
                    'direction': 'DDoS',
                    'percentage': 100,
                    'desc': 'Protokol pemicu aturan mitigasi khusus'
                },
                {
                    'feature': 'Port Tujuan',
                    'value': str(meta['dport']),
                    'contrib_score': 1.0,
                    'direction': 'DDoS',
                    'percentage': 100,
                    'desc': 'Port layanan kritis yang rentan amplifikasi'
                },
                {
                    'feature': 'Jumlah Paket',
                    'value': f"{meta['packet_count']} paket",
                    'contrib_score': 1.0,
                    'direction': 'DDoS',
                    'percentage': 100,
                    'desc': 'Volume paket melebihi ambang batas toleransi'
                }
            ]
        })
        
    # 2. Machine Learning Explanation (XGBoost)
    if model is None or scaler is None or selector is None or imputer is None:
        return jsonify({
            'method': 'Machine Learning (Simulasi)',
            'prediction': prediction,
            'explanation_text': 'Model machine learning tidak dimuat secara penuh. Penjelasan visual dinonaktifkan.',
            'contributions': []
        })
        
    flow_row = LAST_ANALYZED_DF.iloc[[flow_idx]]
    df_ordered = flow_row[FEATURE_COLUMNS].copy()
    
    # Run exact pipeline steps
    X_imputed = imputer.transform(df_ordered.values)
    X_scaled = scaler.transform(X_imputed)
    X_k30 = selector.transform(X_scaled)
    
    # Get SHAP values from XGBoost
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_k30)
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [FEATURE_COLUMNS[idx] for idx in selected_indices]
    
    contributions = []
    
    for i, feat_name in enumerate(selected_feature_names):
        shap_val = float(contribs[i])
        raw_val = flow_row.iloc[0][feat_name]
        
        formatted_val = str(raw_val)
        if isinstance(raw_val, float):
            if raw_val.is_integer():
                formatted_val = str(int(raw_val))
            else:
                formatted_val = f"{raw_val:.4f}"
        
        if 'Duration' in feat_name or 'IAT' in feat_name:
            try:
                us_val = float(raw_val)
                if us_val >= 1000000:
                    formatted_val = f"{us_val/1000000:.2f} s"
                elif us_val >= 1000:
                    formatted_val = f"{us_val/1000:.2f} ms"
                else:
                    formatted_val = f"{us_val:.1f} µs"
            except:
                pass
        elif 'Bytes' in feat_name or 'Length' in feat_name:
            try:
                b_val = float(raw_val)
                if b_val >= 1048576:
                    formatted_val = f"{b_val/1048576:.2f} MB"
                elif b_val >= 1024:
                    formatted_val = f"{b_val/1024:.2f} KB"
                else:
                    formatted_val = f"{int(b_val)} B"
            except:
                pass
        elif 'Packets/s' in feat_name:
            try:
                formatted_val = f"{float(raw_val):.1f} pkt/s"
            except:
                pass
        elif 'Bytes/s' in feat_name:
            try:
                bs_val = float(raw_val)
                if bs_val >= 1048576:
                    formatted_val = f"{bs_val/1048576:.2f} MB/s"
                elif bs_val >= 1024:
                    formatted_val = f"{bs_val/1024:.2f} KB/s"
                else:
                    formatted_val = f"{bs_val:.1f} B/s"
            except:
                pass
                
        friendly_name = FEATURE_NAME_MAP.get(feat_name, feat_name.strip())
        direction = "DDoS" if shap_val > 0 else "BENIGN"
        
        contributions.append({
            'feature': friendly_name,
            'raw_name': feat_name,
            'value': formatted_val,
            'contrib_score': shap_val,
            'abs_score': abs(shap_val),
            'direction': direction
        })
        
    contributions.sort(key=lambda x: x['abs_score'], reverse=True)
    top_contributions = contributions[:5]
    
    max_abs_score = max(x['abs_score'] for x in top_contributions) if top_contributions else 1.0
    for c in top_contributions:
        c['percentage'] = int((c['abs_score'] / max_abs_score) * 100) if max_abs_score > 0 else 0
        
    base_val = float(contribs[-1])
    sum_contribs = float(np.sum(contribs[:-1]))
    final_output = base_val + sum_contribs
    
    explanation_text = ""
    if prediction == "DDoS (XGBoost)":
        explanation_text = (
            f"Model XGBoost mengklasifikasikan aliran data ini sebagai <strong>serangan DDoS</strong> (Log-odds: {final_output:.2f}). "
            f"Faktor terbesar yang mendorong keputusan ini adalah <strong>{top_contributions[0]['feature']}</strong> dengan nilai <strong>{top_contributions[0]['value']}</strong> "
            f"yang memberikan dorongan ke arah DDoS sebesar +{top_contributions[0]['contrib_score']:.2f}. "
            f"Akumulasi ciri statistik dari fitur-fitur tersebut sangat mirip dengan pola serangan <em>DDoS Flooding</em> pada dataset pelatihan."
        )
    else:
        explanation_text = (
            f"Model XGBoost mengklasifikasikan aliran data ini sebagai <strong>lalu lintas normal (BENIGN)</strong> (Log-odds: {final_output:.2f}). "
            f"Faktor yang paling meyakinkan model adalah <strong>{top_contributions[0]['feature']}</strong> dengan nilai <strong>{top_contributions[0]['value']}</strong> "
            f"yang memberikan dorongan ke arah BENIGN sebesar {top_contributions[0]['contrib_score']:.2f}. "
            f"Pola aliran data ini dinilai wajar karena ukuran paket, jeda kedatangan, dan laju data tidak menunjukkan perilaku anomali."
        )
        
    return jsonify({
        'method': 'XGBoost Machine Learning (Explainable AI)',
        'prediction': prediction,
        'explanation_text': explanation_text,
        'contributions': top_contributions
    })

if __name__ == '__main__':
    print("[DDoShield] Starting development server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
