# DDoShield

DDoShield adalah perangkat lunak berbasis web untuk mendeteksi dan melakukan analisis forensik jaringan terhadap serangan *Distributed Denial of Service* (DDoS) secara luring (*offline*). DDoShield menggabungkan kekuatan ekstraksi fitur PCAP otomatis, *Machine Learning* terarah (XGBoost), seleksi fitur (*Information Gain*), dan *Explainable AI* (TreeSHAP) untuk menyajikan laporan serangan secara instan dan visual.

## Fitur Utama
1. **Ekstraksi Aliran Otomatis**: Memparsing file `.pcap` mentah menjadi aliran statistik (*flow statistics*) 5-tuple menggunakan *Scapy*.
2. **Sistem Deteksi Hibrida**: Menggabungkan deteksi cepat berbasis volume/heuristik untuk serangan jenis *amplification* dan model *Machine Learning* untuk serangan tingkat lanjut.
3. **Machine Learning Terarah**: Menggunakan XGBoost yang dioptimasi dengan 30 fitur terpilih (akurasi ~100% pada dataset CIC-IDS2017).
4. **Forensik Berbasis Aliran**: Menggunakan visualisasi *TreeSHAP* untuk mengungkap alasan komputasional di balik vonis DDoS, sehingga analis tidak perlu lagi melakukan *Deep Packet Inspection* membaca heksadesimal secara manual.

## Struktur Direktori
- `app.py`: *Backend* web server (Flask) dan logika mesin utama (Scapy, XGBoost, SHAP).
- `models/`: Folder tempat menyimpan *pipeline* Scikit-Learn `.pkl` terkompresi.
- `templates/` & `static/`: Komponen antarmuka web (HTML, CSS, JS).
- `paper/`: Berkas dokumentasi utama (format LaTeX) untuk dipublikasikan.
- `experiments/`: Skrip Python komparasi riset dan pengukuran (*benchmarking*) XGBoost vs Random Forest.

## Penggunaan
Jalankan aplikasi di *localhost*:
```bash
python app.py
```
Akses dasbor web melalui browser di alamat `http://localhost:5000`.
