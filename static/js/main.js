document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileSelectedInfo = document.getElementById('file-selected-info');
    const selectedFileName = document.getElementById('selected-file-name');
    const selectedFileSize = document.getElementById('selected-file-size');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const uploadForm = document.getElementById('upload-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    // Sniffing DOM Elements
    const tabUploadBtn = document.getElementById('tab-upload-btn');
    const tabSniffBtn = document.getElementById('tab-sniff-btn');
    const uploadTabContent = document.getElementById('upload-tab-content');
    const sniffTabContent = document.getElementById('sniff-tab-content');
    const startSniffBtn = document.getElementById('start-sniff-btn');
    const sniffDuration = document.getElementById('sniff-duration');
    const sniffControlsWrapper = document.getElementById('sniff-controls-wrapper');
    const sniffScanningContainer = document.getElementById('sniff-scanning-container');
    const sniffCountdown = document.getElementById('sniff-countdown');
    
    const loadingContainer = document.getElementById('loading-container');
    const loadingStep = document.getElementById('loading-step');
    const progressBarFill = document.getElementById('progress-bar-fill');
    
    const resultsContainer = document.getElementById('results-container');
    const statTotalFlows = document.getElementById('stat-total-flows');
    const statDdosFlows = document.getElementById('stat-ddos-flows');
    const statBenignFlows = document.getElementById('stat-benign-flows');
    const statLatency = document.getElementById('stat-latency');
    const statMsFlow = document.getElementById('stat-ms-flow');
    
    const threatPct = document.getElementById('threat-pct');
    const threatLbl = document.getElementById('threat-lbl');
    const threatDesc = document.getElementById('threat-desc');
    const threatDial = document.querySelector('.threat-dial');
    
    const statThroughput = document.getElementById('stat-throughput');
    const barCpu = document.getElementById('bar-cpu');
    const statCpu = document.getElementById('stat-cpu');
    const barRam = document.getElementById('bar-ram');
    const statRam = document.getElementById('stat-ram');
    
    const flowTableBody = document.getElementById('flow-table-body');
    const tableSearch = document.getElementById('table-search');
    const downloadReport = document.getElementById('download-report');
    
    // XAI Modal DOM Elements
    const xaiModal = document.getElementById('xai-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const xaiFlowIdent = document.getElementById('xai-flow-ident');
    const xaiFlowMethod = document.getElementById('xai-flow-method');
    const xaiFlowPrediction = document.getElementById('xai-flow-prediction');
    const xaiFeaturesContainer = document.getElementById('xai-features-container');
    const xaiExplanationText = document.getElementById('xai-explanation-text');
    
    let trafficChart = null;
    let allFlows = []; // To store flow list globally for filtering and exporting
    
    // Close Modal Listeners
    closeModalBtn.addEventListener('click', () => {
        xaiModal.classList.add('hidden');
    });
    xaiModal.addEventListener('click', (e) => {
        if (e.target === xaiModal) {
            xaiModal.classList.add('hidden');
        }
    });
    let selectedFile = null; // Global variable to store selected file
    
    // Tab Switching Handlers
    tabUploadBtn.addEventListener('click', () => {
        tabUploadBtn.classList.add('active');
        tabSniffBtn.classList.remove('active');
        uploadTabContent.classList.remove('hidden');
        sniffTabContent.classList.add('hidden');
    });
    
    tabSniffBtn.addEventListener('click', () => {
        tabSniffBtn.classList.add('active');
        tabUploadBtn.classList.remove('active');
        sniffTabContent.classList.remove('hidden');
        uploadTabContent.classList.add('hidden');
    });
    
    // Live Sniffing Handler
    startSniffBtn.addEventListener('click', () => {
        const duration = parseInt(sniffDuration.value);
        sniffControlsWrapper.classList.add('hidden');
        sniffScanningContainer.classList.remove('hidden');
        
        // Start countdown
        sniffCountdown.textContent = duration;
        let timeLeft = duration;
        const countdownInterval = setInterval(() => {
            timeLeft--;
            sniffCountdown.textContent = Math.max(0, timeLeft);
            if (timeLeft <= 0) {
                clearInterval(countdownInterval);
            }
        }, 1000);
        
        fetch('/api/live_sniff', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ duration: duration })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => { throw new Error(data.error || 'Terjadi kesalahan sistem saat sniffing') });
            }
            return response.json();
        })
        .then(data => {
            clearInterval(countdownInterval);
            sniffScanningContainer.classList.add('hidden');
            sniffControlsWrapper.classList.remove('hidden');
            resultsContainer.classList.remove('hidden');
            
            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
            
            displayResults(data);
        })
        .catch(err => {
            clearInterval(countdownInterval);
            sniffScanningContainer.classList.add('hidden');
            sniffControlsWrapper.classList.remove('hidden');
            alert(`Sniffing Gagal: ${err.message}`);
        });
    });
    
    // Drag & Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    function handleFileSelect(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (ext !== 'pcap' && ext !== 'pcapng' && ext !== 'csv') {
            alert('Format file tidak didukung! Gunakan file .pcap, .pcapng, atau .csv');
            return;
        }
        
        selectedFile = file; // Store file globally
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = formatBytes(file.size);
        
        dropZone.classList.add('hidden');
        fileSelectedInfo.classList.remove('hidden');
    }
    
    removeFileBtn.addEventListener('click', () => {
        fileInput.value = '';
        selectedFile = null; // Clear global file
        fileSelectedInfo.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });
    
    // Form Submit
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const file = selectedFile; // Use global file variable
        if (!file) return;
        
        fileSelectedInfo.classList.add('hidden');
        loadingContainer.classList.remove('hidden');
        
        // Start simulated progress
        progressBarFill.style.width = '0%';
        let progress = 0;
        const steps = [
            { limit: 25, label: "Membaca data file capture..." },
            { limit: 55, label: "Mengekstraksi flow dan menyusun dataset..." },
            { limit: 85, label: "Menjalankan pipeline preprocessing & scaling..." },
            { limit: 95, label: "Mengklasifikasikan trafik dengan model XGBoost K=30..." }
        ];
        
        let stepIdx = 0;
        const progressInterval = setInterval(() => {
            if (progress < 95) {
                progress += Math.floor(Math.random() * 5) + 1;
                
                // Update text based on progress level
                if (stepIdx < steps.length && progress >= steps[stepIdx].limit) {
                    loadingStep.textContent = steps[stepIdx].label;
                    stepIdx++;
                }
                
                progressBarFill.style.width = `${Math.min(progress, 95)}%`;
            }
        }, 150);
        
        // Prepare data
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => { throw new Error(data.error || 'Server error occurred') });
            }
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            progressBarFill.style.width = '100%';
            loadingStep.textContent = "Analisis Selesai! Menyusun Dashboard...";
            
            setTimeout(() => {
                loadingContainer.classList.add('hidden');
                resultsContainer.classList.remove('hidden');
                
                // Scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
                
                displayResults(data);
            }, 600);
        })
        .catch(err => {
            clearInterval(progressInterval);
            loadingContainer.classList.add('hidden');
            fileSelectedInfo.classList.remove('hidden');
            alert(`Error: ${err.message}`);
        });
    });
    
    function displayResults(data) {
        // Tag flows with their index in the backend flows list
        data.flows.forEach((flow, idx) => {
            flow.original_idx = idx;
        });
        allFlows = data.flows;
        
        // Populate stats cards
        statTotalFlows.textContent = formatNumber(data.total_flows);
        statDdosFlows.textContent = formatNumber(data.ddos_count);
        statBenignFlows.textContent = formatNumber(data.benign_count);
        statLatency.textContent = `${data.processing_time_ms} ms`;
        if (statMsFlow) {
            statMsFlow.textContent = `${data.ms_per_flow.toFixed(4)} ms/flow`;
        }
        statThroughput.textContent = `${formatNumber(data.throughput_flows_s)} flow/s`;
        
        // Update threat indicator dial
        const pct = data.threat_percent;
        threatPct.textContent = `${pct}%`;
        
        // Dial conic-gradient background
        // Colors: Green (#10b981) for benign, Red (#ef4444) for DDoS
        threatDial.style.background = `conic-gradient(var(--danger) ${pct}%, rgba(255, 255, 255, 0.05) ${pct}%)`;
        
        // Badge color and label
        threatLbl.textContent = data.threat_level;
        threatLbl.className = 'threat-dial-label';
        
        if (data.threat_level === 'CRITICAL') {
            threatLbl.classList.add('status-ddos');
            threatDesc.textContent = `Peringatan Kritis! Ditemukan ${pct}% lalu lintas DDoS. Lakukan mitigasi pemblokiran segera!`;
        } else if (data.threat_level === 'WARNING') {
            threatLbl.classList.add('status-ddos');
            threatLbl.style.background = 'rgba(245, 158, 11, 0.15)';
            threatLbl.style.color = 'var(--warning)';
            threatLbl.style.animation = 'none';
            threatDesc.textContent = `Lalu lintas mencurigakan terdeteksi (${pct}% DDoS). Monitor port tujuan yang diserang.`;
        } else {
            threatLbl.classList.add('status-benign');
            threatDesc.textContent = "Kondisi lalu lintas jaringan aman dan normal. Tidak ada serangan terdeteksi.";
        }
        
        // System metrics
        statCpu.textContent = `${data.cpu_usage}%`;
        barCpu.style.width = `${data.cpu_usage}%`;
        statRam.textContent = `${data.ram_usage}%`;
        barRam.style.width = `${data.ram_usage}%`;
        
        // Draw Pie Chart
        renderChart(data.benign_count, data.ddos_count);
        
        // Draw table
        renderTable(allFlows);
    }
    
    function renderChart(benign, ddos) {
        if (trafficChart) {
            trafficChart.destroy();
        }
        
        const ctx = document.getElementById('trafficChart').getContext('2d');
        trafficChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Lalu Lintas Aman (BENIGN)', 'Serangan DDoS'],
                datasets: [{
                    data: [benign, ddos],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#0d1220',
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#9ca3af',
                            font: {
                                family: 'Inter',
                                size: 11
                            },
                            padding: 15
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }
    
    function renderTable(flows) {
        flowTableBody.innerHTML = '';
        if (flows.length === 0) {
            flowTableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Tidak ada flow cocok ditemukan</td></tr>`;
            return;
        }
        
        flows.forEach((flow, index) => {
            const tr = document.createElement('tr');
            tr.setAttribute('data-flow-index', flow.original_idx);
            tr.style.cursor = 'pointer';
            
            const badgeClass = flow.prediction.includes('DDoS') ? 'status-ddos' : 'status-benign';
            
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td><span class="badge" style="background: rgba(255,255,255,0.03); color: var(--text-main); border: 1px solid var(--border-color); font-size: 0.7rem; padding: 0.2rem 0.4rem;">${flow.proto}</span></td>
                <td>${flow.src_ip}:${flow.sport}</td>
                <td>${flow.dst_ip}</td>
                <td><strong style="color: var(--primary);">${flow.dport}</strong></td>
                <td>${flow.flow_duration_s}</td>
                <td>${flow.packet_count}</td>
                <td>${formatBytes(flow.byte_count)}</td>
                <td><span class="status-badge ${badgeClass}">${flow.prediction}</span></td>
            `;
            
            tr.addEventListener('click', () => {
                const flowIdx = tr.getAttribute('data-flow-index');
                showExplanation(flowIdx);
            });
            
            flowTableBody.appendChild(tr);
        });
    }
    
    // Search/Filtering
    tableSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        if (!query) {
            renderTable(allFlows);
            return;
        }
        
        const filtered = allFlows.filter(flow => {
            return flow.src_ip.toLowerCase().includes(query) ||
                   flow.dst_ip.toLowerCase().includes(query) ||
                   flow.dport.toString().includes(query) ||
                   flow.sport.toString().includes(query) ||
                   flow.proto.toLowerCase().includes(query) ||
                   flow.prediction.toLowerCase().includes(query);
        });
        renderTable(filtered);
    });
    
    // Export Table to CSV
    downloadReport.addEventListener('click', () => {
        if (allFlows.length === 0) return;
        
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "No,Protocol,Source IP,Source Port,Destination IP,Destination Port,Duration(s),Packets,Bytes,Prediction\n";
        
        allFlows.forEach((f, idx) => {
            const row = [
                idx + 1,
                f.proto,
                f.src_ip,
                f.sport,
                f.dst_ip,
                f.dport,
                f.flow_duration_s,
                f.packet_count,
                f.byte_count,
                f.prediction
            ].join(",");
            csvContent += row + "\n";
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `DDoShield_Report_${new Date().toISOString().slice(0,10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    
    // Utils
    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }
    
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
    }

    function showExplanation(flowIdx) {
        // Set loading state in modal
        xaiFlowIdent.textContent = "Loading...";
        xaiFlowMethod.textContent = "-";
        xaiFlowPrediction.textContent = "-";
        xaiFlowPrediction.className = "summary-val status-badge";
        xaiFeaturesContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 1rem;"><i class="fa-solid fa-spinner fa-spin"></i> Menganalisis keputusan model...</div>';
        xaiExplanationText.innerHTML = "";
        
        xaiModal.classList.remove('hidden');
        
        fetch(`/api/explain/${flowIdx}`)
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => { throw new Error(data.error || 'Gagal mengambil penjelasan') });
            }
            return response.json();
        })
        .then(data => {
            const flowMeta = allFlows[flowIdx];
            
            xaiFlowIdent.textContent = `${flowMeta.src_ip}:${flowMeta.sport} -> ${flowMeta.dst_ip} (${flowMeta.proto})`;
            xaiFlowMethod.textContent = data.method;
            xaiFlowPrediction.textContent = data.prediction;
            
            const badgeClass = data.prediction.includes('DDoS') ? 'status-ddos' : 'status-benign';
            xaiFlowPrediction.className = `summary-val status-badge ${badgeClass}`;
            
            xaiFeaturesContainer.innerHTML = '';
            
            if (data.contributions.length === 0) {
                xaiFeaturesContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 1rem;">Tidak ada kontribusi fitur numerik untuk metode ini.</div>';
            } else {
                data.contributions.forEach(c => {
                    const row = document.createElement('div');
                    row.className = 'xai-feature-row';
                    
                    const isDdos = c.direction === 'DDoS';
                    const barColorClass = isDdos ? 'xai-bar-ddos' : 'xai-bar-benign';
                    const textClass = isDdos ? 'contrib-ddos' : 'contrib-benign';
                    const sign = isDdos ? '+' : '-';
                    
                    row.innerHTML = `
                        <div class="xai-feature-header">
                            <span class="xai-feature-name">${c.feature}</span>
                            <span class="xai-feature-val">Nilai: <strong>${c.value}</strong></span>
                        </div>
                        <div class="xai-bar-outer">
                            <div class="xai-bar-fill ${barColorClass}" style="width: 0%;"></div>
                        </div>
                        <div class="xai-feature-header">
                            <span class="xai-feature-contrib-text ${textClass}">Dorongan ke arah: ${c.direction}</span>
                            <span class="xai-feature-contrib-text ${textClass}" style="font-family: monospace;">${sign}${Math.abs(c.contrib_score).toFixed(4)}</span>
                        </div>
                    `;
                    
                    xaiFeaturesContainer.appendChild(row);
                    
                    // Animate bar width expansion
                    setTimeout(() => {
                        row.querySelector('.xai-bar-fill').style.width = `${c.percentage}%`;
                    }, 50);
                });
            }
            
            xaiExplanationText.innerHTML = data.explanation_text;
        })
        .catch(err => {
            xaiFlowIdent.textContent = "Error";
            xaiFeaturesContainer.innerHTML = `<div style="text-align: center; color: var(--danger); padding: 1rem;"><i class="fa-solid fa-triangle-exclamation"></i> Gagal memuat penjelasan: ${err.message}</div>`;
        });
    }
});
