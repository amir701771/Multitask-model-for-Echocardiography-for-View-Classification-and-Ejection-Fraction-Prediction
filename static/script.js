document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    // Toast Notification
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        // Trigger reflow
        toast.offsetHeight;

        toast.classList.add('show');

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    window.showToast = showToast; // Expose globally
    function switchSection(sectionId) {
        // Update Nav
        navLinks.forEach(link => {
            if (link.dataset.section === sectionId) link.classList.add('active');
            else link.classList.remove('active');
        });

        // Update Sections
        sections.forEach(sec => {
            sec.style.display = 'none';
        });
        document.getElementById(`section-${sectionId}`).style.display = 'block';

        // Load History if needed
        if (sectionId === 'history') {
            loadHistory();
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            // Only intercept navigation links that have a section defined
            if (link.dataset.section) {
                e.preventDefault();
                switchSection(link.dataset.section);
            }
            // Otherwise (like Logout), let the default action happen (server request)
        });
    });

    // Upload & Analysis Logic
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultSection = document.getElementById('result-section');
    const videoPreview = document.getElementById('preview-video');

    // Upload Panel Elements
    const filePreviewArea = document.getElementById('file-preview-area');
    const filenameDisplay = document.getElementById('filename-display');
    const loader = document.getElementById('loader');

    // Output Elements
    const outputContent = document.getElementById('output-content');
    const predView = document.getElementById('pred-view');
    const predConf = document.getElementById('pred-conf');
    const predEF = document.getElementById('pred-ef');
    const efBadge = document.getElementById('ef-badge');

    // Drag & Drop Events
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
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        // Ensure we are on the upload section
        switchSection('upload');

        // UI Updates in Sidebar
        filePreviewArea.style.display = 'block';
        filenameDisplay.textContent = file.name;
        dropZone.style.display = 'block';

        // Show result section but hide content until ready
        resultSection.style.display = 'block';
        loader.style.display = 'block';
        // Add text feedback for conversion
        loader.innerHTML = '<div class="spinner"></div><p class="status-text">Uploading and Optimizing Video... (This may take a moment)</p>';
        outputContent.style.display = 'none';

        // Set a loading placeholder or keep blank until server returns valid URL
        // We do NOT set src here immediately to avoid playing unsupported formats
        videoPreview.src = "";
        videoPreview.load();

        // Upload and Analyze
        uploadVideo(file);
    }

    function uploadVideo(file) {
        const formData = new FormData();
        formData.append('video', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    loader.style.display = 'none';
                    return;
                }
                // Display Results
                loader.style.display = 'none';
                outputContent.style.display = 'flex';

                // Update Video Player with Server URL (Converted if needed)
                if (data.video_url) {
                    videoPreview.src = data.video_url;
                    videoPreview.muted = true; // Mute to allow autoplay
                    videoPreview.load(); // Force reload
                    videoPreview.play().catch(e => {
                        console.log("Auto-play prevented:", e);
                        // Unmute if autoplay failed
                        videoPreview.muted = false;
                    });
                }

                // Animate Text / Set Values
                predView.textContent = data.label;
                predConf.textContent = data.confidence;

                // EF Display Logic
                if (data.ef_message) {
                    predEF.textContent = data.ef_message;

                    if (data.ef_category && data.ef_category !== "N/A") {
                        efBadge.style.display = 'inline-block';
                        efBadge.textContent = data.ef_category;

                        // Badge Colors
                        if (data.ef_category === 'Normal') {
                            efBadge.style.backgroundColor = '#10b981'; // Green
                            efBadge.style.color = 'white';
                        } else if (data.ef_category === 'Mildly Reduced') {
                            efBadge.style.backgroundColor = '#f59e0b'; // Amber
                            efBadge.style.color = 'black';
                        } else {
                            efBadge.style.backgroundColor = '#ef4444'; // Red
                            efBadge.style.color = 'white';
                        }
                    } else {
                        efBadge.style.display = 'none';
                    }
                } else {
                    predEF.textContent = "--";
                    efBadge.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                loader.innerHTML = '<p class="status-text" style="color:#ef4444">Analysis Failed</p>';
            });
    }

    // History Logic
    function loadHistory() {
        fetch('/history')
            .then(response => response.json())
            .then(data => {
                const tbody = document.getElementById('history-list');
                tbody.innerHTML = '';

                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted)">No history available</td></tr>';
                    return;
                }

                data.forEach(item => {
                    const tr = document.createElement('tr');

                    // Format EF string for table
                    let efDisplay = "N/A";
                    if (item.ef_value != null) {
                        efDisplay = `${item.ef_value}%`;
                        if (item.ef_category) {
                            efDisplay += ` (${item.ef_category})`;
                        }
                    } else if (item.prediction === 'A2C' || item.prediction === 'A4C') {
                        // Backwards compatibility if column empty but view matches
                        if (item.ef_value === null && (item.prediction === 'A2C' || item.prediction === 'A4C')) {
                            efDisplay = "Not calc";
                        }
                    }

                    tr.innerHTML = `
                        <td>${new Date(item.timestamp).toLocaleString()}</td>
                        <td>${item.filename}</td>
                        <td style="font-weight:bold; color:var(--secondary)">${item.prediction}</td>
                        <td>${(item.confidence * 100).toFixed(1)}%</td>
                        <td>${efDisplay}</td>
                        <td>
                            <button type="button" class="delete-btn" data-id="${item.id}">
                                <i class="fa-solid fa-trash"></i>
                            </button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Failed to load history:", err));
    }



    // --- Training Logic ---
    let trainingPollInterval = null;

    window.startTraining = function () {
        if (!confirm("Are you sure you want to start model training? This may take some time.")) return;

        const btn = document.getElementById('train-btn');
        const area = document.getElementById('training-status-area');

        btn.disabled = true;
        btn.style.opacity = "0.6";
        area.style.display = "block";
        document.getElementById('train-status-text').textContent = "Requesting Server...";

        // Use the new simplified endpoint
        fetch('/train', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                // Look for 'started' or 'already_training'
                if (data.status === 'started' || data.status === 'already_training') {
                    showToast(data.message || "Training started in background", "success");
                    pollTrainingStatus();
                } else {
                    showToast("Failed to start: " + (data.message || "Unknown error"), "error");
                    btn.disabled = false;
                    btn.style.opacity = "1";
                }
            })
            .catch(err => {
                console.error(err);
                showToast("Network Error: Could not start training", "error");
                btn.disabled = false;
                btn.style.opacity = "1";
            });
    };

    window.stopTraining = function () {
        if (!confirm("Are you sure you want to stop the training process?")) return;

        const stopBtn = document.getElementById('stop-train-btn');
        stopBtn.disabled = true;
        stopBtn.style.opacity = "0.6";

        fetch('/api/stop-training', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    showToast("Stop signal sent. Training will stop after current epoch.", "success");
                } else {
                    showToast(data.message || "Failed to stop training", "error");
                    stopBtn.disabled = false;
                    stopBtn.style.opacity = "1";
                }
            })
            .catch(err => {
                console.error(err);
                showToast("Network Error: Could not stop training", "error");
                stopBtn.disabled = false;
                stopBtn.style.opacity = "1";
            });
    };

    function pollTrainingStatus() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);

        const btn = document.getElementById('train-btn');
        const stopBtn = document.getElementById('stop-train-btn');
        const progressBar = document.getElementById('train-progress-bar');
        const statusText = document.getElementById('train-status-text');
        const epochText = document.getElementById('train-epoch-text');
        const lossText = document.getElementById('train-loss');
        const accText = document.getElementById('train-acc');
        const msgText = document.getElementById('train-message');

        trainingPollInterval = setInterval(() => {
            fetch('/api/training-status')
                .then(res => res.json())
                .then(state => {
                    // status: "idle", "training", "completed", "failed"

                    if (state.status === 'idle') {
                        // Probably hasn't started or reset
                        stopBtn.style.display = 'none';
                        return;
                    }

                    // Show stop button only when training
                    if (state.status === 'training') {
                        stopBtn.style.display = 'inline-block';
                    } else {
                        stopBtn.style.display = 'none';
                    }

                    // Update UI
                    statusText.textContent = state.status.toUpperCase();
                    msgText.textContent = state.message || "";

                    if (state.total_epochs > 0) {
                        epochText.textContent = `Epoch: ${state.epoch}/${state.total_epochs}`;
                        // Approximate progress calculation
                        const pct = Math.min(100, Math.round((state.epoch / state.total_epochs) * 100));
                        progressBar.style.width = `${pct}%`;
                    }

                    lossText.textContent = state.loss.toFixed(4);
                    accText.textContent = (state.accuracy * 100).toFixed(1) + "%";

                    if (state.status === 'completed') {
                        clearInterval(trainingPollInterval);
                        btn.disabled = false;
                        btn.style.opacity = "1";
                        btn.innerHTML = '<i class="fa-solid fa-brain"></i> Train Again';
                        stopBtn.style.display = 'none';
                        progressBar.style.width = "100%";
                        progressBar.style.backgroundColor = "var(--success)";
                        statusText.style.color = "var(--success)";
                        showToast("Training Completed Successfully!", "success");
                    } else if (state.status === 'failed') {
                        clearInterval(trainingPollInterval);
                        btn.disabled = false;
                        btn.style.opacity = "1";
                        stopBtn.style.display = 'none';
                        statusText.style.color = "#ef4444";
                        progressBar.style.backgroundColor = "#ef4444";
                        showToast("Training Failed.", "error");
                    } else if (state.status === 'stopped') {
                        clearInterval(trainingPollInterval);
                        btn.disabled = false;
                        btn.style.opacity = "1";
                        btn.innerHTML = '<i class="fa-solid fa-brain"></i> Resume Training';
                        stopBtn.style.display = 'none';
                        stopBtn.disabled = false;
                        stopBtn.style.opacity = "1";
                        statusText.style.color = "#f59e0b";
                        progressBar.style.backgroundColor = "#f59e0b";
                        showToast("Training Stopped by User", "success");
                    }
                })
                .catch(err => console.error("Polling error:", err));
        }, 2000);
    }
}); // End of DOMContentLoaded

// --- Global Delete Handler (Outside DOMContentLoaded to ensure attachment) ---
document.addEventListener('click', function (e) {
    // Check if clicked element is a delete button or inside one
    const btn = e.target.closest('.delete-btn');
    if (!btn) return;

    // Prevent default button behavior
    e.preventDefault();
    e.stopPropagation();

    console.log("Delete button clicked", btn);

    const id = btn.dataset.id;
    if (!id) {
        console.error("No ID found on delete button");
        return;
    }

    if (!confirm('Are you sure you want to delete this analysis?')) return;

    const row = btn.closest('tr');
    if (row) row.style.opacity = '0.5';

    fetch(`/api/history/${id}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' }
    })
        .then(async response => {
            if (response.status === 401) {
                window.location.href = '/login';
                throw new Error("Unauthorized");
            }
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');
            return data;
        })
        .then(data => {
            if (data.success) {
                // Function to show toast accessed from window if needed, or re-implement
                // For now, simple alert or we assume showToast is global? 
                // Warning: showToast is defined INSIDE DOMContentLoaded. It is NOT global.
                // I need to handle UI feedback without showToast or move showToast out.
                // I'll make showToast global in the next step or here.

                if (row) {
                    row.remove();
                    // Check if table empty logic duplicated? 
                    // Minimal inline fallback for empty table check
                    const tbody = document.getElementById('history-list');
                    if (tbody && tbody.children.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-muted)">No history available</td></tr>';
                    }
                } else {
                    // If we can't find row, reload. But loadHistory is also scoped.
                    window.location.reload();
                }
            } else {
                if (row) row.style.opacity = '1';
                alert('Failed to delete: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => {
            if (row) row.style.opacity = '1';
            if (err.message !== "Unauthorized") {
                console.error('Delete error:', err);
                alert('Error: ' + err.message);
            }
        });
});
