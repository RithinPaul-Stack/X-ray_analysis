/**
 * X-Ray Analyzer - Frontend JavaScript
 * Handles file upload, API communication, and UI updates
 */

// State
let selectedFile = null;
let analysisResults = null;
let currentFilename = null;
let yoloResults = null;
let annotatedImageUrl = null;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const dropzoneFile = document.getElementById('dropzone-file');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const previewCard = document.getElementById('preview-card');
const previewImage = document.getElementById('preview-image');
const loadingOverlay = document.getElementById('loading-overlay');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const analysisIdSpan = document.getElementById('analysis-id');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropzone();
});

/**
 * Setup dropzone event listeners
 */
function setupDropzone() {
    // Click to open file dialog
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop events
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');

        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type (including DICOM)
    const validTypes = ['image/jpeg', 'image/png', 'application/dicom'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        showError('Invalid file type. Please upload DICOM, PNG, or JPEG images.');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Update UI
    dropzone.classList.add('has-file');
    dropzoneEmpty.style.display = 'none';
    dropzoneFile.style.display = 'flex';
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    // Show preview (DICOM files can't be previewed directly in browser)
    if (fileExtension === '.dcm' || fileExtension === '.dicom') {
        // For DICOM files, show a placeholder
        previewImage.src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiB2aWV3Qm94PSIwIDAgMjAwIDIwMCI+PHJlY3QgZmlsbD0iIzFhMWEyZSIgd2lkdGg9IjIwMCIgaGVpZ2h0PSIyMDAiLz48dGV4dCB4PSI1MCUiIHk9IjQwJSIgZmlsbD0iIzY2NjY4OCIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjE0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5ESUNPTSBGaWxlPC90ZXh0Pjx0ZXh0IHg9IjUwJSIgeT0iNTUlIiBmaWxsPSIjNDQ0NDY2IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTIiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlByZXZpZXcgYWZ0ZXIgYW5hbHlzaXM8L3RleHQ+PC9zdmc+';
        previewCard.style.display = 'block';
    } else {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewCard.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Enable buttons
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;

    // Clear previous results
    hideResults();
    hideError();
}

/**
 * Analyze the uploaded image
 */
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an X-ray image first.');
        return;
    }

    // Show loading overlay
    showLoading();

    try {
        const formData = new FormData();
        formData.append('xray_image', selectedFile);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            analysisResults = data.results;
            currentFilename = data.filename;
            displayResults(data.results, data.analysis_id);
            // Enable deep analysis and PDF download buttons after successful analysis
            document.getElementById('deep-analysis-btn').disabled = false;
            document.getElementById('download-pdf-btn').disabled = false;
        } else {
            showError(data.error || 'Analysis failed. Please try again.');
        }
    } catch (error) {
        showError('Unable to connect to server. Please ensure the backend is running.');
        console.error('Analysis error:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Display analysis results
 */
function displayResults(results, analysisId) {
    // Hide placeholder, show results
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    // Show analysis ID
    if (analysisId) {
        analysisIdSpan.textContent = `ID: ${analysisId}`;
        analysisIdSpan.style.display = 'block';
    }

    const isAbnormal = results.status === 'ABNORMAL';

    // Update status banner
    const statusBanner = document.getElementById('status-banner');
    statusBanner.className = `status-banner ${isAbnormal ? 'status-abnormal' : 'status-normal'}`;

    const statusIcon = document.getElementById('status-icon');
    statusIcon.innerHTML = isAbnormal
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
           </svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
           </svg>`;

    document.getElementById('status-title').textContent =
        isAbnormal ? 'Abnormal Findings Detected' : 'Normal Study';
    document.getElementById('status-message-text').textContent = results.status_message;
    document.getElementById('stat-findings').textContent = results.abnormal_count;
    document.getElementById('stat-screened').textContent = results.total_pathologies;

    // Display top findings
    const findingsSection = document.getElementById('findings-section');
    const findingsList = document.getElementById('findings-list');

    if (results.top_findings && results.top_findings.length > 0) {
        findingsSection.style.display = 'block';
        findingsList.innerHTML = results.top_findings.map(finding => `
            <div class="finding-card">
                <div class="finding-header">
                    <span class="severity-badge severity-${finding.severity}">${finding.severity.toUpperCase()}</span>
                    <span class="finding-name">${finding.name}</span>
                    <span class="finding-score">${finding.percentage}%</span>
                </div>
                <div class="finding-progress">
                    <div class="progress-bar severity-${finding.severity}" style="width: ${Math.min(finding.percentage, 100)}%"></div>
                </div>
                <p class="finding-description">${finding.description}</p>
            </div>
        `).join('');
    } else {
        findingsSection.style.display = 'none';
    }

    // Display recommendations
    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = results.recommendations.map(rec => `
        <li class="recommendation-item">
            <span class="rec-bullet">&#8227;</span>
            <span class="rec-text">${rec}</span>
        </li>
    `).join('');

    // Display all pathologies
    const pathologiesCount = document.getElementById('pathologies-count');
    pathologiesCount.textContent = `${results.total_pathologies} pathologies screened | Threshold: ${(results.threshold * 100).toFixed(0)}%`;

    const pathologiesGrid = document.getElementById('pathologies-grid');
    pathologiesGrid.innerHTML = results.pathologies.map(path => `
        <div class="pathology-item ${path.is_abnormal ? 'pathology-abnormal' : ''}">
            <div class="pathology-header">
                <span class="pathology-name">${path.name}</span>
                <span class="pathology-score ${path.is_abnormal ? 'score-abnormal' : ''}">${path.percentage}%</span>
            </div>
            <div class="pathology-bar-container">
                <div class="pathology-bar ${path.is_abnormal ? 'severity-' + path.severity : ''}"
                     style="width: ${Math.min(path.percentage, 100)}%"></div>
                <div class="threshold-marker" style="left: ${results.threshold * 100}%"></div>
            </div>
        </div>
    `).join('');

    // Display disclaimer
    document.getElementById('disclaimer-text').textContent = results.disclaimer;

    // Reset to overview tab
    switchTab('overview');
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
    });
    document.getElementById(`tab-${tabName}`).style.display = 'block';
}

/**
 * Clear all state and reset UI
 */
function clearAll() {
    selectedFile = null;
    analysisResults = null;
    currentFilename = null;
    yoloResults = null;
    annotatedImageUrl = null;

    // Reset file input
    fileInput.value = '';

    // Reset dropzone
    dropzone.classList.remove('has-file');
    dropzoneEmpty.style.display = 'flex';
    dropzoneFile.style.display = 'none';

    // Hide preview
    previewCard.style.display = 'none';
    previewImage.src = '';
    document.getElementById('annotated-image').src = '';
    document.getElementById('annotated-image').style.display = 'none';
    document.getElementById('show-annotated-btn').style.display = 'none';
    showOriginalImage();

    // Hide YOLO results
    document.getElementById('yolo-results-card').style.display = 'none';

    // Disable buttons
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    document.getElementById('deep-analysis-btn').disabled = true;
    document.getElementById('download-pdf-btn').disabled = true;

    // Hide results
    hideResults();
    hideError();
}

/**
 * Hide results and show placeholder
 */
function hideResults() {
    resultsPlaceholder.style.display = 'flex';
    resultsContent.style.display = 'none';
    analysisIdSpan.style.display = 'none';
}

/**
 * Show loading overlay
 */
function showLoading() {
    loadingOverlay.style.display = 'flex';
    progressFill.style.width = '30%';
    progressText.textContent = 'Processing...';

    // Simulate progress
    let progress = 30;
    const interval = setInterval(() => {
        if (progress < 90) {
            progress += Math.random() * 10;
            progressFill.style.width = `${progress}%`;
        }
    }, 500);

    loadingOverlay.dataset.interval = interval;
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const interval = loadingOverlay.dataset.interval;
    if (interval) {
        clearInterval(parseInt(interval));
    }
    progressFill.style.width = '100%';
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
    }, 300);
}

/**
 * Show error message
 */
function showError(message) {
    const errorBanner = document.getElementById('error-banner');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorBanner.style.display = 'flex';
}

/**
 * Hide error message
 */
function hideError() {
    document.getElementById('error-banner').style.display = 'none';
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Run GradCAM Heatmap Analysis
 */
async function runDeepAnalysis() {
    if (!currentFilename) {
        showError('Please run the standard analysis first.');
        return;
    }

    const deepAnalysisBtn = document.getElementById('deep-analysis-btn');
    const yoloResultsCard = document.getElementById('yolo-results-card');
    const yoloSummary = document.getElementById('yolo-summary');
    const detectionsList = document.getElementById('detections-list');

    // Show loading state
    deepAnalysisBtn.disabled = true;
    deepAnalysisBtn.innerHTML = '<span class="btn-icon">&#8987;</span> Generating Heatmap...';

    yoloResultsCard.style.display = 'block';
    yoloSummary.innerHTML = `
        <div class="yolo-loading">
            <div class="yolo-loading-spinner"></div>
            <p class="yolo-loading-text">Generating GradCAM heatmap visualization...</p>
        </div>
    `;
    detectionsList.innerHTML = '';

    try {
        const response = await fetch('/api/deep-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: currentFilename })
        });

        const data = await response.json();

        if (data.success) {
            yoloResults = data.results;
            displayYoloResults(data.results);
        } else {
            yoloSummary.innerHTML = `
                <div class="yolo-summary-title">
                    <span>&#9888;</span> Analysis Error
                </div>
                <p class="yolo-summary-text">${data.error}</p>
            `;
        }
    } catch (error) {
        yoloSummary.innerHTML = `
            <div class="yolo-summary-title">
                <span>&#9888;</span> Connection Error
            </div>
            <p class="yolo-summary-text">Unable to connect to heatmap analysis service. ${error.message}</p>
        `;
        console.error('Deep analysis error:', error);
    } finally {
        deepAnalysisBtn.disabled = false;
        deepAnalysisBtn.innerHTML = '<span class="btn-icon">&#128269;</span> Region Heatmap';
    }
}

/**
 * Display GradCAM heatmap analysis results
 */
function displayYoloResults(results) {
    const yoloSummary = document.getElementById('yolo-summary');
    const detectionsList = document.getElementById('detections-list');
    const showAnnotatedBtn = document.getElementById('show-annotated-btn');
    const annotatedImage = document.getElementById('annotated-image');

    // Update summary
    yoloSummary.innerHTML = `
        <div class="yolo-summary-title">
            <span>&#127919;</span> ${results.has_findings ? 'Pathology Regions Highlighted' : 'No Significant Pathologies Above Threshold'}
        </div>
        <p class="yolo-summary-text">${results.summary}</p>
        <div class="yolo-stats">
            <div class="yolo-stat">
                <span class="yolo-stat-value">${results.total_detections}</span>
                <span class="yolo-stat-label">Pathologies</span>
            </div>
            <div class="yolo-stat">
                <span class="yolo-stat-value">${results.high_confidence_count}</span>
                <span class="yolo-stat-label">High Confidence</span>
            </div>
        </div>
    `;

    // Display detections
    if (results.detections && results.detections.length > 0) {
        detectionsList.innerHTML = results.detections.map((det, index) => `
            <div class="detection-card">
                <div class="detection-icon">&#128293;</div>
                <div class="detection-info">
                    <div class="detection-name">${det.class_name}</div>
                    <div class="detection-confidence">Confidence: ${det.confidence}%</div>
                    <div class="detection-bbox">Region highlighted in heatmap</div>
                </div>
            </div>
        `).join('');
    } else {
        detectionsList.innerHTML = `
            <div class="no-detections">
                <div class="no-detections-icon">&#10004;</div>
                <p class="no-detections-text">No significant pathologies detected above threshold.</p>
            </div>
        `;
    }

    // Show annotated image if available
    if (results.annotated_image) {
        annotatedImageUrl = `/api/annotated/${results.annotated_image}`;
        annotatedImage.src = annotatedImageUrl;
        showAnnotatedBtn.style.display = 'inline-block';

        // Automatically show annotated image
        showAnnotatedImage();
    }
}

/**
 * Show original image
 */
function showOriginalImage() {
    const previewImage = document.getElementById('preview-image');
    const annotatedImage = document.getElementById('annotated-image');
    const showOriginalBtn = document.getElementById('show-original-btn');
    const showAnnotatedBtn = document.getElementById('show-annotated-btn');

    previewImage.style.display = 'block';
    annotatedImage.style.display = 'none';
    showOriginalBtn.classList.add('active');
    showAnnotatedBtn.classList.remove('active');
}

/**
 * Show annotated image
 */
function showAnnotatedImage() {
    const previewImage = document.getElementById('preview-image');
    const annotatedImage = document.getElementById('annotated-image');
    const showOriginalBtn = document.getElementById('show-original-btn');
    const showAnnotatedBtn = document.getElementById('show-annotated-btn');

    if (annotatedImageUrl) {
        previewImage.style.display = 'none';
        annotatedImage.style.display = 'block';
        showOriginalBtn.classList.remove('active');
        showAnnotatedBtn.classList.add('active');
    }
}

/**
 * Download PDF Report
 */
async function downloadPdfReport() {
    if (!analysisResults || !currentFilename) {
        showError('Please run the analysis first before downloading the PDF report.');
        return;
    }

    const downloadBtn = document.getElementById('download-pdf-btn');

    // Show loading state
    downloadBtn.disabled = true;
    downloadBtn.innerHTML = '<span class="btn-icon">&#8987;</span> Generating PDF...';

    try {
        // Prepare data for PDF generation
        const pdfData = {
            filename: currentFilename,
            analysis_results: analysisResults,
            heatmap_results: yoloResults || null,
            heatmap_filename: yoloResults ? yoloResults.annotated_image : null
        };

        // Request PDF generation
        const response = await fetch('/api/generate-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(pdfData)
        });

        const data = await response.json();

        if (data.success) {
            // Download the PDF
            const pdfUrl = `/api/reports/${data.pdf_filename}`;

            // Create a temporary link and trigger download
            const link = document.createElement('a');
            link.href = pdfUrl;
            link.download = data.pdf_filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show success message briefly
            showSuccessMessage('PDF report downloaded successfully!');
        } else {
            showError(data.error || 'Failed to generate PDF report.');
        }
    } catch (error) {
        showError('Unable to generate PDF report. Please try again.');
        console.error('PDF generation error:', error);
    } finally {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<span class="btn-icon">&#128196;</span> Download PDF';
    }
}

/**
 * Show success message
 */
function showSuccessMessage(message) {
    // Create a temporary success banner
    const successBanner = document.createElement('div');
    successBanner.className = 'success-banner';
    successBanner.innerHTML = `
        <span class="success-icon">&#10003;</span>
        <span class="success-message">${message}</span>
    `;
    successBanner.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;

    document.body.appendChild(successBanner);

    // Remove after 3 seconds
    setTimeout(() => {
        successBanner.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(successBanner);
        }, 300);
    }, 3000);
}
