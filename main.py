from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple
import concurrent.futures
import multiprocessing

app = FastAPI()

def detect_all_black(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    max_brightness = np.max(gray)
    if mean_brightness < 10 and max_brightness < 20:
        return True, "Image is almost all black"
    return False, ""

def detect_vertical_line_artifacts(img: np.ndarray) -> Tuple[bool, str]:
    # Detect vertical line artifacts by looking at column differences
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute vertical gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_abs = np.abs(sobel_x)
    # Compute column means
    col_means = np.mean(sobel_abs, axis=0)
    # Look for sudden spikes in column means
    threshold = np.percentile(col_means, 99.5)
    high_cols = col_means > threshold
    # Count how many high columns we have, and if they're clustered
    high_col_count = np.sum(high_cols)
    if high_col_count > len(col_means) * 0.1:
        return True, "Potential vertical line artifacts detected"
    return False, ""

def detect_horizontal_line_artifacts(img: np.ndarray) -> Tuple[bool, str]:
    # Detect horizontal line artifacts by looking at row differences
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute horizontal gradients
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = np.abs(sobel_y)
    # Compute row means
    row_means = np.mean(sobel_abs, axis=1)
    # Look for sudden spikes in row means
    threshold = np.percentile(row_means, 99.5)
    high_rows = row_means > threshold
    # Count how many high rows we have
    high_row_count = np.sum(high_rows)
    if high_row_count > len(row_means) * 0.1:
        return True, "Potential horizontal line artifacts detected"
    return False, ""

def detect_exposure_issues(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    issues = []
    if mean_brightness < 20:
        issues.append("Underexposed image")
    elif mean_brightness > 235:
        issues.append("Overexposed image")
    if std_brightness < 10:
        issues.append("Low contrast image")
    if issues:
        return True, ", ".join(issues)
    return False, ""

def detect_motion_blur(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    if blur_score < 50:
        return True, "Potential motion blur detected"
    return False, ""

def detect_stitching_seams(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Look for sharp vertical and horizontal edges that might be seams
    edges = cv2.Canny(gray, 50, 150)
    # Check for long vertical lines
    vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=gray.shape[0]*0.7, maxLineGap=10)
    # Check for long horizontal lines
    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=gray.shape[1]*0.7, maxLineGap=10)
    
    if vertical_lines is not None and len(vertical_lines) > 5:
        return True, "Potential stitching seam issues (vertical lines) detected"
    if horizontal_lines is not None and len(horizontal_lines) > 5:
        return True, "Potential stitching seam issues (horizontal lines) detected"
    return False, ""

def detect_color_banding(img: np.ndarray) -> Tuple[bool, str]:
    # Check for color banding by looking at histogram smoothness
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        for channel in [b, g, r]:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            # Count number of empty bins
            empty_bins = np.sum(hist == 0)
            if empty_bins > 150:
                return True, "Potential color banding detected"
    return False, ""

def detect_large_black_regions(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to find black pixels
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Check each component except the first one (background)
    total_black_pixels = 0
    total_pixels = img.shape[0] * img.shape[1]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        total_black_pixels += area
    # If more than 5% of the image is black, flag it
    if (total_black_pixels / total_pixels) > 0.05:
        return True, "Large black regions detected (incomplete stitching)"
    return False, ""

def detect_glitching_tearing(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_abs = np.abs(edges)
    row_means = np.mean(edges_abs, axis=1)
    threshold = np.percentile(row_means, 99.5)
    high_edges = row_means > threshold
    transitions = np.diff(high_edges.astype(int))
    num_transitions = np.sum(np.abs(transitions))
    if num_transitions > 20:
        return True, "Potential glitching/tearing detected"
    # Also check for abnormal color shifts between rows (very high threshold)
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        row_diff_b = np.abs(np.diff(b, axis=0))
        row_diff_g = np.abs(np.diff(g, axis=0))
        row_diff_r = np.abs(np.diff(r, axis=0))
        total_diff = np.mean(row_diff_b + row_diff_g + row_diff_r)
        if total_diff > 150:
            return True, "Potential color glitching/tearing detected"
    return False, ""

def detect_noise_lines(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    region_y = magnitude_spectrum[center_y - 5 : center_y + 5, :]
    region_x = magnitude_spectrum[:, center_x - 5 : center_x + 5]
    max_y = np.max(region_y)
    max_x = np.max(region_x)
    threshold = np.mean(magnitude_spectrum) * 5
    if max_y > threshold or max_x > threshold:
        return True, "Potential digital noise lines detected"
    return False, ""

def detect_camera_issues(img: np.ndarray) -> Tuple[bool, str]:
    issues = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    if blur_score < 30:
        issues.append("Low image clarity")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    if s_mean < 20 or s_mean > 230:
        issues.append("Unusual color saturation")
    if len(issues) > 0:
        return True, ", ".join(issues)
    return False, ""

def create_thumbnail(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1600, 1200))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def scan_single_image(args: Tuple[bytes, str]) -> Dict:
    image_bytes, filename = args
    # Load image
    img = np.array(Image.open(io.BytesIO(image_bytes)))
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Downscale for detection to speed things up
    height, width = img_bgr.shape[:2]
    scale_factor = min(1.0, 1000 / max(width, height))
    if scale_factor < 1.0:
        new_w = int(width * scale_factor)
        new_h = int(height * scale_factor)
        img_small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr
    
    issues = []
    all_black, all_black_msg = detect_all_black(img_small)
    if all_black:
        issues.append(all_black_msg)
    glitch_detected, glitch_msg = detect_glitching_tearing(img_small)
    if glitch_detected:
        issues.append(glitch_msg)
    
    thumbnail = create_thumbnail(image_bytes)
    return {
        "filename": filename,
        "status": "Flagged" if issues else "Passed",
        "issues": "; ".join(issues) if issues else "No issues detected",
        "thumbnail": thumbnail
    }

@app.get("/", response_class=HTMLResponse)
async def get_home():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panorama Scanner QA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
        body { background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%); color: #334155; padding: 3rem 1rem; min-height: 100vh; }
        .container { max-width: 1600px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 3rem; }
        h1 { font-size: 2.75rem; font-weight: 800; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.75rem; letter-spacing: -0.02em; }
        .subtitle { font-size: 1.15rem; color: #64748b; font-weight: 400; max-width: 600px; margin: 0 auto; }
        .card { background: white; border-radius: 1rem; padding: 2.5rem; box-shadow: 0 10px 40px rgba(15, 23, 42, 0.08); border: 1px solid #f1f5f9; margin-bottom: 2rem; }
        .upload-section { display: flex; flex-direction: column; gap: 1.5rem; }
        .file-input-wrapper { width: 100%; }
        input[type="file"] { width: 100%; padding: 2rem; border: 2px dashed #cbd5e1; border-radius: 0.75rem; background: #f8fafc; cursor: pointer; transition: all 0.3s ease; }
        input[type="file"]:hover { border-color: #3b82f6; background: #eff6ff; transform: translateY(-2px); }
        .btn-primary { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; border: none; padding: 1rem 2.5rem; border-radius: 0.75rem; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25); }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35); }
        .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; transform: none; }
        .btn-secondary { background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-size: 0.9rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease; }
        .btn-secondary:hover { background: #e2e8f0; }
        .results-section { display: none; }
        .results-header { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 2px solid #f1f5f9; }
        .results-header h2 { font-size: 1.75rem; color: #1e293b; font-weight: 700; }
        .filter-section { display: flex; gap: 1rem; align-items: center; }
        .filter-section label { font-weight: 500; color: #475569; }
        select { padding: 0.625rem 1rem; border: 1px solid #e2e8f0; border-radius: 0.5rem; background: white; font-size: 0.95rem; cursor: pointer; }
        .stats { display: flex; gap: 1.5rem; }
        .stat { text-align: center; padding: 0.75rem 1.25rem; background: #f8fafc; border-radius: 0.5rem; border: 1px solid #e2e8f0; }
        .stat-value { font-size: 1.5rem; font-weight: 700; display: block; }
        .stat-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
        table { width: 100%; border-collapse: separate; border-spacing: 0; }
        thead th { padding: 1rem 1.25rem; text-align: left; border-bottom: 2px solid #e2e8f0; background: #f8fafc; font-weight: 700; color: #475569; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; position: sticky; top: 0; cursor: pointer; user-select: none; }
        thead th:hover { background: #e2e8f0; }
        thead th.active { background: #e0ecff; color: #1e3a8a; }
        .sort-icon { margin-left: 0.5rem; opacity: 0.5; }
        .sort-icon.asc::after { content: "↑"; }
        .sort-icon.desc::after { content: "↓"; }
        tbody tr { transition: all 0.2s ease; }
        tbody tr:hover { background: #f8fafc; }
        tbody td { padding: 1.25rem; border-bottom: 1px solid #f1f5f9; font-size: 0.95rem; }
        .status-passed { color: #059669; font-weight: 700; display: inline-flex; align-items: center; gap: 0.5rem; }
        .status-passed::before { content: "✓"; background: #d1fae5; padding: 0.125rem 0.4rem; border-radius: 50%; font-size: 0.8rem; }
        .status-flagged { color: #dc2626; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 0.5rem; }
        .status-flagged::before { content: "!"; background: #fee2e2; padding: 0.125rem 0.45rem; border-radius: 50%; font-size: 0.8rem; }
        .status-flagged:hover { text-decoration: underline; }
        .view-btn { background: #1e293b; color: white; border: none; padding: 0.625rem 1.25rem; border-radius: 0.5rem; font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease; }
        .view-btn:hover { background: #0f172a; transform: translateY(-1px); }
        .export-section { margin-top: 2rem; padding-top: 2rem; border-top: 2px solid #f1f5f9; display: flex; justify-content: flex-start; }
        .loading { display: none; text-align: center; padding: 3rem; }
        .spinner { width: 50px; height: 50px; border: 4px solid #e2e8f0; border-top-color: #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(15, 23, 42, 0.9); backdrop-filter: blur(5px); overflow: auto; }
        .modal-content { position: relative; background-color: white; margin: 2rem auto; padding: 2.5rem; border-radius: 1rem; max-width: 95%; max-height: 90vh; overflow: auto; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        .close-btn { color: #94a3b8; float: right; font-size: 32px; font-weight: 700; cursor: pointer; line-height: 1; transition: all 0.2s ease; }
        .close-btn:hover { color: #1e293b; transform: scale(1.1); }
        .modal-image { max-width: 100%; height: auto; display: block; margin: 1.5rem auto; border-radius: 0.75rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .modal-title { font-size: 1.35rem; margin-bottom: 0.75rem; color: #0f172a; font-weight: 700; }
        .modal-issues { color: #475569; font-size: 1rem; padding: 1rem; background: #f8fafc; border-radius: 0.5rem; border-left: 4px solid #3b82f6; }
        .modal-nav { display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem; }
        .nav-btn { background: #3b82f6; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-size: 1rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease; }
        .nav-btn:hover { background: #2563eb; transform: translateY(-1px); }
        .nav-btn:disabled { background: #94a3b8; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Panorama Scanner QA</h1>
            <p class="subtitle">Automatically check stitched 360° panorama images for defects</p>
        </header>
        
        <div class="card upload-section">
            <div class="file-input-wrapper">
                <input type="file" id="imageInput" multiple accept="image/*">
            </div>
            <button id="scanBtn" class="btn-primary">Scan Images</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Scanning images...</p>
        </div>

        <div class="card results-section" id="results">
            <div class="results-header">
                <h2>Scan Results</h2>
                <div class="filter-section">
                    <label for="statusFilter">Filter by Status:</label>
                    <select id="statusFilter">
                        <option value="all">All</option>
                        <option value="Passed">Passed</option>
                        <option value="Flagged">Flagged</option>
                    </select>
                </div>
                <div class="stats" id="stats"></div>
            </div>
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th data-sort="filename">File Name<span class="sort-icon"></span></th>
                        <th data-sort="status">Status<span class="sort-icon"></span></th>
                        <th data-sort="issues">Detected Issue<span class="sort-icon"></span></th>
                        <th>Preview</th>
                    </tr>
                </thead>
                <tbody id="resultsBody"></tbody>
            </table>
            <div class="export-section" id="exportSection">
                <button id="exportBtn" class="btn-primary">📋 Export Passed File Names</button>
            </div>
        </div>
    </div>

    <div id="imageModal" class="modal">
        <div class="modal-content">
            <span class="close-btn">&times;</span>
            <h2 id="modalTitle" class="modal-title"></h2>
            <p id="modalIssues" class="modal-issues"></p>
            <img id="modalImage" class="modal-image" src="" alt="Preview">
            <div class="modal-nav">
                <button id="prevBtn" class="nav-btn">← Previous</button>
                <button id="nextBtn" class="nav-btn">Next →</button>
            </div>
        </div>
    </div>

    <script>
        let allResults = [];
        let filteredResults = [];
        let currentModalIndex = 0;
        let sortField = null;
        let sortDirection = 'asc';
        
        document.getElementById('scanBtn').addEventListener('click', async () => {
            const fileInput = document.getElementById('imageInput');
            if (fileInput.files.length === 0) {
                alert('Please select at least one image!');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('scanBtn').disabled = true;
            
            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files', fileInput.files[i]);
            }
            
            try {
                const response = await fetch('/scan', { method: 'POST', body: formData });
                allResults = await response.json();
                filteredResults = [...allResults];
                sortField = null;
                sortDirection = 'asc';
                clearSortIcons();
                displayResults(filteredResults);
            } catch (error) {
                alert('Error scanning images: ' + error);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('scanBtn').disabled = false;
            }
        });
        
        document.getElementById('statusFilter').addEventListener('change', (e) => {
            const filter = e.target.value;
            if (filter === 'all') {
                filteredResults = [...allResults];
            } else {
                filteredResults = allResults.filter(r => r.status === filter);
            }
            if (sortField) {
                sortResults(filteredResults);
            }
            displayResults(filteredResults);
        });
        
        document.querySelectorAll('th[data-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const field = th.dataset.sort;
                if (sortField === field) {
                    sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    sortField = field;
                    sortDirection = 'asc';
                }
                clearSortIcons();
                th.classList.add('active');
                const icon = th.querySelector('.sort-icon');
                icon.classList.add(sortDirection);
                sortResults(filteredResults);
                displayResults(filteredResults);
            });
        });
        
        function clearSortIcons() {
            document.querySelectorAll('th').forEach(th => {
                th.classList.remove('active');
                const icon = th.querySelector('.sort-icon');
                if (icon) {
                    icon.classList.remove('asc', 'desc');
                }
            });
        }
        
        function sortResults(results) {
            results.sort((a, b) => {
                let valueA = a[sortField];
                let valueB = b[sortField];
                
                if (typeof valueA === 'string') {
                    valueA = valueA.toLowerCase();
                    valueB = valueB.toLowerCase();
                }
                
                if (valueA < valueB) {
                    return sortDirection === 'asc' ? -1 : 1;
                }
                if (valueA > valueB) {
                    return sortDirection === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        
        function displayResults(results) {
            // Calculate stats
            const total = results.length;
            const passed = results.filter(r => r.status === 'Passed').length;
            const flagged = results.filter(r => r.status === 'Flagged').length;
            
            // Update stats display
            document.getElementById('stats').innerHTML = `
                <div class="stat">
                    <span class="stat-value">${total}</span>
                    <span class="stat-label">Total</span>
                </div>
                <div class="stat">
                    <span class="stat-value" style="color: #059669;">${passed}</span>
                    <span class="stat-label">Passed</span>
                </div>
                <div class="stat">
                    <span class="stat-value" style="color: #dc2626;">${flagged}</span>
                    <span class="stat-label">Flagged</span>
                </div>
            `;
            
            const tbody = document.getElementById('resultsBody');
            tbody.innerHTML = '';
            results.forEach((result, index) => {
                const originalIndex = allResults.findIndex(r => r.filename === result.filename);
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${result.filename}</td>
                    <td class="${result.status === 'Passed' ? 'status-passed' : 'status-flagged'}" data-index="${originalIndex}">${result.status}</td>
                    <td>${result.issues}</td>
                    <td><button class="view-btn" data-index="${originalIndex}">View</button></td>
                `;
                tbody.appendChild(row);
            });
            document.querySelectorAll('.view-btn, .status-flagged').forEach(el => {
                el.addEventListener('click', (e) => {
                    const index = parseInt(e.target.dataset.index);
                    openModal(index);
                });
            });
            document.getElementById('results').style.display = 'block';
            document.getElementById('exportSection').style.display = 'block';
        }
        
        function openModal(index) {
            currentModalIndex = index;
            updateModal();
            document.getElementById('imageModal').style.display = 'block';
        }
        
        function updateModal() {
            const result = allResults[currentModalIndex];
            document.getElementById('modalTitle').textContent = result.filename;
            document.getElementById('modalIssues').textContent = result.issues;
            document.getElementById('modalImage').src = result.thumbnail;
            
            // Update nav button states
            document.getElementById('prevBtn').disabled = currentModalIndex === 0;
            document.getElementById('nextBtn').disabled = currentModalIndex === allResults.length - 1;
        }
        
        document.getElementById('prevBtn').addEventListener('click', () => {
            if (currentModalIndex > 0) {
                currentModalIndex--;
                updateModal();
            }
        });
        
        document.getElementById('nextBtn').addEventListener('click', () => {
            if (currentModalIndex < allResults.length - 1) {
                currentModalIndex++;
                updateModal();
            }
        });
        
        document.querySelector('.close-btn').addEventListener('click', () => {
            document.getElementById('imageModal').style.display = 'none';
        });
        
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('imageModal');
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        document.getElementById('exportBtn').addEventListener('click', () => {
            const passedFiles = allResults.filter(r => r.status === 'Passed').map(r => r.filename);
            if (passedFiles.length === 0) {
                alert('No passed files to export!');
                return;
            }
            const blob = new Blob([passedFiles.join('\\n')], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'passed_panoramas.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/scan")
async def scan_files(files: List[UploadFile] = File(...)):
    # Read all files first
    file_data = []
    for file in files:
        contents = await file.read()
        file_data.append((contents, file.filename))
    
    # Use parallel processing to scan
    num_workers = min(multiprocessing.cpu_count(), len(file_data))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(scan_single_image, file_data):
            results.append(result)
    
    # Sort results by original filename order
    filename_order = [f[1] for f in file_data]
    results.sort(key=lambda x: filename_order.index(x["filename"]))
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)

