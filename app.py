import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Tuple

st.set_page_config(page_title="Panorama Scanner QA", page_icon="📸", layout="wide")

def detect_glitching_tearing(img: np.ndarray) -> Tuple[bool, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_abs = np.abs(edges)
    row_means = np.mean(edges_abs, axis=1)
    threshold = np.percentile(row_means, 99)
    high_edges = row_means > threshold
    transitions = np.diff(high_edges.astype(int))
    num_transitions = np.sum(np.abs(transitions))
    if num_transitions > 10:
        return True, "Potential glitching/tearing detected"
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
    threshold = np.mean(magnitude_spectrum) * 3
    if max_y > threshold or max_x > threshold:
        return True, "Potential digital noise lines detected"
    return False, ""

def detect_camera_issues(img: np.ndarray) -> Tuple[bool, str]:
    issues = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = np.var(laplacian)
    if blur_score < 50:
        issues.append("Low image clarity")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    if s_mean < 30 or s_mean > 200:
        issues.append("Unusual color saturation")
    if len(issues) > 0:
        return True, ", ".join(issues)
    return False, ""

def scan_image(image_bytes: bytes) -> Dict:
    img = np.array(Image.open(io.BytesIO(image_bytes)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    issues = []
    glitch_detected, glitch_msg = detect_glitching_tearing(img)
    if glitch_detected:
        issues.append(glitch_msg)
    noise_detected, noise_msg = detect_noise_lines(img)
    if noise_detected:
        issues.append(noise_msg)
    camera_detected, camera_msg = detect_camera_issues(img)
    if camera_detected:
        issues.append(camera_msg)
    return {
        "status": "Flagged" if issues else "Passed",
        "issues": "; ".join(issues) if issues else "No issues detected"
    }

st.title("📸 Panorama Scanner QA")
st.markdown("Automatically scan stitched 360° panorama images for defects")

uploaded_files = st.file_uploader("Upload panorama images", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Scanning images..."):
        results = []
        for file in uploaded_files:
            result = scan_image(file.getvalue())
            results.append({
                "filename": file.name,
                "status": result["status"],
                "issues": result["issues"]
            })
    st.subheader("Scan Results Dashboard")
    col1, col2, col3 = st.columns([3, 2, 4])
    with col1:
        st.write("**File Name**")
    with col2:
        st.write("**Status**")
    with col3:
        st.write("**Detected Issue**")
    st.divider()
    passed_files = []
    for res in results:
        c1, c2, c3 = st.columns([3, 2, 4])
        with c1:
            st.write(res["filename"])
        with c2:
            if res["status"] == "Passed":
                st.success(res["status"])
                passed_files.append(res["filename"])
            else:
                st.error(res["status"])
        with c3:
            st.write(res["issues"])
    st.divider()
    if passed_files:
        st.subheader("Export Passed Files")
        passed_text = "\n".join(passed_files)
        st.download_button(
            label="📋 Copy/Export Passed File Names",
            data=passed_text,
            file_name="passed_panoramas.txt",
            mime="text/plain"
        )
