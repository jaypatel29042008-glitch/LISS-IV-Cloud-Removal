# ISRO Hackathon 2025: LISS-IV Cloud Removal & Reconstruction System

This repository contains the complete, production-ready frontend website for the ISRO Hackathon project submission: **"Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery"**.

This website is engineered as a single-page interactive application that presents the problem statement, technical solution, pipeline architecture, model evaluation benchmarks, and downstream real-world applications (like crop monitoring, disaster management, and the NHAI Green Cover Index).

## 🚀 How to Run the Website

The website is fully self-contained and has **zero external dependencies**. It works completely offline and requires no build steps or local servers.

1. Locate the `isro_hackathon.html` file inside this folder.
2. Double-click `isro_hackathon.html` to open it in any modern web browser (Google Chrome, Microsoft Edge, Firefox, Safari).
3. (Optional) To host it locally, run `python -m http.server 8000` in this directory and navigate to `http://localhost:8000`.

---

## 🛰️ Technical Concept Summary

### The Problem
* **LISS-IV Sensor Constraints:** Resourcesat-2A's LISS-IV sensor captures 3 VNIR bands (Green, Red, NIR) at 5.8m resolution, but has **no Blue or SWIR bands**.
* **Cloud Cover Obstruction:** 67% of the Earth's surface is cloud-covered at any moment, rendering large portions of satellite data unusable.
* **Limitations of Existing Methods:** Traditional GANs (like Pix2Pix) suffer from mode collapse and hallucinate features, while standard diffusion models (like DDPM) are too slow (1000+ steps) and suffer from color/spectral shifts.

### The Solution: DB-CR (Diffusion Bridge for Cloud Removal)
* **Diffusion Bridge Theory:** Instead of diffusing from pure Gaussian noise, the model creates a direct probability path (Schrödinger Bridge) from the cloudy image to the clean image, preserving structural integrity.
* **Multimodal Fusion:** Integrates Sentinel-1 SAR (radar) data to "see" through clouds and guide the optical reconstruction.
* **Spectral Synthesis:** Generates a synthetic Blue band (for true-color display and HOT haze correction) and SWIR band (to segment clouds from snow/ice over the Himalayas) using cross-sensor polynomial regression calibrated against Sentinel-2.

---

## 🎨 Website Structure & Interactive Features

The site consists of 10 highly interactive sections built with premium visuals and animations:

1. **Space Canvas Background:** A performance-optimized HTML5 Canvas particle system rendering a starfield, drifting constellation nodes, passing satellites, and cloud-removal scanning rings.
2. **Hero Section:** Animated header typing effect, stat pills, and a 3D-effect CSS/SVG Earth globe showing a dynamic AI cloud-scanning and clearing cycle.
3. **Problem Statement:** 3 interactive constraint cards detail LISS-IV limitations, accompanied by an animated 67% coverage bar and a scrolling specs marquee.
4. **8-Stage Pipeline:** An interactive flowchart where clicking any stage dynamically updates a detail panel with input/output specs, technical details, and the underlying mathematical formulas (e.g., TOA calibration, Blue band synthesis, Diffusion Bridge transition).
5. **Model Comparison:** Evaluates 5 architectures (Pix2Pix, DiffCR, DB-CR, DC4CR, SRP-CR) with animated ratings. Includes a toggle switch to dynamically animate early-stage vs. late-stage training loss weights.
6. **Before/After Live Slider:** A drag-to-reveal comparison slider demonstrating cloud reconstruction across 4 geographic regions (Punjab crop belts, Assam flood zones, Delhi NCR urban areas, and Himalayan foothills), using responsive HTML5 canvases.
7. **Metrics Dashboard:** Scroll-triggered counting numbers showing SOTA results (PSNR: **35.2 dB**, SSIM: **0.963**, SAM: **0.042 rad**, GCI Correlation: **90%**), supported by a grouped CSS chart.
8. **Downstream Applications:** Focuses on the NHAI Green Cover Index (Green canopy monitoring) and disaster response. Includes an interactive SVG India map with pulsing geographic pins.
9. **Tech Stack & Datasets:** Clean badges for frameworks used (PyTorch, Rasterio, GDAL) and cards detailing datasets (SEN12MS-CR, RICE, Bhuvan).
10. **Team & Footer:** Clean team cards and a professional footer with quick-links and ISRO hackathon credits.

---

## 🛠️ Technology Stack

* **Structure:** Semantic HTML5
* **Styling:** Custom Vanilla CSS3 (CSS Grid, Flexbox, Keyframes, Custom Variables)
* **Logic & Animations:** Pure ES6+ JavaScript, HTML5 Canvas API, SVG manipulation
* **Icons:** Embedded, light-weight inline SVGs
