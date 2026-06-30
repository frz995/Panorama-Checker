# Panorama Scanner QA

A simple, web-based tool to automatically check stitched 360° panorama images for defects.

## Features

- Upload multiple panorama images at once
- Automatically detect common defects:
  - All-black images
  - Image glitching/tearing
- Clean, modern dashboard displaying scan results
- One-click export of passed file names
- Fast parallel processing
- Image preview functionality

## Tech Stack

- **Backend**: Python + FastAPI
- **Frontend**: HTML/CSS/JavaScript
- **Image Processing**: OpenCV

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd qaqc-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Open your browser and go to http://localhost:8500

3. Upload your panorama images and click "Scan Images"

4. View the results and export the passed file names

## Requirements

- Python 3.7+
- All dependencies listed in requirements.txt

## License

MIT License
