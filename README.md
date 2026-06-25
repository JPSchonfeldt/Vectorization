# Vectorization

A self-hosted image to vector conversion service built with FastAPI, Docker, VTracer, Pillow and scikit-learn. It turns raster images (PNG, JPG, WEBP) into clean SVG and PDF vectors through a guided two phase workflow.

---

## Overview

This project is a proof of concept for a self-hosted, subscription ready SaaS image vectorization service. It is designed for raster to SVG and PDF conversion with full user control over each phase of the pipeline, while paving a clear path toward an end user friendly Auto Mode and eventual AI assisted vectorization.

The architecture deliberately separates the cleanup and vectorize stages so that users can iterate on each independently. This makes it easier to understand what each setting does, and ultimately makes a future smart Auto Mode much easier to build.

---

## Features

### Pipeline

- Two phase workflow: Upload, Cleanup, Vectorize, Download
- Per phase preview with toggle between Original, Processed and SVG views
- Multiple presets: Balanced, Simple or Logo, Detailed, Smooth, Vectorizer like
- Live setting summaries showing exactly what was applied

### Cleanup Phase

- Two colour reduction algorithms:
    - Smart Hue spreads colours across the hue spectrum
    - LAB K-means provides perceptual colour grouping
- Two smoothing methods:
    - Gaussian for uniform blur
    - Bilateral for edge preserving smoothing
- Image upscaling from 1x to 4x using LANCZOS, capped at 4000 px
- Sharpen after smoothing
- Palette sidecar saves the chosen palette as JSON for later phases

### Vectorize Phase

- VTracer integration with full slider control
- Auto colour precision uses the smallest precision that fits the cleanup palette
- SVG curve fairing re-fits Bezier curves to remove wobbly joins
- SVG cleanup:
    - Round path coordinates
    - Remove tiny speckle paths
    - Merge adjacent same colour paths
    - Strip metadata

### Output

- Download as SVG or PDF
- Big preview with toggle between Processed image and final SVG

### Future Hooks

- Stage C: rembg background removal (planned)
- Stage D: Auto Mode with image analysis and preset picker (planned)
- AI segmentation for shape aware vectorization (longer term)

---

## Architecture

~~~
image-vector-poc/
|
|-- docker-compose.yml
|
|-- backend/
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- main.py
|   |-- config.py
|   |
|   |-- models/
|   |   |-- settings.py
|   |
|   |-- routes/
|   |   |-- pages.py
|   |   |-- upload.py
|   |   |-- convert.py
|   |   |-- download.py
|   |
|   |-- services/
|   |   |-- preprocessing.py
|   |   |-- color_reduction.py
|   |   |-- vectorizer.py
|   |   |-- svg_smoothing.py
|   |   |-- svg_cleanup.py
|   |
|   |-- templates/
|       |-- upload.html
|       |-- cleanup.html
|       |-- vectorize.html
|
|-- storage/
    |-- uploads/
    |-- processed/
    |-- outputs/
~~~

---

## Tech Stack

| Layer            | Technology                              |
|------------------|------------------------------------------|
| Web Framework    | FastAPI                                  |
| Server           | Uvicorn                                  |
| Templates        | Jinja2                                   |
| Image Processing | Pillow, NumPy, scikit-image              |
| Colour Reduction | scikit-learn (K-means)                   |
| Vectorization    | VTracer                                  |
| SVG Curve Fitting| svgpathtools                             |
| PDF Generation   | svglib, reportlab                        |
| Container        | Docker, Docker Compose                   |
| Frontend         | Vanilla HTML, CSS and JavaScript         |

---

## Getting Started

### Prerequisites

- Windows, macOS or Linux
- Docker Desktop installed and running
- About 2 GB of free disk space for the image

### Clone the Repository

~~~
git clone https://github.com/JPSchonfeldt/Vectorization.git
cd Vectorization
~~~

### Build and Run

~~~
docker compose up --build
~~~

This will:

1. Pull the python:3.12-slim base image
2. Install all Python dependencies (FastAPI, scikit-image, scikit-learn, VTracer and others)
3. Build the container
4. Start the API on http://localhost:8000

First run takes a few minutes for downloads and dependency install. Subsequent runs are fast.

### Use It

Open in your browser:

~~~
http://localhost:8000
~~~

You will see the new drag and drop upload page. From there:

1. Drop or pick an image
2. Click Upload and Continue
3. Tune cleanup sliders, click Update Preview, then click Continue to Vectorize
4. Pick a preset and tune vector sliders, then click Update Preview
5. Choose SVG or PDF and click Download

## API

The frontend is purely a client of the API. All endpoints are also available through Swagger:

~~~
http://localhost:8000/docs
~~~

### Key Endpoints

| Method | Endpoint                                | Purpose                                  |
|--------|-----------------------------------------|------------------------------------------|
| POST   | /upload                                 | Save an uploaded image, returns a file_id|
| POST   | /api/preprocess/{file_id}               | Run cleanup phase only                   |
| POST   | /api/vectorize/{file_id}                | Run vectorize phase only                 |
| POST   | /convert-upload                         | Legacy one shot upload and convert       |
| GET    | /download/svg/{filename}                | Download SVG                             |
| GET    | /download/pdf/{filename}                | Download PDF                             |
| GET    | /download/processed/{filename}          | Download processed PNG                   |
| GET    | /download/upload/{filename}             | Download original upload                 |
| GET    | /health                                 | Health check                             |

---

## Settings Reference

### Cleanup Phase

| Setting                  | Range                  | Purpose                                                |
|--------------------------|------------------------|--------------------------------------------------------|
| Limit Colours            | on or off              | Whether to reduce to a target palette                  |
| Colour Reduction Method  | smart_hue or lab_kmeans| Algorithm for palette selection                        |
| Target Colours           | 2 to 128               | Number of palette colours to keep                      |
| Upscale Before Trace     | 1x to 4x               | Enlarges image for smoother curves                     |
| Smoothing Method         | gaussian or bilateral  | Uniform versus edge preserving smoothing               |
| Smoothing Strength       | 0 to 5                 | How much to smooth                                     |
| Sharpen Strength         | 0 to 5                 | Restore edge clarity after smoothing                   |

### Vectorize Phase

| Setting           | Range     | Purpose                                                          |
|-------------------|-----------|------------------------------------------------------------------|
| Color Precision   | 1 to 8    | Acts as a cap, auto locked to the palette when present           |
| Speckle Filter    | 0 to 30   | Drops paths smaller than this size                               |
| Layer Difference  | 1 to 64   | Controls colour layer separation                                 |
| Corner Threshold  | 0 to 180  | Lower means sharper corners, higher means smoother curves        |
| Length Threshold  | 3.5 to 10 | Lower means more detail, higher means simpler paths              |
| Path Precision    | 1 to 8    | Coordinate precision                                             |
| Max Iterations    | 1 to 50   | More iterations give smoother curves but slower                  |
| Splice Threshold  | 0 to 180  | Where paths are joined                                           |

### Curve Smoothing (Phase 1E)

| Setting               | Range      | Purpose                                                     |
|-----------------------|------------|-------------------------------------------------------------|
| Smooth Jagged Curves  | on or off  | Re-fit Bezier curves after VTracer                          |
| Smoothing Passes      | 0 to 10    | More passes give smoother output but lose sharpness         |
| Sample Step           | 0.5 to 10  | Lower means finer sampling, higher quality but slower       |

### SVG Cleanup (Phase 1C)

| Setting                  | Range     | Purpose                                              |
|--------------------------|-----------|------------------------------------------------------|
| Round Coordinates        | on or off | Reduces SVG file size                                |
| Coordinate Decimals      | 1 to 5    | Lower gives smaller file                             |
| Remove Tiny Paths        | on or off | Drops speckle paths                                  |
| Min Path Size            | 0 to 50   | Drops paths below this square area                   |
| Merge Same Colour Paths  | on or off | Merges adjacent paths of the same colour             |

---

## Presets

Each preset is a starting point, not a final answer. Tweak from there.

| Preset           | Best For                                        |
|------------------|-------------------------------------------------|
| Balanced         | General purpose default                         |
| Simple or Logo   | Logos with few flat colours                     |
| Detailed         | Complex illustrations and cartoons              |
| Smooth           | When curve quality matters more than detail     |
| Vectorizer like  | Trying to imitate Vectorizer AI style output    |

---

## Roadmap

### Phase 1 - Classical quality improvements (done)

- Phase 1A: LAB K-means colour reduction
- Phase 1B: Bilateral edge preserving smoothing
- Phase 1C: SVG path cleanup
- Phase 1D: Palette locked vectorization and auto colour precision
- Phase 1E: SVG curve fairing

### Phase 2 - Planned

- Phase 1F: Edge aware classical cleanup (median and morphological)
- Stage C: rembg background removal as a dedicated phase
- Stage D: Auto Mode with image analysis and preset picker

### Phase 3 - Future

- AI segmentation (SAM, BiRefNet, U2-Net) for shape aware vectorization
- Real-ESRGAN AI upscaling as a Pro tier
- User accounts and monthly subscription billing
- Public API access for business tier
- Batch upload and processing

---

## Limitations

- This is a proof of concept, not a production ready system.
- No user accounts, authentication or billing yet.
- Classical vectorization cannot fully replicate shape aware AI tools.
- Output quality on photographic or noisy images is limited.
- All processing is synchronous and there is no background worker queue yet.
- No request rate limiting.

---

## Status

This project is in active development. The backbone is stable and the architecture is intentionally modular so each new capability can be added cleanly.

---

## License

This project is currently unlicensed and is not intended for redistribution while still in POC phase. A license will be added before any public or business use.

---

## Acknowledgements

- VTracer at https://github.com/visioncortex/vtracer for the heart of the vectorization step
- scikit-image at https://scikit-image.org for image processing primitives
- scikit-learn at https://scikit-learn.org for K-means clustering
- svgpathtools at https://github.com/mathandy/svgpathtools for SVG path parsing
- FastAPI at https://fastapi.tiangolo.com for the modern Python web framework
- Pillow at https://pillow.readthedocs.io for image manipulation
