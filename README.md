# AI Garment Multi-Label Size Inspection System

> **Platform**: Windows PC · NVIDIA GPU (RTX 2060+) · Python 3.10+  
> **UI**: PyQt6 desktop (line workers) + FastAPI web dashboard (supervisors)

---

## Overview

This system uses computer vision and OCR to automatically verify that **all garment labels** (size label, care label, hang tag) are present **and consistent** — catching mismatched sizes or missing labels before garments leave the production line.

A camera captures each garment; the pipeline detects every label region, reads the size text, and raises an audible/visual alarm within **200 ms** when anything is wrong.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INSPECTION PIPELINE                          │
│                                                                     │
│  Camera        YOLO           Feature         PaddleOCR   Judge    │
│  Capture  ─►  Detector  ─►   Extractor   ─►  (OCR)   ─►  Pass /  │
│  (OpenCV)     (labels)       + FAISS KB       (size)      Fail     │
│                                                             │       │
│                                             ┌───────────────┘       │
│                                             ▼                       │
│                                         AlarmController             │
│                                        (winsound / serial)          │
└─────────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
  PyQt6 Desktop UI                        FastAPI Web Dashboard
  (line workers — full-screen)            (supervisors — browser)
         │                                        │
         └───────────────┬────────────────────────┘
                         ▼
                   SQLite + FAISS
                 (knowledge base)
```

---

## Quick Start

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA Toolkit | 11.8 or 12.x |
| NVIDIA GPU | RTX 2060 or better (8 GB VRAM recommended) |
| OS | Windows 10/11 64-bit |

### 2. Install dependencies

```powershell
# (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> [!NOTE]
> `paddlepaddle-gpu` requires the matching CUDA version.  
> See [PaddlePaddle install guide](https://www.paddlepaddle.org.cn/install/quick) if the pip wheel fails.

### 3. Verify environment

```powershell
python scripts/setup_check.py
```

All rows should show **PASS**. Fix any **FAIL** rows before continuing.

### 4. Run the desktop UI (line worker)

```powershell
python -m src.ui.main_window
```

### 5. Run the FastAPI supervisor dashboard

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs
```

### 6. Test the pipeline on a single image

```powershell
python -m src.core.pipeline path\to\garment.jpg
```

---

## Project Structure

```
ai-ocr-label/
├── config.yaml                    # Main configuration
├── requirements.txt
├── README.md
│
├── src/
│   ├── core/
│   │   ├── models.py              # Shared dataclasses & enums
│   │   ├── judge.py               # ConsistencyJudge
│   │   └── pipeline.py            # InspectionPipeline orchestrator
│   ├── teaching/                  # Interactive label teaching module
│   ├── alignment/                 # Garment alignment / perspective correction
│   ├── api/
│   │   ├── main.py                # FastAPI app entry point
│   │   └── routes/                # API route handlers
│   └── ui/                        # PyQt6 desktop application
│
├── hardware/
│   ├── alarm_controller.py        # winsound / serial alarm driver
│   └── config/                    # Hardware-specific config files
│
├── models/
│   ├── label_detector/weights/    # YOLO .pt weights
│   ├── feature_extractor/         # MobileNetV3 ONNX embedder
│   └── ocr/{det,rec}/             # PaddleOCR det/rec model files
│
├── data/
│   ├── knowledge_base.db          # SQLite — template metadata
│   ├── faiss_index.bin            # FAISS vector index
│   ├── reference_images/          # Golden-sample reference crops
│   └── snapshots/                 # Auto-saved inspection frames
│
├── training/
│   ├── label_detector/dataset/    # YOLO training images + labels
│   └── fine_tune_embedding/       # Embedding fine-tune scripts
│
├── tests/
│   └── benchmarks/                # Latency / accuracy benchmarks
│
└── scripts/
    └── setup_check.py             # Environment verification
```

---

## Configuration

All runtime settings live in [`config.yaml`](config.yaml).  Key sections:

| Section | Purpose |
|---|---|
| `camera` | Resolution, FPS, trigger mode, device index |
| `models.label_detector` | YOLO weights path, confidence / IOU thresholds |
| `models.feature_extractor` | ONNX embedder path, similarity threshold |
| `models.ocr` | Engine choice (`paddleocr`/`easyocr`), GPU flag |
| `inspection` | Required label types, latency budget (ms) |
| `alarm` | Enable/disable, beep durations, serial port |
| `database` | SQLite + FAISS paths |
| `logging` | Log level, snapshot saving, retention days |

---

## Teaching Mode

When a new label design appears on the production line the system cannot yet recognise it, it flags `needs_teaching = True` and queues the unknown crop.

A supervisor opens the **Teaching UI**:

1. The unknown crop is shown alongside existing templates.
2. The supervisor either **assigns it to an existing template** or **creates a new template**.
3. The new embedding is inserted into the FAISS index; the SQLite record is updated.
4. The pipeline immediately recognises the new design without retraining.

```powershell
# Launch the teaching UI standalone (supervisor workstation)
python -m src.teaching.teaching_ui
```

---

## Alarm Patterns

| Event | Frequency | Duration | Pattern |
|---|---|---|---|
| PASS | 1000 Hz | 100 ms | 1 short beep |
| SIZE_MISMATCH | 500 Hz | 300 ms × 3 | 3 rapid beeps |
| MISSING_LABEL | 400 Hz | 1000 ms × 2 | 2 long beeps |
| SIZE_NOT_FOUND | 600 Hz | 150+150+800 ms | 2 short + 1 long |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA RTX 2060 (6 GB) | RTX 3070 / 4070 (8 GB) |
| CPU | Intel i5-10th gen | Intel i7-12th gen or Ryzen 7 |
| RAM | 16 GB | 32 GB |
| Camera | USB 3.0, 1080p @ 30 fps | GigE industrial, 1080p @ 60 fps |
| Storage | 256 GB SSD | 1 TB NVMe SSD |
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CUDA | 11.8 | 12.x |

---


| File | Contents |
| --- | --- |
| `config.yaml` | Full YAML: camera, models, inspection, alarm, database, logging |
| `requirements.txt` | 22 packages including `colorama` |
| `src/core/``models.py` | `LabelType`, `ErrorCode`, `BBox`, `LabelDetection`, `JudgeResult`, `InspectionResult` |
| `src/core/``judge.py` | `ConsistencyJudge` — missing labels, unreadable sizes, mismatch detection |
| `src/core/``pipeline.py` | `InspectionPipeline` with `warm_up()`, `run_sync()`, stub sub-components, CLI `__main__` |
| `hardware/``alarm_controller.py` | `AlarmController` — winsound patterns, threading, colorama fallback, context manager |
| `scripts/``setup_check.py` | Checks Python 3.10, CUDA/GPU, all packages, config.yaml, dirs — ASCII table output |
| `README.md` | ASCII architecture diagram, quick-start, structure, teaching mode, hardware table |
| 9× `__init__.py` | All `src/`, `src/core/`, `src/teaching/`, `src/alignment/`, `src/api/`, `src/api/routes/`, `src/ui/`, `hardware/`, `tests/` |

## License

Internal use only — © 2024 Your Organisation.
