# PLF Pipeline — Video to PLF Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Pre-alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)]()

Automatic generation of `.raw.plf` files from video footage. Orchestrates a
pipeline of AI models that analyze video and audio to produce structured PLF.

## Pipeline stages

```
VIDEO FILE
    │
    ├── Video Track
    │   ├── scene_detector  → PySceneDetect → shot boundaries (i:/o:)
    │   ├── frame_sampler   → Extract representative frames per shot
    │   └── vision          → Qwen2-VL 7B → shot type, camera, mood, elements
    │
    └── Audio Track
        ├── transcriber      → Whisper large-v3 → word-level transcript
        ├── diarizer         → pyannote-audio 3.x → speaker identification
        └── audio_classifier → PANNs → ambient/music classification
            │
            ▼
        assembler → merge all outputs into .raw.plf
```

## Installation

```bash
pip install -e .
```

## Quick start

```python
from pipeline import run_pipeline, UserContext

context = UserContext(
    description="A short film about memory and loss at the beach.",
    characters=[
        {"name": "Carlos", "desc": "man mid-30s dark-hair"},
        {"name": "Elena", "desc": "woman late-20s white-dress"},
    ],
    locations=[
        {"name": "ext-beach-sunset", "time": "atardecer", "desc": "Beach at sunset"},
    ],
    language="es",
)

result = run_pipeline("mi_video.mp4", context, output_path="mi_video.raw.plf")
print(f"Generated PLF with {len(result.shots)} shots")
```

## CLI

```bash
python -m pipeline.pipeline video.mp4 context.json
```

## Dependencies

Heavy ML dependencies included only when the corresponding module is imported:
- `scenedetect[opencv]` — shot detection
- `openai-whisper` — transcription
- `ollama` — vision analysis (Qwen2-VL)
- `pyannote-audio` — speaker diarization (requires HF token)
- `panns-inference` — audio classification

## Format spec

See [PLF Format Specification](https://github.com/akolormedia/plf-spec).
