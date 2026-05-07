# PLF Pipeline — Video to PLF generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: Pre-alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)]()

Automatic generation of `.raw.plf` files from video footage. Orchestrates a
pipeline of AI models that analyze video and audio to produce structured PLF.

## Pipeline modules

- `scene_detector` — PySceneDetect wrapper for shot boundaries
- `frame_sampler` — Extract representative frames per shot
- `vision` — Qwen2-VL 7B for shot description and element detection
- `transcriber` — Whisper large-v3 for word-level transcript
- `diarizer` — pyannote-audio 3.x for speaker identification
- `audio_classifier` — PANNs for ambient/music classification
- `assembler` — Calls Model A to merge outputs into PLF text
- `pipeline` — Orchestrator with parallel execution

## Usage

```bash
python -m pipeline --video input.mp4 --context context.json --output input.raw.plf
```

## Dependencies

Heavy ML dependencies: PySceneDetect, Whisper, PyTorch, pyannote-audio, PANNs, Demucs.
Not suitable for the lightweight `plf-python` parser package.

## Format spec

See [PLF Format Specification](https://github.com/tu-org/plf-spec).
