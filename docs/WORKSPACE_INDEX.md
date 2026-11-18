# Workspace Index

This repository is now physically regrouped at the top level.

## Main Folders

- `scripts/`: runnable Python scripts, generators, diagnostics, and local test helpers
- `experiments/`: structured comparison workspaces, one-off evaluation harnesses, and ranking notes
- `library/`: long-form source-text collections and project-specific experiment trees
- `engines/`: engine-specific bundles that carry their own code, configs, and assets
- `demo/inputs/`: demo source text used for TTS and reading-video runs
- `demo/outputs/`: generated demo audio/video artifacts that are worth checking manually
- `config/`: requirements files and project config files
- `docs/`: repo documentation and workflow notes
- `inputs/`: loose root-style source/test text that is not part of the large content trees
- `outputs/`: bulky generated outputs, scratch renders, and test artifacts
- `logs/`: root-level log files
- `assets/`: loose audio samples, reference clips, and voice assets moved out of the root

## Scripts

The main entry points now live in `scripts/`:

- `audioChunkedKukuro.py`: Kokoro TTS generation
- `make_autoscroll_video.py`: autoscrolling reading-video renderer
- `make_reading_video.py`: phrase-card reading-video renderer
- `tts_download_production.py`: web TTS downloader for text-to-speech.online
- `textChunkingAdapter.py`: chunk planning for larger web-TTS work
- `voice_parameter_tester.py`: Kokoro voice comparison

## Library Contents

These long-form collections now live under `library/`:

- `PrometheusAtlas/`
- `amnesia/`
- `atlantis/`
- `bicameralMind/`
- `blackroots/`
- `christianityAndTraditioanReligion/`
- `credoMutwa/`
- `futureMoney/`
- `futurePhilosophy/`
- `giantsUponTheEarth/`
- `IzangomaHealingAfrica/`
- `knowing1/`
- `messagengersOfShamanicAfrica/`
- `sirius/`
- `southernRhodesiaMemories/`
- `spiritWorldOfAfrica/`
- `templesOfAfricanGods/`
- `yorubaTraditionalHealers/`
- `zimbabwesCultrualHeritage/`
- `zodiac/`

## Engine Bundles

- `engines/chatterbox-audio/`

## Asset Moves

- `assets/reference_audio/`
- `assets/voices/`

## Working Convention

- Put new demo source text in `demo/inputs/`.
- Put reviewable demo outputs in `demo/outputs/`.
- Put disposable large artifacts in `outputs/`.
- Put structured model comparisons and evaluation notes in `experiments/`.
- Put long-form source collections in `library/`.
- Keep reusable code in `scripts/`.

## Current Status

The repo root is now grouped by role, and the large content trees are no longer sitting directly at the top level.