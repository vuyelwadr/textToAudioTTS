# TTS Source Catalog

Updated: 2026-04-12

This file is the working shortlist for long-form audiobook TTS, with a bias toward:

- free local generation when possible
- low-cost APIs only when they still make sense at 10 to 50+ hours
- sources that are actually usable for narration, not just voice chat demos or benchmark pages

## Current Repo State

- Free disk space now: about 45 GiB free on the repo filesystem.
- Shared comparison environment size: about 1.4 GiB at `.venv-tts-experiments`.
- Pocket TTS environment: dedicated `.venv-pocket-tts` so the CPU-only baseline stays isolated from the shared experiment stack.
- Hugging Face cache size: about 3.9 GiB at `~/.cache/huggingface`.
- Synced local comparison outputs in `experiments/tts_naturalness/generated/all_samples/`: 46 files, including `24_xtts_v2.wav`, `40_spark_tts_0_5b_clone.wav`, `54_parler_tts_large_v1_plain.wav`, and `55_moss_tts_local_transformer_clone.wav`.
- `scripts/mahti/` now also includes a direct Parler-TTS Large CSC runner targeting slot `54_parler_tts_large_v1_plain.wav`. Retry job `33929825` completed and the output is now local.
- `scripts/mahti/` now also includes a direct Spark-TTS CSC runner targeting slot `40_spark_tts_0_5b_clone.wav`. Retry job `33929826` completed and the output is now local.
- `scripts/mahti/` now also includes a shared-dispatch MOSS-TTS Local runner targeting slot `55_moss_tts_local_transformer_clone.wav`. After OOM and torchcodec-save fixes, retry `33930342` completed and the output is now local.
- Prototype-only CSC runners now also exist for KugelAudio Open (`56`), Higgs Audio (`36/37`), Voxtral-TTS (`41/42`), Echo-TTS (`78/79`), Chatterbox-Turbo (`80/81`), and VibeVoice (`82/83/84`). Those files are present in-repo but were not yet wired into the shared dispatcher or generated locally when this status block was updated.
- The `experiments/tts_naturalness/generated/all_samples/top/all/` top-3 batch is also synced locally now, with 42 files across `qwen3_faster`, `voxcpm2`, and `vibevoice` for the 7-input top set.
- Verified distinct-source samples in `experiments/tts_naturalness/generated/all_samples/`:
   - `01_kokoro_82m_af_alloy.wav`
   - `03_text_to_speech_online_luke_neural.mp3`
   - `11_orpheus_tts_hf_space.wav`
   - `12_fish_speech_hf_space.wav`
   - `13_parler_tts_hf_space.wav`
   - `17_pocket_tts_alba.wav`
- Verified manifest for the current distinct-source set:
   - `experiments/tts_naturalness/generated/runs/run_manifest_20260410T001236Z.json`
- Current shared env package set:
  - `torch 2.6.0`
  - `torchaudio 2.6.0`
  - `kokoro 0.9.4`
  - `misaki 0.9.4`
  - `playwright 1.58.0`
  - `chatterbox-tts 0.1.4`
  - `librosa 0.11.0`
  - `transformers 4.46.3`
  - `diffusers 0.29.0`
  - `numpy 1.25.2`
  - `pykakasi 2.3.0`
   - `gradio_client 2.4.0`

## Latest User-Mandated Additions

The user explicitly raised the following models and adjacent techniques in the most recent takeover messages. This section overrides older “nice to have later” prioritization.

| Item | Exact target(s) | Current repo status | Required action |
| --- | --- | --- | --- |
| Parler large | `parler-tts/parler-tts-large-v1` | Direct CSC runner now wired; retry job `33929825` completed for slot `54_parler_tts_large_v1_plain.wav` and the WAV is local | Keep in immediate generation wave |
| VibeVoice family | `microsoft/VibeVoice-1.5B`, `microsoft/VibeVoice-Realtime-0.5B`, docs under `github.com/microsoft/VibeVoice/tree/main/docs` | Community-mirror prototype bootstrap/runner now exist for slots `82`–`84`, but they are not yet wired into the shared dispatcher and the weights remain research-use-only | Keep in the next CSC wave, but preserve the community-mirror + license caveats |
| MOSS-TTS | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` (1.7 B) + `OpenMOSS-Team/MOSS-Audio-Tokenizer` | **Wired — slot `55_moss_tts_local_transformer_clone.wav`** (voice-clone mode); runners: `scripts/mahti/bootstrap_moss_tts.sh` + `run_moss_tts.py`; pure-PyTorch, `transformers==5.0.0`; no ONNX; 24 kHz; sdpa attention on V100 | Mandatory shortlist item — Local 1.7B fits Puhti V100; voice-clone uses shared 8 s ref clip |
| Echo-TTS | `jordand/echo-tts-base` | Prototype bootstrap/runner added for slots `78` / `79`; not yet wired into the shared dispatcher or generated locally | Mandatory shortlist item; prioritize because the user specifically called it out and provided fine-tuning notes |
| Higgs Audio TTS | `bosonai/higgs-audio-v2-generation-3B-base` | Prototype bootstrap/runner added for slots `36` / `37`; not yet wired into the shared dispatcher or generated locally | Keep in top expansion wave |
| Spark-TTS | `SparkAudio/Spark-TTS-0.5B` | Direct CSC runner now wired; retry job `33929826` completed for slot `40_spark_tts_0_5b_clone.wav` and the WAV is local | Keep in immediate generation wave |
| KugelAudio Open | `kugelaudio/kugelaudio-0-open` and `github.com/Kugelaudio/kugelaudio-open` | Prototype bootstrap/runner added for slot `56`; open release uses preset voices only; not yet wired into the shared dispatcher or generated locally | Mandatory shortlist item; likely after Spark/MOSS |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-*` | Already wired on CSC | Keep in primary comparison set |
| Voxtral-TTS | `mistralai/Voxtral-4B-TTS-2603` | Prototype bootstrap/runner added for slots `41` / `42`; not yet wired into the shared dispatcher or generated locally | Keep in top expansion wave |
| Fish Audio TTS | `fishaudio/s2-pro`, `github.com/fishaudio/fish-speech` | Already wired on CSC | Keep in primary comparison set |
| Chatterbox-Turbo | `ResembleAI/chatterbox-turbo` | Prototype bootstrap/runner added for slots `80` / `81`; not yet wired into the shared dispatcher or generated locally | Keep in top expansion wave |
| GPT-SoVITS | `github.com/RVC-Boss/GPT-SoVITS` | Already wired on CSC | Keep in primary comparison set |
| Sesame CSM-1B | `sesame/csm-1b` | Already wired on CSC | Keep in primary comparison set |
| Kyutai TTS | `github.com/kyutai-labs/delayed-streams-modeling`, `kyutai/pocket-tts` | Already wired on CSC plus local Pocket TTS baseline | Keep in primary comparison set |
| F5-TTS | `SWivid/F5-TTS` | Already wired on CSC | Keep in primary comparison set |

## Latest Prototype Wave Status

| Model | Scripts | Slots | Current state |
| --- | --- | --- | --- |
| KugelAudio Open | `bootstrap_kugelaudio_open.sh`, `run_kugelaudio_open.py` | `56` | Prototype exists; preset-voice only; not yet shared-dispatch wired or generated locally |
| Higgs Audio | `bootstrap_higgs_audio.sh`, `run_higgs_audio.py` | `36`, `37` | Prototype exists; not yet shared-dispatch wired or generated locally |
| Voxtral-TTS | `bootstrap_voxtral.sh`, `run_voxtral.py` | `41`, `42` | Prototype exists; not yet shared-dispatch wired or generated locally |
| Echo-TTS | `bootstrap_echo_tts.sh`, `run_echo_tts.py` | `78`, `79` | Prototype exists; not yet shared-dispatch wired or generated locally |
| Chatterbox-Turbo | `bootstrap_chatterbox_turbo.sh`, `run_chatterbox_turbo.py` | `80`, `81` | Prototype exists; not yet shared-dispatch wired or generated locally |
| VibeVoice | `bootstrap_vibevoice.sh`, `run_vibevoice.py` | `82`, `83`, `84` | Community-mirror prototype exists; not yet shared-dispatch wired or generated locally |

## User-Reported Tuning And Post-Processing Leads

These are user-provided operating notes, not repo-verified findings yet, but they are important enough to track alongside the model list:

- **Spark-TTS prompt sensitivity:** zero-shot quality appears highly dependent on the prompt audio, with good results possible for specific voices but weak style/timbre transfer on many other prompts.
- **Spark-TTS fine-tuning track:** user reports strong speaker-style gains from LoRA fine-tuning with Unsloth, low VRAM use, and good results with roughly 30 minutes of training audio. Treat this as a potential second-phase path after zero-shot comparison runs.
- **Echo-TTS fine-tuning track:** user reports Echo-TTS as more stable but slightly noisier than Spark-TTS, with LoRA fine-tuning substantially improving speaker-style similarity. The follow-up research confirms Echo-TTS uses a bespoke diffusion stack rather than an Unsloth-friendly causal LM path, so treat it as a separate PEFT/diffusion project.
- **Deterministic Spark serving note:** user reports strong Spark-TTS throughput under SGLang with high batch, `bfloat16`, and deterministic inference. The follow-up research agrees that SGLang is the preferred serving path; if a VLLM path is tested, preserve the note that `VLLM_BATCH_INVARIANT=1` matters for batch-stable output.
- **Spark upsampling companions:** track `Flow-High` and `LavaSR` as post-processing companions for Spark-TTS because Spark emits 16 kHz audio. The follow-up research favors LavaSR as the immediate practical add and keeps Flow-High as the secondary quality-comparison option.
- **License caveat:** both Spark-TTS and Echo-TTS are currently tracked as CC-BY-NC-SA style non-commercial paths, so they remain fine for academic comparison but should be treated carefully for any later publication/distribution plan.

## Important Note About `sugegstion/`

- `experiments/tts_naturalness/sugegstion/` is an external-notes area, not a verified source of truth about this repo or this Mac.
- Treat those files as an idea pool and cross-check every claim against what actually runs here.

## What Was Generated

1. Core distinct-source comparison samples
   - Outputs:
     - `experiments/tts_naturalness/generated/all_samples/01_kokoro_82m_af_alloy.wav`
     - `experiments/tts_naturalness/generated/all_samples/03_text_to_speech_online_luke_neural.mp3`
     - `experiments/tts_naturalness/generated/all_samples/11_orpheus_tts_hf_space.wav`
     - `experiments/tts_naturalness/generated/all_samples/12_fish_speech_hf_space.wav`
     - `experiments/tts_naturalness/generated/all_samples/13_parler_tts_hf_space.wav`
   - Why: this is the current one-sample-per-source listening set.
2. Optional same-engine voice variants
   - These were generated during bring-up to verify alternate Kokoro and `text-to-speech.online` voices.
   - They are not part of the default comparison run anymore and were pruned from `generated/all_samples/`.

## What Was Installed And Set Up

1. Shared comparison venv
   - Local path: `.venv-tts-experiments`
   - Purpose: single experiment env for the sources already wired into this repo.
2. New runner and experiment workspace
   - Runner: `scripts/run_tts_naturalness_experiment.py`
   - Workspace: `experiments/tts_naturalness/`
   - Default behavior now runs the distinct-source set rather than every optional voice variant.
3. Pocket TTS baseline
   - Local path: `.venv-pocket-tts`
   - Runner: `scripts/generate_pocket_tts_sample.py`
   - Status: verified locally with the built-in `alba` voice and promoted into the stable listening set as `17_pocket_tts_alba.wav`.
4. Web baseline hardening
   - Exact voice locking for `LukeNeural(Male)` was added to avoid silent voice drift.
5. Kokoro harness fix
   - The runner enables `PYTORCH_ENABLE_MPS_FALLBACK=1` for Kokoro so Apple Silicon can fall back to CPU for unsupported MPS ops like `aten::angle`.
6. Chatterbox status
   - The code path was updated to use `audio_prompt_path` correctly.
   - It is still not verified end-to-end in this env.
   - Fresh failure is now runtime-side: `Failed to initialize Chatterbox: 'NoneType' object is not callable`.
7. CSC heavy-status checkpoint
   - Puhti bootstrap validation now passes for Fish Speech, F5-TTS, GPT-SoVITS, and OpenVoice V2.
   - Full Puhti jobs for those four models are submitted.
   - CSC helpers now auto-load the repo `.env` and map `HUGGINGFACE_API_KEY` to the Hugging Face token variables used by `huggingface_hub`.
   - Orpheus is repointed to `canopylabs/orpheus-3b-0.1-ft` now that gated access is available.

## What Is Still To Run

1. Chatterbox
   - Code path is wired.
   - Still blocked by dependency/runtime mismatch in the current shared env.
2. GPT-SoVITS
   - Still not wired into the local comparison runner.
   - Heavy CSC bootstrap is now validated on Puhti.
3. Dia2
   - Worth auditioning, but not a primary audiobook engine because generation is capped around 2 minutes.

## Why These First Picks

1. Kokoro 82M
   - It is the strongest practical open-weight model for this Mac right now.
   - It ranks well on Artificial Analysis while staying cheap or free.
   - Source URLs:
     - https://artificialanalysis.ai/text-to-speech/leaderboard
     - https://artificialanalysis.ai/text-to-speech/speech-explorer
2. Chatterbox
   - Strong quality reputation, open weights, and voice cloning.
   - Good candidate for narration if the local env stabilizes.
   - Source URLs:
     - https://github.com/resemble-ai/chatterbox
     - https://artificialanalysis.ai/text-to-speech/leaderboard
3. text-to-speech.online
   - Free right now, chunkable, and already integrated into this repo.
   - Good as a zero-cost baseline, not as the final quality ceiling.
   - Source URL:
     - https://www.text-to-speech.online/

## Cost Math Used Here

- Working assumption for rough audiobook estimates: 1 minute of audio is about 1,000 characters.
- 10 hours is about 600,000 characters.
- 50 hours is about 3,000,000 characters.
- For Artificial Analysis prices, see methodology:
  - https://artificialanalysis.ai/text-to-speech/methodology

These are rough planning numbers, not invoices.

## Gemini 2.5 Pro Preview TTS Quick Math

- Pricing page shows `gemini-2.5-pro-preview-tts` at `$1 / 1M text tokens` and `$20 / 1M audio tokens`.
- Google Cloud TTS pricing says Gemini TTS audio output is billed at `25 audio tokens per second`.
- Google AI forum guidance for chunking discussion uses `32 audio tokens per second` and a `16,000` output-token ceiling, which works out to about `8.33` minutes per request.
- That means `1M audio tokens` is about `11.11 hours` at `25 tok/sec`, or about `8.68 hours` at `32 tok/sec`.
- For `50 hours` of audio, the output side is about `4.5M` to `5.76M` audio tokens, so the output bill is roughly `$90` to `$115.20`.
- Input text cost is tiny by comparison for audiobook prose, usually under `$1` unless you repeat large steering prompts over and over.
- Paid tier says content is not used to improve Google products; free tier says it is.
- URLs:
   - https://ai.google.dev/gemini-api/docs/pricing
   - https://cloud.google.com/text-to-speech/pricing
   - https://discuss.ai.google.dev/t/gemini-2-5-tts-workflow-questions/87410

## Free Open-Source Or Local Models

### Strongest Real Candidates

1. Kokoro 82M v1.0
   - Type: open weights local TTS.
   - Audiobook fit: yes, with chunking.
   - Mac fit: good.
   - Why it matters: currently the easiest high-quality local path in this repo.
   - URLs:
     - https://artificialanalysis.ai/text-to-speech/leaderboard
     - https://artificialanalysis.ai/text-to-speech/speech-explorer

2. Fish Audio S2 Pro / Fish Speech
   - Type: open-source / open-weight multilingual TTS family.
   - Audiobook fit: yes, very promising for long-form and expressive narration.
   - Mac fit: weaker than Kokoro, stronger on NVIDIA/Linux.
   - Why it matters: best-ranked open weights on Artificial Analysis right now.
   - Caveat: license is Fish Audio Research License, not a simple MIT-style license.
   - URLs:
     - https://github.com/fishaudio/fish-speech
     - https://speech.fish.audio/install/
     - https://huggingface.co/fishaudio/s2-pro
     - https://artificialanalysis.ai/text-to-speech/leaderboard

3. Pocket TTS
    - Type: open-weight local TTS with a small CPU-first model.
    - Audiobook fit: yes, especially as a lightweight fallback or offline baseline.
    - Mac fit: very good.
    - Why it matters: easy local setup, fast startup, and now wired into this repo as a stable comparison source.
    - URLs:
       - https://pypi.org/project/pocket-tts/
       - https://huggingface.co/kyutai/pocket-tts
       - https://kyutai-labs.github.io/pocket-tts/

4. GPT-SoVITS
   - Type: few-shot voice cloning + TTS WebUI.
   - Audiobook fit: yes, especially if cloning and chunk control matter more than perfect plug-and-play simplicity.
   - Mac fit: officially tested on Apple Silicon, though quality and speed tradeoffs remain.
   - Why it matters: one of the most active free voice-cloning TTS stacks.
   - URLs:
     - https://github.com/RVC-Boss/GPT-SoVITS
     - https://huggingface.co/lj1995/GPT-SoVITS

5. Orpheus TTS
   - Type: open-source Llama-based TTS with streaming and emotional controls.
   - Audiobook fit: maybe, especially for expressive narration.
   - Mac fit: not ideal by default, but they do document a no-GPU Apple Silicon path through `orpheus-cpp`.
   - Why it matters: strong open model family with narration-oriented expressiveness.
    - Caveat: the main fast path depends on `vllm`, so it is still a CSC/Linux-first target here.
   - URLs:
     - https://github.com/canopyai/Orpheus-TTS
       - https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
       - https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained
       - https://huggingface.co/collections/canopylabs/orpheus-tts

6. Inworld TTS open code
   - Type: open MIT training/inference code for Inworld TTS-1 and TTS-1-Max.
   - Audiobook fit: yes in theory.
   - Mac fit: poor for your setup right now because the repo is tested on Ubuntu 22.04 with CUDA 12.4 or 12.8.
   - Why it matters: high-quality model family with open code and great benchmark results.
   - Caveat: the public GitHub repo is a trainer / bring-your-own-checkpoint inference stack, not a drop-in release of the official TTS-1 or TTS-1.5 weights. The official self-hosted path is a private on-prem container gated through Inworld sales, and the public Hugging Face org does not currently expose runnable TTS model weights.
   - URLs:
     - https://github.com/inworld-ai/tts
     - https://inworld-ai.github.io/tts/
     - https://arxiv.org/abs/2507.21138
     - https://docs.inworld.ai/tts/on-premises

7. Coqui XTTS v2
   - Type: multilingual voice cloning TTS.
   - Audiobook fit: yes, with chunking.
   - Mac fit: reasonable, but older quality tier than the newest leaders.
   - Why it matters: mature, widely used, huge community.
   - Caveat: license is CPML, not plain MIT.
   - URLs:
     - https://huggingface.co/coqui/XTTS-v2
     - https://github.com/coqui-ai/TTS

8. OpenVoice v2
   - Type: zero-shot cross-lingual voice cloning TTS.
   - Audiobook fit: maybe.
   - Mac fit: better than some CUDA-only projects, and Apple Silicon fixes are in the repo.
   - Why it matters: strong free commercial-use cloning option.
   - Caveat: usually a better cloning tool than a full narration pipeline by itself.
   - URLs:
     - https://github.com/myshell-ai/OpenVoice
     - https://research.myshell.ai/open-voice
     - https://arxiv.org/abs/2312.01479

9. StyleTTS 2
   - Type: older open-source high-naturalness TTS research model.
   - Audiobook fit: maybe, but more fragile than newer stacks.
   - Mac fit: limited and not repo-packaged for easy audiobook batching.
   - Why it matters: still historically important and can sound very good.
   - Caveat: more of a research repo than a clean long-form production path.
   - URLs:
     - https://github.com/yl4579/StyleTTS2
     - https://styletts2.github.io/
     - https://arxiv.org/abs/2306.07691

### High-Priority Expansion Candidates

1. F5-TTS
    - Type: open-weight voice-cloning TTS focused on strong speaker consistency.
    - Audiobook fit: high, especially for long single-speaker narration with a reference clip.
    - Repo fit: already wired on CSC and stronger than the browser baseline, so it should be treated as a primary comparison target instead of an optional extra.
    - URLs:
       - https://github.com/SWivid/F5-TTS
       - https://huggingface.co/SWivid/F5-TTS

2. OmniVoice
    - Type: open-weight multilingual speech LLM with cloning and voice design modes.
    - Audiobook fit: high-potential for expressive narration and speaker transfer.
    - Repo fit: already wired on CSC and should stay in the main listening pass.
    - URLs:
       - https://huggingface.co/k2-fsa/OmniVoice
       - https://pypi.org/project/omnivoice/

3. Qwen3-TTS
    - Type: open-weight TTS family with separate clone, custom-voice, and voice-design checkpoints.
    - Audiobook fit: high, because it offers multiple control modes instead of a single fixed synthesis path.
    - Repo fit: already wired on CSC and worth expanding beyond the single current voice-design sample.
    - URLs:
       - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
       - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
       - https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

4. Kyutai TTS
    - Type: open-weight delayed-streams speech model from Kyutai.
    - Audiobook fit: promising for naturalness and streaming-style generation, even if it is not yet as battle-tested for book pipelines as Kokoro or F5.
    - Repo fit: already wired on CSC and should be treated as a serious comparison slot, not just a curiosity.
    - URLs:
       - https://huggingface.co/kyutai/pocket-tts
       - https://github.com/kyutai-labs/delayed-streams-modeling

5. Step Audio EditX
    - Type: open-weight speech generation/editing model currently ranked near the very top of the open-weights leaderboard.
    - Audiobook fit: high-potential, especially if the editing workflow proves good for fixing specific passages without full rerenders.
    - Repo fit: one of the clearest missing top-tier open-weight families.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

6. Maya1
    - Type: open-weight TTS family ranked near the top of the current open-weights leaderboard.
    - Audiobook fit: high-potential based on blind-listening signal alone.
    - Repo fit: another obvious shortlist gap if the goal is to hear the strongest open models rather than just the easiest local ones.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

7. Voxtral TTS
    - Type: open-weight TTS family from Mistral currently appearing in the open-weights leaderboard.
    - Audiobook fit: promising enough to audition given its leaderboard position.
    - Repo fit: worth treating as a serious next-add candidate instead of leaving it out of the catalog.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

8. Magpie-Multilingual 357M
    - Type: compact open-weight multilingual TTS model from NVIDIA.
    - Audiobook fit: promising where multilingual narration or accent flexibility matters more than maximal expressiveness.
    - Repo fit: attractive because the parameter count is modest compared with the giant 7B-class candidates.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

9. Zonos-v0.1
    - Type: open-weight expressive TTS and cloning stack.
    - Audiobook fit: promising for emotional narration and character voices.
    - Repo fit: it is one of the most discussed open-weight missing families in the current repo notes.
    - URLs:
       - https://github.com/Zyphra/Zonos
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

10. VibeVoice 7B
    - Type: larger open-weight TTS family ranked in the current open-weights leaderboard.
    - Audiobook fit: promising, but likely a CSC-first target rather than a Mac-first one.
    - Repo fit: should be treated as a heavyweight audition target, not just a watchlist footnote.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

11. MetaVoice v1
    - Type: open-weight voice-cloning TTS.
    - Audiobook fit: moderate-to-promising if the cloning path is more important than state-of-the-art Elo.
    - Repo fit: still worth a direct listening test because it remains one of the better-known open cloning stacks.
    - URLs:
       - https://github.com/metavoiceio/metavoice-src
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights

12. Fish Speech 1.5
    - Type: earlier open-weight Fish Speech release that still ranks well independently of S2 Pro.
    - Audiobook fit: good enough that it deserves a direct A/B against S2 Pro rather than being treated as obsolete by default.
    - Repo fit: useful for understanding whether the newer Fish stack is actually better for your narration style.
    - URLs:
       - https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights
       - https://github.com/fishaudio/fish-speech

### More Situational Or Lower Priority

10. Bark
   - Type: text-to-audio model, not traditional narration-first TTS.
   - Audiobook fit: weak.
   - Why it matters: expressive, still useful for special effects or weird stylized lines.
   - Caveat: short default outputs, high variance, can drift away from the prompt.
   - URLs:
     - https://github.com/suno-ai/bark
     - https://huggingface.co/spaces/suno/bark

11. Dia2-2B
   - Type: streaming dialogue TTS.
   - Audiobook fit: weak as a primary engine.
   - Why it matters: interesting if you want dialogue or realtime turn-taking.
   - Caveat: explicitly capped around 2 minutes, focused on dialogue, wants CUDA 12.8+ for the intended path.
   - URLs:
     - https://huggingface.co/nari-labs/Dia2-2B
     - https://github.com/nari-labs/dia2
     - https://huggingface.co/spaces/nari-labs/Dia2-2B

## Free Or Hacky Online Sources

1. text-to-speech.online
   - Best use: free chunked audiobook baseline when you can tolerate UI automation.
   - Caveat: web UI stability, no SLA, and possible site changes.
   - URL:
     - https://www.text-to-speech.online/

2. Artificial Analysis Speech Explorer
   - Best use: listen to many model families quickly before integrating anything.
   - Caveat: it is for auditioning and benchmarking, not bulk audiobook generation.
   - URL:
     - https://artificialanalysis.ai/text-to-speech/speech-explorer

3. Artificial Analysis Leaderboard
   - Best use: price/quality screening.
   - Caveat: it benchmarks serverless APIs and some hosted open-weight endpoints, not your exact local setup.
   - URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

4. TTS Arena and TTS Arena V2 on Hugging Face
   - Best use: discovery and subjective listening.
   - Caveat: these are benchmark/demo spaces, not a practical pipeline for 10 to 50 hours.
   - URLs:
     - https://huggingface.co/spaces/TTS-AGI/TTS-Arena
     - https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2

5. Hugging Face Spaces for specific models
   - Best use: quick audition before local install.
   - Caveat: usually not suitable for sustained long-form production.
   - URLs:
     - https://huggingface.co/spaces/nari-labs/Dia2-2B
     - https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS
     - https://huggingface.co/spaces/coqui/xtts

## Cheap API Or Commercial Options

These rough audiobook cost estimates use the 1,000 characters per minute assumption.

1. Inworld TTS 1.5 Mini
   - Benchmark position: one of the best current quality/price combinations.
   - Approx usage cost: about $3 for 10 hours, about $15 for 50 hours at founder-rate pricing.
   - Real budget caveat: to keep founder rates and get a creator workflow with downloads, the plan floor is about $25 per month.
   - Verdict: the only paid option in your current budget range that still looks genuinely interesting.
   - URLs:
     - https://artificialanalysis.ai/text-to-speech/model-families/inworld
     - https://artificialanalysis.ai/text-to-speech/leaderboard
     - https://inworld.ai/founder-pricing
     - https://inworld.ai/tts

2. Inworld TTS 1.5 Max
   - Approx usage cost: about $6 for 10 hours, about $30 for 50 hours at founder-rate pricing.
   - Verdict: quality monster, but 50 hours starts pushing beyond your stated budget once plan cost is included.
   - URLs:
     - https://artificialanalysis.ai/text-to-speech/model-families/inworld
     - https://inworld.ai/founder-pricing

3. Gemini 2.5 Flash Lite TTS
   - Approx usage cost: about $6 for 10 hours, about $30 for 50 hours.
   - Verdict: plausible budget option if Google’s actual access terms fit your workflow.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

4. Hume Octave 2
   - Approx usage cost: about $4.56 for 10 hours, about $22.8 for 50 hours.
   - Verdict: price is good on paper, worth listening to if expressiveness matters.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

5. Async Flash v1.0
   - Approx usage cost: about $5 for 10 hours, about $25 for 50 hours.
   - Verdict: cost is interesting, but it is less proven as an audiobook-first choice than Inworld Mini.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

6. OpenAI TTS-1
   - Approx usage cost: about $9 for 10 hours, about $45 for 50 hours.
   - Verdict: okay for 10 hours, weak for 50 hours on your budget.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

7. Fish Audio S2 Pro API
   - Approx usage cost: about $9 for 10 hours, about $45 for 50 hours.
   - Verdict: borderline for your budget, but attractive if local S2 setup is too heavy.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

8. Azure Neural
   - Approx usage cost: about $9 for 10 hours, about $45 for 50 hours.
   - Verdict: respectable baseline, but not exciting enough versus cheaper leaders.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

9. Speechify SIMBA 1.6
   - Approx usage cost: about $6 for 10 hours, about $30 for 50 hours.
   - Verdict: paper price is okay, but not enough direct evidence yet to put it ahead of Inworld Mini.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

10. Cartesia Sonic 3
   - Approx usage cost: about $28 for 10 hours, about $140 for 50 hours.
   - Verdict: too expensive for your stated budget.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

11. Eleven v3
   - Approx usage cost: about $124 for 10 hours, about $618 for 50 hours using Artificial Analysis subscription-normalized pricing.
   - Verdict: out of scope for your budget.
   - Source URL:
     - https://artificialanalysis.ai/text-to-speech/leaderboard

## Things That Look Relevant But Are Not Primary Audiobook Engines

1. Ultravox
   - It is a multimodal voice interaction model that currently takes audio and emits streaming text.
   - It is not the thing to build a 10 to 50 hour audiobook pipeline around.
   - URLs:
     - https://github.com/fixie-ai/ultravox
     - https://ultravox.ai/

2. Dia2
   - Great for streaming dialogue experiments, not ideal for long-form single-speaker narration.
   - The issue is not just that you have to chunk. Lots of TTS stacks chunk. The issue is that Dia2 is designed around short dialogue turns, so the chunk boundaries are much more likely to sound like turn resets than with a calmer narration-oriented model such as Kokoro.
   - URLs:
     - https://github.com/nari-labs/dia2
     - https://huggingface.co/nari-labs/Dia2-2B

3. TTS Arena / TTS Arena V2 / Speech Explorer
   - Useful for ranking by ear.
   - Not bulk generation platforms.
   - URLs:
     - https://huggingface.co/spaces/TTS-AGI/TTS-Arena
     - https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2
     - https://artificialanalysis.ai/text-to-speech/speech-explorer

## Community Signals Worth Watching

1. Reddit comparison thread
   - Useful because it focuses on realism, emotions, expressiveness, install difficulty, VRAM, and real-time factor.
   - It is community opinion, not a controlled benchmark.
   - Models explicitly raised there include Bark, Dia, Kokoro, Qwen3-TTS, VibeVoice, MOSS-TTS, HumeAI/tada, SoulX, neuphonic/neutts, Supertonic 2, Maya1, Veena, kittenTTS, and MegaTTS.
   - URLs:
     - https://old.reddit.com/r/LocalLLM/comments/1s31xrs/tts_model_comparison_chart_my_personal_rankings/
     - https://github.com/mirfahimanwar/TTS-Model-Comparison-Chart/

## My Practical Recommendation For Your Budget

1. Free-first path
   - Keep using Kokoro locally.
   - Add Fish Speech next.
   - Add GPT-SoVITS next after that if voice cloning matters.
   - Keep text-to-speech.online as the free web fallback.

2. If you are willing to pay a little
   - The one paid family that actually fits your stated ceiling is Inworld Mini founder-rate pricing.
   - Everything ElevenLabs-like is basically out of scope for 50 hours.

3. What I would not prioritize for audiobook work
   - Ultravox
   - Dia2 as your main narration engine
   - Bark as the main narrator
   - TTS Arena / Speech Explorer as if they were production tools

## Space Reclaiming

You asked whether we can install, run, then delete to save space. Yes.

Use the cleanup helper:

```bash
python scripts/cleanup_tts_experiment_space.py --help
```

Examples:

```bash
python scripts/cleanup_tts_experiment_space.py --targets experiment-runs chatterbox-hf --execute
python scripts/cleanup_tts_experiment_space.py --targets venv --execute
```

---

## Expansion Research Report — 2025–2026 New Model Candidates

*Researched and compiled July 2025. Covers models not yet wired into slots 01–30.*
*Next available output-file slot: 31_*

### What Is Already Covered (Slots 01–30)

| Slot range | Integration path | Models |
|---|---|---|
| 01–04 | Local Mac (MPS/CPU) venv | Kokoro 82M v1.0, Kokoro v0.19 legacy, Chatterbox (broken), Pocket TTS |
| 05–10 | Local Mac (MPS/CPU) venv | text-to-speech.online |
| 11–13 | HF Space / Playwright | Orpheus TTS, Fish Speech, Parler TTS |
| 21–30 | CSC Mahti A100 SLURM | Orpheus 3B, Fish Speech S2 Pro, F5-TTS, XTTS v2, GPT-SoVITS, OpenVoice v2, OmniVoice, Qwen3-TTS 1.7B VoiceDesign, Qwen3-TTS 1.7B CustomVoice, Kyutai TTS |

**Not yet wired but flagged in earlier catalog edits:** StyleTTS2, Bark/Coqui TTS, ElevenLabs, Cartesia, Google Speech Gen, Step Audio EditX, Maya1, Voxtral TTS, Magpie-Multilingual 357M, Inworld TTS.

---

### New Candidate Models (25+)

Each entry covers: type, parameters, license, VRAM/compute needs, integration path, and verdict.

**Abbreviations:**
- Clone = zero-shot voice cloning from short reference audio
- VD = voice design (natural-language voice description)
- Emo = emotion control tags
- CSC = CSC Mahti A100 SLURM (preferred for >8GB VRAM models)
- Local = local Mac MPS/CPU

---

#### Tier 1 — Implement First

**1. Fun-CosyVoice3 / CosyVoice2** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`)

- Type: Clone + Emo, multilingual
- Params: 0.5B
- License: Apache 2.0
- VRAM: ~6 GB
- Languages: 9 major + 18 Chinese dialects
- Highlights: Industry-leading CER (0.81%), speaker similarity (77.4%), 150ms streaming latency, bi-directional streaming, pronunciation inpainting, vLLM/TensorRT-LLM compatible. MOS 5.53. Alibaba FunAudio team. Uses `reference_voice.wav` for zero-shot cloning.
- Integration path: CSC slot 31. `pip install -r requirements.txt` from `FunAudioLLM/CosyVoice` GitHub. Also runs on Puhti V100. Requires conda or isolated venv due to WeTextProcessing dependency.
- URLs: https://github.com/FunAudioLLM/CosyVoice — https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- **Verdict: HIGH — best value/quality/size ratio of any newcomer. Implement now.**

**2. Sesame CSM-1B** (`sesame/csm-1b`)

- Type: Clone, context-aware dialogue
- Params: 1B
- License: Apache 2.0
- VRAM: ~8 GB
- Languages: English
- Highlights: Conversational Speech Model — generates contextually appropriate speech given full dialogue context (prior turns, speaker roles). Natively supported in Hugging Face Transformers >=4.52.1 (May 2025). Sesame AI Labs. Built-in watermarking. Uses `reference_voice.wav` for speaker conditioning.
- Integration path: CSC slot 32. `pip install "transformers>=4.52.1"` in an isolated venv (shared venv has 4.46.3 — incompatible). Standard `AutoModel` pipeline.
- Current repo status: wired and verified on Puhti as `scripts/mahti/bootstrap_csm_1b.sh` + `scripts/mahti/run_csm_1b.py`. The current comparison outputs are `32_csm_1b_plain.wav` and `76_csm_1b_context_clone.wav`. The clone path now bypasses TorchCodec by preloading the reference audio with `soundfile` and feeding waveform arrays directly into the chat template.
- URLs: https://github.com/SesameAILabs/csm — https://huggingface.co/sesame/csm-1b
- **Verdict: HIGH — unique context-aware dialogue approach; clean HF integration; fills a niche no existing slot covers.**

**3. Dia 1.6B / Dia2** (`nari-labs/Dia-1.6B`, `nari-labs/Dia2-2B`)

- Type: Clone, multi-speaker dialogue with nonverbal sounds
- Params: 1.6B (Dia) / 1B or 2B (Dia2)
- License: Apache 2.0
- VRAM: ~8 GB (Dia 1.6B)
- Languages: English
- Highlights: Dialogue-first — uses [S1]/[S2] speaker tags and inline nonverbal annotations (laughs, sighs, gasps). Voice conditioning from reference audio. Dia2 adds streaming + smaller 1B variant. Best choice for dialogue/podcast-style naturalness samples.
- Integration path: CSC slot 33. `pip install git+https://github.com/nari-labs/dia.git`. Dia2 (1B) may also run on local Mac with MPS fallback.
- Current repo status: wired and verified on Puhti as `scripts/mahti/bootstrap_dia.sh` + `scripts/mahti/run_dia.py`. The current comparison outputs are `33_dia_1_6b_dialogue.wav` and `77_dia_1_6b_clone.wav`, with the clone run using `haCxgJSFmFc_reference_8s.wav`.
- URLs: https://github.com/nari-labs/dia — https://huggingface.co/nari-labs/Dia-1.6B
- **Verdict: HIGH — unique dialogue/nonverbal-sound coverage not present in any existing slot.**

**4. IndexTTS-2** (`IndexTeam/IndexTTS-2`)

- Type: Clone + Emo + duration control
- Params: ~50M VAE + large GPT backbone
- License: Apache 2.0
- VRAM: ~12 GB
- Languages: Chinese, English
- Highlights: Lowest published WER (0.2418) of any open model in benchmarks. Disentangles emotion from speaker identity — can control timbre and emotion via separate reference prompts. FSQ codebook, BPE tokenizer, GPT latent representations for clarity. Explicit duration-control mode for video dubbing. Trained on 50,000+ hours.
- Integration path: CSC slot 34. Clone `IndexTeam/IndexTTS-2` repo; standard `inference.py` pattern. Fits Puhti V100 (12GB model).
- URLs: https://huggingface.co/IndexTeam/IndexTTS-2
- **Verdict: HIGH — best-in-class accuracy; essential for a naturalness benchmark.**

**5. Neuphonic NeuTTS Air** (`neuphonic/neutts-air`)

- Type: Clone
- Params: 0.5B (Qwen backbone) + NeuCodec 50Hz
- License: Apache 2.0
- VRAM: CPU-only sufficient; MPS GPU also supported via llama-cpp-python Metal
- Languages: English (multilingual Nano variants: DE, FR, ES)
- Highlights: GGUF quantized (Q4/Q8); runs real-time on laptop CPUs including Apple M-series. 3–15s reference audio for voice cloning. Fills the critical local Mac expansion gap — Kokoro is the only strong local slot currently. Instant install: `pip install neutts[all]`.
- Integration path: **Local Mac slot 05** (or 06). Install into shared venv or isolated venv. Requires `brew install espeak-ng` for phonemization. `PYTORCH_ENABLE_MPS_FALLBACK=1` not needed (CPU path).
- URLs: https://github.com/neuphonic/neutts — https://huggingface.co/neuphonic/neutts-air
- **Verdict: HIGH — only newcomer that comfortably runs on Mac CPU with voice cloning. Fills critical local gap.**

**6. Hume AI TADA-1B / TADA-3B-ml** (`HumeAI/tada-1b`, `HumeAI/tada-3b-ml`)

- Type: Plain TTS (zero hallucination architecture), narration-optimized
- Params: 1B (English) / 3B (multilingual, 11 languages)
- License: MIT
- VRAM: ~6 GB (1B) / ~16 GB (3B)
- Languages: English (1B); 11 languages (3B)
- Highlights: Text-Acoustic Dual Alignment — strict 1:1 token/audio mapping eliminates content hallucinations by construction. RTF 0.09 (11× faster than real-time on A100). Supports up to 700-second single-pass generation. Llama-based backbone. Ideal for long-form narration / audiobook use cases. March 2026 release.
- Integration path: CSC slot 39 (TADA-1B may also run locally on Mac MPS). GitHub: https://github.com/HumeAI/tada.
- URLs: https://huggingface.co/HumeAI/tada-1b — https://huggingface.co/HumeAI/tada-3b-ml
- **Verdict: HIGH — zero-hallucination + long-form + 11x RT speed makes it uniquely valuable for audiobook evaluation.**

**7. Maya1** (`maya-research/maya1`)

- Type: VD + Emo (inline tags), voice design
- Params: 3B (Llama-style backbone + SNAC codec)
- License: Apache 2.0
- VRAM: ~16 GB
- Languages: English (multi-accent)
- Highlights: Natural-language voice descriptions, inline emotion tags (`<cry>`, `<snort>`, `<laugh>`), 24 kHz mono streaming, vLLM compatible, production-ready. Backed by South Park Commons. Nov 2025.
- Integration path: CSC slot 38. vLLM inference recommended (>=0.9.0). Uses `AutoModelForCausalLM`.
- URLs: https://huggingface.co/maya-research/maya1
- **Verdict: HIGH — voice design + inline emotion; unique in English; Apache 2.0; no equivalent in existing slots.**

**8. Higgs Audio V2 / V2.5** (`bosonai/higgs-audio-v2-generation-3B-base`)

- Type: Clone + Emo, multi-speaker dialogue
- Params: 5.77B (full) / 3B (base); built on Llama 3.2 3B backbone
- License: Apache 2.0
- VRAM: ~28 GB (5.77B full) / ~16 GB (3B base)
- Languages: English + multilingual
- Highlights: #1 trending TTS on HF throughout Q3–Q4 2025. Dual-FFN expert architecture for audio-specific boosting. 10M+ hours training. Best-in-class emotion, question-asking inflection, dialogue naturalness. V2.5 (Dec 2025) adds production stability.
- Integration path: CSC slot 36 — Mahti A100 80GB (5.77B full) or Puhti V100 (3B base). Uses `bosonai/higgs-audio` GitHub repo.
- URLs: https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base — https://github.com/boson-ai/higgs-audio
- **Verdict: HIGH — best emotion/expressiveness open model available; requires A100 for full model.**

**9. MegaTTS3** (`ByteDance/MegaTTS3`)

- Type: Clone, zero-shot
- Params: 0.45B (Sparse Alignment Enhanced Latent Diffusion Transformer)
- License: Apache 2.0
- VRAM: ~8 GB
- Languages: Chinese, English
- Highlights: WaveVAE compresses 24 kHz audio to 25Hz latents. Qwen2.5-0.5B G2P model. Good latency on short texts. ByteDance. Linux-first (conda env required). March 2025.
- Integration path: CSC slot 35. Requires conda; clone `ByteDance/MegaTTS3` repo. Linux-only limits local Mac path.
- URLs: https://github.com/bytedance/MegaTTS3
- **Verdict: MEDIUM-HIGH — solid quality/size ratio from a credible team; Linux-only is a minor friction.**

**10. GLM-TTS** (`zai-org/GLM-TTS`)

- Type: Clone, RL-optimized expressiveness
- Params: not disclosed; Qwen/GLM backbone
- License: Open source — check LICENSE for commercial terms before use
- VRAM: ~12 GB
- Languages: English, Chinese
- Highlights: GRPO multi-reward reinforcement learning training. 13.6% CER reduction vs base model. Explicitly optimized for emotional expressiveness while preserving speaker similarity. ZhipuAI / ZAI (China). Dec 2025.
- Integration path: CSC slot 37. `git clone; pip install -r requirements.txt`. WeTextProcessing may need manual install order on Linux.
- URLs: https://github.com/THUDM/GLM-TTS — https://huggingface.co/zai-org/GLM-TTS
- **Verdict: MEDIUM-HIGH — RL-expressiveness angle is novel; confirm commercial license terms before use.**

---

#### Tier 2 — Next Wave

**11. Spark-TTS-0.5B** (`SparkAudio/Spark-TTS-0.5B`)

- Type: Clone + Emo
- Params: 0.5B (Qwen2.5 backbone, single-stream decoupled speech tokens)
- License: Apache 2.0
- VRAM: ~6 GB; can run on Mac MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Languages: Chinese, English
- Integration path: Local Mac slot 06 or CSC slot 40. `pip install -e .` from `SparkAudio/Spark-TTS` repo.
- URLs: https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- **Verdict: MEDIUM — compact and flexible; dual local/CSC paths; good complement to CosyVoice3.**

**12. Llasa-3B** (`HKUSTAudio/Llasa-3B`)

- Type: Clone
- Params: 3B (also 1B and 8B variants available)
- License: check HF repo (likely Apache 2.0)
- VRAM: ~16 GB (3B)
- Languages: English (multilingual 1B variant also available)
- Highlights: Benchmarked alongside Orpheus, CSM, MegaTTS3 by Inferless; competitive quality/size balance.
- Integration path: CSC slot 41.
- URLs: https://huggingface.co/HKUSTAudio/Llasa-3B
- **Verdict: MEDIUM — solid but less differentiated vs Orpheus 3B already in slot 21.**

**13. Chatterbox-Turbo** (`ResembleAI/chatterbox-turbo`)

- Type: Clone, production low-latency
- Params: 350M
- License: MIT
- VRAM: ~4 GB; MPS-capable on Apple Silicon
- Languages: English
- Highlights: Faster and lower VRAM than full Chatterbox. Resemble AI. ONNX variant also available. Since full Chatterbox (slot 03) is currently broken (`NoneType object is not callable`), Turbo is the recommended replacement path.
- Integration path: Local Mac slot 03 (replace broken Chatterbox) or new slot 07. `pip install chatterbox-turbo` (separate package from `chatterbox-tts`; use isolated venv to avoid conflicts).
- URLs: https://huggingface.co/ResembleAI/chatterbox-turbo
- **Verdict: MEDIUM-HIGH — practical fix for broken slot 03; lower VRAM; MIT license.**

**14. MeloTTS** (`myshell-ai/MeloTTS-English`)

- Type: Plain TTS, CPU-first multilingual
- Params: small encoder-decoder (not disclosed)
- License: MIT
- VRAM: CPU-only; extremely fast
- Languages: English, Chinese, Japanese, Korean, Spanish, French
- Highlights: Designed for CPU inference without quality sacrifice. MyShell AI. Easiest install of any newcomer. Good multilingual baseline.
- Integration path: Local Mac slot 08. `pip install melo-tts`.
- URLs: https://github.com/myshell-ai/MeloTTS
- **Verdict: MEDIUM — great CPU multilingual baseline; less unique if Kokoro is already the primary local model.**

**15. VibeVoice-Realtime-0.5B** (`microsoft/VibeVoice-Realtime-0.5B`)

- Type: Plain TTS, streaming, style-controlled
- Params: 0.5B
- License: MIT (weights)
- VRAM: ~4 GB; possibly MPS-capable
- Languages: 9 languages + 11 English style voices
- Highlights: Ultra-low 7.5Hz tokenizer for real-time streaming. Microsoft. 9 languages + 11 English styles.
- **CAVEAT:** Microsoft removed the TTS inference code from GitHub in September 2025 citing misuse concerns. Weights remain on HF but the training/inference repository was pulled. Verify current inference code availability before investing integration time.
- Integration path: Local Mac or CSC slot 42 — verify code availability first.
- URLs: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- **Verdict: MEDIUM — interesting if code is available; do not prioritize until code status is confirmed.**

**16. VibeVoice-1.5B** (`microsoft/VibeVoice-1.5B`)

- Type: Clone, multi-speaker long-form (up to 90 min, 4 simultaneous speakers)
- Params: 1.5B
- License: MIT (weights)
- VRAM: ~12 GB
- Languages: 9 languages
- **CAVEAT:** Same inference-code removal issue as Realtime above. Weights-only on HF.
- Integration path: CSC slot 43 — only if inference code is confirmed available.
- **Verdict: MEDIUM — compelling multi-speaker long-form capability; currently blocked by code unavailability.**

**17. E2-TTS** (`SWivid/E2-TTS`)

- Type: Clone, flow-matching (sibling of F5-TTS)
- Params: similar to F5-TTS (~335M)
- License: check repo (same authors as F5-TTS; likely Apache 2.0)
- VRAM: ~8 GB
- Languages: English, Chinese
- Highlights: Alternative simpler flow-matching objective from the F5-TTS authors. Largely shares the F5 codebase and environment — low marginal integration cost if F5-TTS (slot 23) is already running.
- Integration path: CSC — reuse F5-TTS conda/venv environment; slot 44.
- URLs: https://huggingface.co/SWivid/E2-TTS
- **Verdict: LOW-MEDIUM — easy add given F5 overlap; limited independent quality signal.**

**18. Zonos-v0.1-hybrid** (`Zyphra/Zonos-v0.1-hybrid`)

- Type: Clone + Emo, conditioning-rich
- Params: not disclosed; hybrid Transformer + Mamba architecture
- License: Zyphra Permissive License — check commercial terms carefully
- VRAM: ~12 GB; 44 kHz audio output
- Languages: English + limited multilingual
- Highlights: Rich conditioning: pace, pitch, emotion, speaking rate, filler sounds. High audio quality (44 kHz). Zyphra.
- Integration path: CSC slot 45.
- URLs: https://huggingface.co/Zyphra/Zonos-v0.1-hybrid
- **Verdict: MEDIUM — strong voice conditioning; review Zyphra license commercial terms before publishing audio.**

**19. Supertone Supertonic-2** (`Supertone/supertonic-2`)

- Type: Plain TTS
- Params: 66M
- License: OpenRAIL-M — review commercial use restrictions before publishing generated audio
- VRAM: CPU-first; <1 GB GPU
- Languages: English + 4 others
- Highlights: Tiny 66M model, flow-matching, handles numbers/dates/currency without G2P preprocessing. ONNX available. WebGPU-capable browser deployment.
- Integration path: Local Mac slot 09.
- URLs: https://huggingface.co/Supertone/supertonic-2
- **Verdict: LOW-MEDIUM — useful local tiny-model baseline; OpenRAIL-M license needs review.**

**20. OuteTTS / Llama-OuteTTS-1.0-1B** (`OuteAI/Llama-OuteTTS-1.0-1B`)

- Type: Plain TTS, LLM-based
- Params: 1B (also 0.6B)
- License: Apache 2.0
- VRAM: ~8 GB; GGUF variants for CPU inference
- Languages: English
- Highlights: Pure LLM autoregressive approach. GGUF variants enable CPU inference on Mac. However, benchmarks show highest latency of all newcomers tested (>4 min for 200 words with GGUF) — unsuitable for batch naturalness runs at scale.
- Integration path: Local Mac (GGUF) or CSC slot 46.
- URLs: https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B
- **Verdict: LOW — interesting architecture but latency disqualifies it for batch experiment runs.**

**21. VoxCPM / VoxCPM2** (`openbmb/VoxCPM-0.5B`)

- Type: Clone
- Params: 0.5B (VoxCPM), larger (VoxCPM2)
- License: Apache 2.0
- Languages: Chinese, English
- Integration path: CSC slot 47.
- URLs: https://huggingface.co/openbmb/VoxCPM-0.5B
- **Verdict: LOW-MEDIUM — limited English evaluation evidence; primarily Chinese-focused.**

**22. FireRedTTS2** (`FireRedTeam/FireRedTTS2`)

- Type: Clone, long-form multi-speaker streaming
- Params: not disclosed
- License: Apache 2.0
- Languages: 7 languages including English
- Highlights: Designed for stable long-form dialogue generation. Multi-speaker streaming. Sept 2025.
- Integration path: CSC slot 48.
- URLs: https://huggingface.co/FireRedTeam/FireRedTTS2
- **Verdict: MEDIUM — good complement to Dia for long dialogue; less benchmarked publicly.**

**23. Step-Audio-TTS-3B / Step-Audio-EditX** (`stepfun-ai/Step-Audio-TTS-3B`, `stepfun-ai/Step-Audio-EditX`)

- Type: EditX = iterative speech editing; TTS-3B = expressive standard TTS
- Params: 3B (~4B BF16 footprint, or 6–8 GB with AWQ 4-bit)
- License: Apache 2.0
- VRAM: 12–15 GB standard; 6–8 GB AWQ 4-bit; 8–10 GB (memory-efficient mode)
- Languages: Chinese, English, Japanese, Korean, Sichuanese, Cantonese
- Highlights: 14 emotion control tags. Polyphone disambiguation. EditX uniquely enables iterative refinement (re-run with edited voice instruction). Generalizes to style-transfer from closed-source reference voices. Both variants already listed as expansion candidates in earlier catalog.
- Integration path: CSC slot 49 (EditX) or slot 50 (TTS-3B). Uses vLLM >=0.9.0 for serving.
- URLs: https://huggingface.co/stepfun-ai/Step-Audio-EditX — https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B
- **Verdict: MEDIUM-HIGH — EditX iterative-refinement is a unique capability; 14 emotions; confirm CSC vLLM version.**

**24. NVIDIA Magpie TTS Multilingual**

- Type: Plain TTS, 9-language
- Params: transformer encoder-decoder (not disclosed)
- License: NVIDIA Open Model License — non-commercial research use only; commercial use requires NVIDIA agreement
- Languages: EN-US, ES-EU, DE, FR, IT, VI, ZH, HI, JA
- Highlights: Multi-codebook prediction (8 codebooks). GRPO-trained. Released Feb 2026 via NVIDIA NGC.
- Integration path: NVIDIA NIM API endpoint only — no self-hosting weights available under open terms. Fits "API" integration category (separate from CSC pattern).
- URLs: https://build.nvidia.com/nvidia/magpie-tts-multilingual
- **Verdict: LOW for CSC self-hosting (weights not freely available); MEDIUM for API-based quality baseline.**

**25. Voxtral-4B-TTS** (`mistralai/Voxtral-4B-TTS-2603`)

- Type: Plain TTS + voice cloning (HF Space demo live)
- Params: 4B
- License: Mistral Research License — check current terms; gated weights
- VRAM: ~24 GB
- Languages: Multilingual (EN + others)
- Highlights: HF Space demo live. 4B Mistral backbone. Already listed as expansion candidate.
- Integration path: HF Space (gradio_client Playwright) slot 14 as fastest path; or CSC slot 51 if weights become accessible.
- URLs: https://huggingface.co/spaces/mistralai/voxtral-tts-demo
- **Verdict: MEDIUM — HF Space path is low-friction; 4B size means CSC A100 required for self-hosted.**

**26. Inworld TTS-1 / TTS-1-Max** (Inworld AI)

- Type: Plain TTS, commercial API + open-source code
- License: Apache 2.0 (code); hosted API has free 2M char/month tier
- Highlights: #1 ELO on Artificial Analysis leaderboard (score 1236 as of July 2025). Open-sourced model code on GitHub, but the public repo expects a trained checkpoint directory and codec checkpoints that are not published as public official weights. The official self-hosted path is a private on-prem container release.
- Integration path: API (free tier) for quick quality reference samples. Do not schedule CSC slot 52 as a public self-host target unless we get official on-prem access or public weights appear.
- URLs: https://inworld.ai/tts
- **Verdict: HIGH via API for quality reference baseline; BLOCKED for public CSC self-hosting right now.**

**27. MaskGCT** (`amphion/MaskGCT`)

- Type: Clone, non-autoregressive
- Params: not disclosed
- License: check Amphion CUHK repo
- Languages: English + limited multilingual
- Highlights: Non-autoregressive masking approach. Appeared in Inferless benchmarks with good latency-quality tradeoff alongside F5/Orpheus. From the Amphion research library (CUHK).
- Integration path: CSC slot 53 or HF Space.
- URLs: https://huggingface.co/amphion/MaskGCT
- **Verdict: LOW-MEDIUM — less differentiated vs F5-TTS already in slot; add if benchmark completeness is a goal.**

**28. KittenTTS** (KittenML / Stellon Labs)

- Type: Plain TTS, ultra-tiny
- Params: <25 MB total
- License: MIT
- Languages: English
- Highlights: Entire model under 25 MB — runnable on any device instantly. YC 2025 backed. Good quality for its size.
- Integration path: Local Mac slot 10. `pip install kittentts` or from `KittenML/KittenTTS`.
- URLs: https://huggingface.co/KittenML/KittenTTS
- **Verdict: LOW-MEDIUM — useful tiny-model baseline comparison point; minimal integration cost.**

---

### Models Explicitly Excluded

| Model | Reason |
|---|---|
| **ChatTTS** | Non-commercial license; deliberately adds noise/compression as anti-misuse measure. Not suitable for naturalness evaluation. |
| **Official VibeVoice inference repo** | Microsoft removed the official GitHub inference repo (Sept 2025). Use the community-mirror prototype path tracked above instead, with research-use-only weight caveats. |
| **NVIDIA Magpie** (self-hosted) | NVIDIA license requires NIM API use only; no freely available self-hosting weights. Incompatible with CSC integration pattern. |
| **Bark (Suno)** | Non-commercial for generated audio; too slow for batch naturalness evaluation. Use only as creative/sound-effects reference. |
| **Supertonic-2** (commercial) | OpenRAIL-M restricts some commercial use — review before publishing generated audio. Fine for internal research. |

---

### Top-10 Priority Implementation List

Ranked by: quality uplift for the naturalness experiment + integration feasibility + licensing cleanliness + differentiation from existing slots 01–30.

| Rank | Slot | Model | Path | Rationale |
|---|---|---|---|---|
| **1** | 31 | Fun-CosyVoice3 (0.5B) | CSC Mahti | Best overall newcomer: 150ms streaming, 9 langs, 0.81% CER, Apache 2.0, production-proven |
| **2** | 32 | Sesame CSM-1B | CSC Mahti | Context-aware dialogue; native transformers 4.52.1+ pipeline; fills unique conversational niche |
| **3** | 33 | Dia 1.6B | CSC Mahti | Unique dialogue/nonverbal [S1]/[S2] tags; Apache 2.0; no equivalent in repo |
| **4** | 34 | IndexTTS-2 | CSC Mahti/Puhti | Lowest WER (0.2418) in benchmarks; duration control; emotion/speaker disentanglement |
| **5** | 05 | Neuphonic NeuTTS Air | Local Mac CPU | Only strong new local-CPU voice-cloning option; GGUF; Apache 2.0; fills Mac expansion gap |
| **6** | 39 | TADA-1B (Hume AI) | CSC or Local | Zero hallucination; 0.09 RTF (11x realtime); 700s long-form; MIT; narration benchmark essential |
| **7** | 36 | Higgs Audio V2 (3B base) | CSC Mahti A100 | Top-trending HF Q3–Q4 2025; best emotion/expressiveness; use 3B-base on Puhti V100 (16GB) |
| **8** | 38 | Maya1 (3B) | CSC Mahti | Voice design + inline emotion tags; Apache 2.0; vLLM-compatible; unique English VD niche |
| **9** | 03 | Chatterbox-Turbo | Local Mac | Replaces broken Chatterbox slot 03; MIT; 350M; lower VRAM; same reference-voice workflow |
| **10** | 35 | MegaTTS3 (0.45B) | CSC Mahti | ByteDance; Apache 2.0; compact 0.45B; good latency for short texts; lightweight CSC slot |

**Runners-up / Wave 2 (slots 36–53):** GLM-TTS (37), Step-Audio-EditX (49), Spark-TTS-0.5B (40 or local), Zonos-v0.1-hybrid (45), FireRedTTS2 (48), Voxtral-4B via HF Space (14), Inworld TTS API.

---

### Integration Notes for Top-10

1. **Voice cloning reference audio:** All voice-cloning models (CosyVoice3, CSM, Dia, IndexTTS-2, NeuTTS Air, Higgs, Maya1, Chatterbox-Turbo, MegaTTS3, GLM-TTS) must use `assets/reference_audio/reference_voice.wav` as the conditioning audio. Never use third-party or copyrighted audio as a reference.

2. **CSC bootstrap scripts:** Follow the pattern in `scripts/mahti/README.md` — create `bootstrap_<model>.sh` (env setup + weight download) and `run_<model>.py` (generation from `experiments/tts_naturalness/inputs/naturalness_prompt.txt`).

3. **Output naming:** `experiments/tts_naturalness/generated/all_samples/31_cosyvoice3.wav`, `32_csm1b.wav`, `33_dia16b.wav`, etc.

4. **Sesame CSM venv:** Shared venv has `transformers 4.46.3` — CSM requires `>=4.52.1`. Use an isolated venv on CSC.

5. **CosyVoice3 on CSC:** Requires conda or isolated venv due to WeTextProcessing. vLLM optional but improves throughput on A100.

6. **Higgs Audio V2 VRAM:** 5.77B full model needs Mahti A100 (80GB). Use `higgs-audio-v2-generation-3B-base` on Puhti V100 (fits in ~16–24 GB).

7. **NeuTTS Air on Mac:** `pip install neutts[all]` installs llama-cpp-python with Metal support automatically. Also run `brew install espeak-ng` for phonemization.

8. **Chatterbox-Turbo isolation:** `chatterbox-turbo` is a separate PyPI package from the broken `chatterbox-tts` in slot 03. Install into a new isolated venv to avoid conflicts.

9. **TADA zero-hallucination property:** Inspect generated audio for the characteristic 1:1 token/frame alignment quality. Compare against Kokoro and Orpheus as naturalness upper bounds.

10. **Cluster sunset planning:** Mahti and Puhti retire in 2026 and are being replaced by Roihu. When wiring new CSC slots, document the SLURM script conventions so they can be ported to Roihu batch system before deprecation.

That lets you keep the final samples while deleting temporary run folders, model cache, or the whole experiment venv later.
