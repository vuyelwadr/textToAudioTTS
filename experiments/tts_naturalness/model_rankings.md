# Model Rankings

This ranking is about model families and source quality, not individual voices.

---

## Evidence-Grounded Quality Rankings (2025)

Synthesised from multiple independent benchmarks and evaluations. Evidence sources are cited in the table and summarised at the end of this section.

**Evidence key:**
- **HFR** = Human Fooling Rate on Expresso conversational speech (Interspeech 2025 study; human reference 70.68%)
- **MUSHRA** = full-ref listening score from same study (higher = more natural-sounding)
- **Inferless** = 12-model multi-criteria evaluation (quality + controllability, July 2025)
- **AA-ELO** = Artificial Analysis Speech Arena crowdsourced ELO
- **SEED-MOS** = MOS on SEED-TTS multilingual benchmark (CosyVoice 2 paper)

### Tier 1 — Strongest External Quality Evidence

| # | Model | Params | License | Key Evidence | Repo Status |
|---|-------|--------|---------|-------------|-------------|
| 1 | **F5-TTS** | ~335 M | Apache 2.0 | HFR **50.26%** (best open-source, Interspeech 2025); MUSHRA 70.75; best balanced quality+controllability (Inferless) | ✅ Slot 23 |
| 2 | **CosyVoice 2 / Fun-CosyVoice3** | 0.5 B | Apache 2.0 | MOS **5.53** on SEED-TTS (best open-source); lowest CER; 150 ms streaming latency | ❌ Slot 31 planned |
| 3 | **Sesame CSM-1B** | 1 B | Apache 2.0 | Best balanced quality+controllability alongside F5-TTS (Inferless); near-commercial MOS per CodeSOTA; native HF Transformers pipeline | ❌ Slot 32 planned |
| 4 | **Kokoro 82M** | 82 M | Apache 2.0 | #1 open-source on TTS Arena (HF Pendrokar space); AA-ELO **~1058** (best open-source on Artificial Analysis); top Inferless quality tier | ✅ Slots 01, 15 |
| 5 | **Orpheus TTS 3B** | 3 B | Apache 2.0 | Top Inferless quality tier; CodeSOTA near-commercial MOS claim (4.7); LLM-based expressiveness | ✅ Slot 21 |

### Tier 2 — Strong Benchmark Support

| # | Model | Params | License | Key Evidence | Repo Status |
|---|-------|--------|---------|-------------|-------------|
| 6 | **XTTS v2** | ~500 M | MPL 2.0 | HFR 41.80%, MUSHRA **76.58** (highest MUSHRA of all open-source, Interspeech 2025); most-downloaded on BentoML | ✅ Slot 24 (**blocked**: Python 3.12 mismatch on Puhti) |
| 7 | **StyleTTS2** | ~200 M | MIT | HFR 38.60%, MUSHRA 71.21 (Interspeech 2025); academic standard for open narration quality benchmarks | ❌ **Missing** (high-priority gap) |
| 8 | **GPT-SoVITS v2** | ~200 M | MIT | HFR 44.61%, MUSHRA 68.21 (Interspeech 2025); widely used multilingual zero-shot cloning | ✅ Slot 25 |
| 9 | **Spark-TTS-0.5B** | 500 M | Apache 2.0 | Named alongside Kokoro / CSM-1B / F5-TTS in Inferless top quality tier; Qwen2.5 backbone; dual local/CSC path | ❌ Slot 40 planned |
| 10 | **Llasa-3B** | 3 B | Apache 2.0 | Top Inferless quality tier (alongside CSM/F5-TTS/Kokoro); scalable (1 B / 3 B / 8 B variants available) | ❌ **Missing** |

### Tier 3 — High Community Signal + Editorial Consensus

| # | Model | Params | License | Key Evidence | Repo Status |
|---|-------|--------|---------|-------------|-------------|
| 11 | **Dia 1.6B** | 1.6 B | Apache 2.0 | Top-3 HF community likes (non-Kokoro open models); unique `[S1]/[S2]` nonverbal tags; Modal top-5 pick | ❌ Slot 33 planned |
| 12 | **Higgs Audio V2** | ~3 B | Apache 2.0 | #1 trending HF at research time; strong emotion+dialogue naturalness; frequent editorial top-5 | ❌ Slot 36/38 planned |
| 13 | **Fish Speech S2 Pro** | ~1 B | Fish Audio Research License | High HF likes; production-grade zero-shot voice cloning; CodeSOTA top-pick for cloning; non-commercial caveat | ✅ Slot 22 |
| 14 | **IndexTTS-2** | ~1 B | Apache 2.0 | WER **0.2418** (benchmark-leading intelligibility); emotion/speaker disentanglement | ❌ Slot 34 planned |
| 15 | **Chatterbox** | 250 M | MIT | Top-5 editorial pick (Modal, BentoML); production-grade low-latency; Resemble AI | ✅ In repo |
| 16 | **Qwen3-TTS** | varies | Apache 2.0 | Alibaba flagship multilingual; high HF likes; voice-design + clone + custom modes | ✅ Slots 28, 29 |
| 17 | **Zonos-v0.1-hybrid** | ~1 B | Zyphra Permissive | **Best controllability** in Inferless evaluation (unique distinction); 44 kHz output | ❌ Slot 45 planned |
| 18 | **Voxtral-4B-TTS** | 4 B | Mistral Research | High HF likes; Mistral ecosystem; gated weights (HF Space path low-friction) | ❌ Slot 14 planned |
| 19 | **MetaVoice-1B** | 1 B | Apache 2.0 | High HF community likes; strong zero-shot American English; Meta research backing | ❌ **Missing** |
| 20 | **OpenVoice v2** | ~350 M | MIT | Well-documented cross-lingual voice cloning; MyShell.ai; strong speaker-transfer | ✅ Slot 26 |

### Tier 4 — Established Baselines / Niche Strengths

| # | Model | Params | License | Key Evidence | Repo Status |
|---|-------|--------|---------|-------------|-------------|
| 21 | **VoiceCraft** | ~330 M | MIT | HFR 30.52%, MUSHRA 49.02 (Interspeech 2025); specialises in in-domain speech editing | ❌ **Missing** |
| 22 | **OmniVoice** | ~500 M | Apache 2.0 | Expressive multi-mode (clone/auto/design); in repo and producing output | ✅ Slot 27 |
| 23 | **MaskGCT** | ~800 M | Apache 2.0 | Academic benchmark completeness; Amphion framework; F5-TTS-class non-autoregressive | ❌ Slot 53 planned |
| 24 | **Fish Speech 1.5** | ~1 B | CC BY-NC-SA | High HF likes; updated Fish Audio stack; non-commercial — check before publishing | ❌ Missing |
| 25 | **Kyutai TTS** | ~1 B | CC BY 4.0 | Streaming-first unique capability; multi-language; in repo | ✅ Slot 30 |

### Best 5–8 Next Additions

*Ranked by: external quality evidence × license freedom × CSC feasibility*

| Priority | Model | Rationale |
|----------|-------|-----------|
| 🥇 1 | **CosyVoice 2 / Fun-CosyVoice3** | Highest open-source MOS (5.53 SEED-TTS); Apache 2.0; already slot 31 — wire first |
| 🥈 2 | **Sesame CSM-1B** | Inferless best-balanced; native HF pipeline (minimal bootstrap friction); slot 32 |
| 🥉 3 | **StyleTTS2** | Only model with Interspeech 2025 HFR data **not yet in repo** — closes a real benchmark gap; MIT |
| 4 | **Spark-TTS-0.5B** | Inferless top-tier; Apache 2.0; reuses existing Qwen env structure; slot 40 |
| 5 | **Dia 1.6B** | High community validation + unique dialogue/nonverbal capability; Apache 2.0; slot 33 |
| 6 | **IndexTTS-2** | WER 0.2418 benchmark leader; Apache 2.0; slot 34 |
| 7 | **Higgs Audio V2** | Fills the repo's emotional-naturalness gap; Apache 2.0; slot 36/38 |
| 8 | **Llasa-3B** | Inferless top-tier quality evidence; Apache 2.0; not yet planned in slots — add next |

### Evidence Source Summary

| Source | What it measures | Models directly benchmarked |
|--------|-----------------|----------------------------|
| Interspeech 2025 HFR study | Human Fooling Rate + MUSHRA on Expresso conversational speech | F5-TTS, GPT-SoVITS, XTTS, StyleTTS2, VoiceCraft |
| Inferless 12-model eval (Jul 2025) | Multi-criteria quality + controllability | Kokoro, CSM-1B, Spark-TTS, Orpheus-3B, F5-TTS, Llasa-3B, Zonos, Chatterbox |
| Artificial Analysis Speech Arena ELO | Crowdsourced human A/B preference listening test | Kokoro 82M v1.0 (~1058, best open-source) |
| SEED-TTS benchmark (CosyVoice 2 paper) | MOS + CER on multilingual speech | CosyVoice 2 (5.53 MOS, lowest CER of open models) |
| TTSDS2 (ICLR 2026 oral) | Distribution-based metric; strongest Spearman corr. with human judgment across domains | 20 open-source systems across 14 languages |
| CodeSOTA MOS survey 2025 | Broad MOS timeline; 4.7 MOS for leading open-source models | Orpheus, CSM, F5-TTS |
| HF community likes | Adoption + reputation proxy | All models |
| Modal / BentoML / Inferless / portalzine editorial | Curated practitioner consensus | Higgs V2, Kokoro, Dia, CSM, Chatterbox, Orpheus |

---

## Quality-First Open-Source Top 20

This is the current repo-backed shortlist for the strongest open-source or open-weight TTS families for **audio quality / naturalness**, with a bias toward audiobook-style narration and models that are practical on CSC. It folds together the existing Puhti results in this repo plus the expansion research recorded in `source_catalog.md`.

1. `Fish Speech S2 Pro` — `[wired]` strongest open-weight quality signal in the current notes; high upside for expressive long-form narration.
2. `Qwen3-TTS` — `[wired]` one of the best control surfaces here because the family spans clone, custom-voice, and voice-design checkpoints.
3. `VoxCPM2` — `[wired]` unusually strong because it covers plain TTS, voice design, controllable cloning, and transcript-backed cloning in one stack.
4. `F5-TTS` — `[wired]` one of the clearest long-form voice-cloning candidates when speaker consistency matters more than pure zero-shot convenience.
5. `Higgs Audio V2 / V2.5` — `[next]` current high-interest heavyweight from the repo research; strong emotion/dialogue naturalness signal.
6. `Fun-CosyVoice3` — `[next]` best value add from the 2025-2026 research pass: multilingual, fast, and still quality-focused.
7. `IndexTTS-2` — `[next]` compelling on quality metrics and a strong candidate for clean narration without needing a cloning-first workflow.
8. `OmniVoice` — `[wired]` high-potential expressive model with separate clone, auto, and design modes already paying off in the local output set.
9. `Orpheus-3B` — `[wired]` one of the strongest expressive open narration candidates already in the CSC stack.
10. `Maya1` — `[next]` voice-design-oriented 3B model that looks especially relevant for narration tone control.
11. `Sesame CSM-1B` — `[next]` very strong next-wave candidate for context-aware speech quality, especially where dialogue matters.
12. `Voxtral-4B-TTS` — `[next]` obvious heavyweight addition from the repo shortlist; high interest and likely strong quality ceiling.
13. `Zonos-v0.1` — `[next]` important expressive/cloning family with much richer conditioning than older baseline stacks.
14. `GLM-TTS` — `[next]` promising new expressive model family from the expansion notes, worth testing against the more established clone stacks.
15. `FireRedTTS2` — `[next]` strong long-form multi-speaker candidate with streaming-friendly design.
16. `Neuphonic NeuTTS Air` — `[next]` smaller but still serious quality candidate, attractive because it broadens the shortlist without demanding the heaviest setup.
17. `GPT-SoVITS` — `[wired]` still one of the most important open cloning stacks to benchmark directly because it remains so widely used in practice.
18. `XTTS v2` — `[wired, blocked on Puhti Python]` older than the newest leaders, but still too established in multilingual cloning to ignore.
19. `OpenVoice V2` — `[wired]` more cloning-oriented than narration-first, but still worth keeping in the quality shootout because of its strong speaker-transfer path.
20. `Kokoro 82M` — `[wired, local]` not the absolute quality ceiling, but still one of the strongest tiny open local narration baselines and valuable as the practical floor.

### Caveats that materially affect the shortlist

- `Fish Speech` uses the Fish Audio Research License rather than a plain permissive software license.
- `Voxtral-4B-TTS` is strategically important, but the current repo notes still treat it as a higher-friction target than the already-wired models.
- `Zonos-v0.1` uses a custom permissive-style license that still deserves a fresh read before publishing large output sets.
- `XTTS v2` remains blocked on Puhti until the Python 3.12 mismatch is fixed or the stack is moved to an older PyTorch/Python combo.

## Priority A

1. `Kokoro 82M`
   Local, already integrated, Apple Silicon friendly, and consistently strong for a lightweight open model.
2. `Chatterbox`
   Strong naturalness reputation, easy to compare locally once the bundled script is pointed at the current repo layout.
3. `ElevenLabs-class web voices`
   Still the naturalness reference point, but access friction is higher than the local models here.
4. `Fish Speech`
   High upside and strong reputation, but the repo still needs a clean editable clone plus shared-env validation.
5. `text-to-speech.online / Edge-style web voices`
   Good free baseline and already partially automated in this repo, but quality ceiling is below the top local targets.

## Priority B

6. `Bark / Coqui TTS`
   Worth benchmarking as an open baseline, but slower and usually less natural than the top-ranked sources.
7. `Cartesia`
   Excellent reputation, but sign-up friction and closed API access make it a later integration.
8. `Google Speech Gen`
   High quality, but limited free quota and account friction reduce experiment value.
9. `Kyutai TTS`
   Interesting emerging model family, but weaker proof of fit for this repo than the sources above.

## Priority C

- `FakeYou`
  Better for character impressions than natural narration.
- `Tortoise TTS`
  Historically important, but too slow and no longer competitive for a clean naturalness shootout.
- `Moe TTS`, `Speechma`, `VanillaVoice`, `LOVO`, and similar web-only catalogs
  Too much signup or catalog noise for too little model-quality signal.

## Why This Order

- Preference goes to sources that are both natural sounding and reproducible on this Mac.
- Local MPS-capable paths outrank browser-only sources when quality is close.
- Free web tools are kept as baselines, not as the center of the experiment.

---

## 2025–2026 Expansion Rankings

*New candidates from expansion research (July 2025). Slots 31+ not yet wired.*

### Priority A — Implement Next

1. `Fun-CosyVoice3` (0.5B, Apache 2.0) — CSC slot 31. Best new value: 9 langs, 0.81% CER, 150ms streaming.
2. `Sesame CSM-1B` (Apache 2.0) — CSC slot 32. Context-aware dialogue; native HF Transformers pipeline.
3. `Dia 1.6B` (Apache 2.0) — CSC slot 33. Unique [S1]/[S2] dialogue + nonverbal sound tags.
4. `IndexTTS-2` (Apache 2.0) — CSC slot 34. Lowest WER (0.2418) in benchmarks; emotion/speaker disentanglement.
5. `Neuphonic NeuTTS Air` (0.5B, Apache 2.0) — Local Mac CPU slot 05. GGUF; voice cloning on laptop CPU.
6. `Hume AI TADA-1B` (MIT) — CSC slot 39 or Local. Zero-hallucination; 11x realtime; 700s long-form.
7. `Maya1` (3B, Apache 2.0) — CSC slot 38. Voice design + inline emotion tags; unique English VD niche.
8. `Higgs Audio V2/V2.5` (Apache 2.0) — CSC Mahti A100 slot 36. #1 trending HF; best emotion + dialogue naturalness.
9. `Chatterbox-Turbo` (350M, MIT) — Local Mac slot 03 replacement. Fixes broken Chatterbox; lower VRAM.
10. `MegaTTS3` (0.45B, Apache 2.0) — CSC slot 35. ByteDance; compact; good latency for short texts.

### Priority B — Next Wave

11. `GLM-TTS` (Apache 2.0) — CSC slot 37. RL/GRPO-optimized expressiveness; 13.6% CER reduction.
12. `Step-Audio-EditX` (3B, Apache 2.0) — CSC slot 49. Iterative speech editing; 14 emotion tags; unique capability.
13. `Spark-TTS-0.5B` (Apache 2.0) — Local Mac or CSC slot 40. Dual local/CSC flexibility; Qwen2.5 backbone.
14. `Zonos-v0.1-hybrid` (Zyphra Permissive) — CSC slot 45. Rich conditioning; 44 kHz; check license terms.
15. `Inworld TTS-1` (open training code; hosted API / private on-prem) — API reference slot only for now, not CSC slot 52. #1 ELO on Artificial Analysis, but public self-host weights are not available.
16. `FireRedTTS2` (Apache 2.0) — CSC slot 48. Long-form multi-speaker streaming; 7 languages.
17. `Voxtral-4B-TTS` (Mistral Research) — HF Space slot 14. Low-friction HF Space path; gated weights.
18. `Llasa-3B` (check license) — CSC slot 41. Competitive benchmarks; 1B/3B/8B variants.
19. `VibeVoice-Realtime-0.5B` (MIT weights) — Slot 42. `CAVEAT:` code repo removed by Microsoft Sept 2025 — verify before wiring.
20. `MeloTTS` (MIT) — Local Mac slot 08. CPU multilingual baseline; 6 languages; trivial install.

### Priority C — Monitoring / Low Marginal Value

- `Supertonic-2` (OpenRAIL-M, 66M) — Local Mac slot 09. Tiny baseline; no voice cloning; check license.
- `E2-TTS` (SWivid, ~335M) — CSC slot 44. Reuses F5-TTS env; limited independent signal.
- `OuteTTS 1.0-1B` (Apache 2.0) — Local GGUF or CSC slot 46. Too slow for batch runs (>4 min/200 words).
- `VoxCPM-0.5B` (Apache 2.0) — CSC slot 47. Limited English evidence; primarily Chinese-focused.
- `MaskGCT` (Amphion) — CSC slot 53. Add for benchmark completeness; overlaps with F5-TTS.
- `VibeVoice-1.5B` (MIT weights) — CSC slot 43. Multi-speaker long-form; same code-removal caveat as Realtime.
- `KittenTTS` (MIT, <25MB) — Local Mac slot 10. Ultra-tiny baseline; minimal integration cost.
- `NVIDIA Magpie TTS` — NIM API only; NVIDIA license prevents CSC self-hosting.
- `Step-Audio-TTS-3B` (Apache 2.0) — CSC slot 50. Supplement to EditX; 14 emotions; vLLM required.

### Excluded (Do Not Add)

- `ChatTTS` — Non-commercial; deliberately degraded audio quality. Not usable for naturalness evaluation.
- `VibeVoice` (inference code) — Microsoft removed code repo Sept 2025. Weights-only on HF; no supported inference path.
- `Bark` (Suno) — Non-commercial for generated audio; too slow for batch runs.

### Licensing Caveats Summary

| Model | License concern |
|---|---|
| Zonos-v0.1-hybrid | Zyphra Permissive License — verify commercial terms before publishing audio |
| Supertonic-2 | OpenRAIL-M — review commercial use restrictions |
| NVIDIA Magpie | NVIDIA Open Model License — NIM API only; no free self-hosting |
| GLM-TTS | Open source — confirm commercial terms in LICENSE before use |
| VibeVoice (any) | MIT weights but inference code removed from GitHub — blocked |
| Fish Speech (slots 12, 22) | Fish Audio Research License — already in repo; terms documented |
