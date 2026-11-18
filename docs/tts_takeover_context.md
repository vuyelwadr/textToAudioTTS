# TTS takeover context

This document captures the approved takeover plan plus the user prompts that drove it, so the repo keeps the execution context instead of leaving it trapped in editor/session state.

## Follow-up status

- `openbmb/VoxCPM2` is fully integrated and already has four synced Puhti outputs in the comparison set.
- `nari-labs/Dia-1.6B-0626` is now integrated and verified on Puhti through the Hugging Face Transformers path. The current local outputs are:
  - `experiments/tts_naturalness/generated/all_samples/33_dia_1_6b_dialogue.wav`
  - `experiments/tts_naturalness/generated/all_samples/77_dia_1_6b_clone.wav`
- The Dia clone path uses the user-provided local source material through the derived 8-second reference clip `haCxgJSFmFc_reference_8s.wav`, because Dia’s own guidance favors roughly 5-10 second prompt audio.
- `sesame/csm-1b` access is restored and verified on Puhti. The current local outputs are:
  - `experiments/tts_naturalness/generated/all_samples/32_csm_1b_plain.wav`
  - `experiments/tts_naturalness/generated/all_samples/76_csm_1b_context_clone.wav`
- The CSM clone path now bypasses TorchCodec by loading prompt audio with `soundfile`, passing waveform arrays directly into the Hugging Face chat template, and casting prompt audio to the model dtype before generation.
- The local comparison folder now has 46 synced outputs, and `experiments/tts_naturalness/generated/all_samples/24_xtts_v2.wav`, `40_spark_tts_0_5b_clone.wav`, `54_parler_tts_large_v1_plain.wav`, and `55_moss_tts_local_transformer_clone.wav` are now present locally.
- The `experiments/tts_naturalness/generated/all_samples/top/all/` comparison batch is now synced locally too, with 42 files across `qwen3_faster`, `voxcpm2`, and `vibevoice` for the 7-input top-3 test set.
- XTTS v2 is now fully completed on Puhti with an isolated Python 3.11 env, scratch-backed `TTS_HOME`, scratch-backed matplotlib cache, and non-interactive Coqui CPML consent so the old Python 3.12 + quota + prompt failures do not recur.
- The key XTTS compatibility fix was pinning `torch==2.3.1` and `torchaudio==2.3.1` after newer PyTorch `weights_only=True` behavior broke Coqui checkpoint loading and the pip-resolved Torch 2.11 / CUDA 13 stack proved too new for the current Puhti driver.
- Sonnet research confirmed `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` as the strongest next unwired slot-31 candidate, but the latest user directive also elevates `parler-tts/parler-tts-large-v1`, the VibeVoice family/docs, `jordand/echo-tts-base`, MOSS-TTS, Higgs Audio, Spark-TTS, and `Kugelaudio/kugelaudio-open` to mandatory shortlist items rather than optional follow-ups.
- The mandatory-model research pass refined that shortlist: direct `parler-tts/parler-tts-large-v1` CSC wiring now targets slot `54_parler_tts_large_v1_plain.wav`; Spark-TTS remains a top practical next add; `OpenMOSS-Team/MOSS-TTS` and `kugelaudio/kugelaudio-0-open` are viable follow-ups; and VibeVoice now has a community-mirror prototype path even though Microsoft removed the official inference repo and the remaining weights stay research-only.
- Parler-TTS Large completed on Puhti as retry job `33929825`, and `54_parler_tts_large_v1_plain.wav` is now synced locally. The first attempt showed that Parler's T5 text encoder needs `attn_implementation="eager"` on the current stack, and that fix is now applied.
- Spark-TTS is now also directly wired on CSC as slot `40_spark_tts_0_5b_clone.wav`, and retry job `33929826` completed successfully. The first attempt hit project-storage quota in the Parler/Spark env area, so Spark now uses the work-root venv path. The current Spark path uses the shared reference audio/transcript and keeps the first run focused on the user's real zero-shot clone use case instead of the control-token UI mode.
- MOSS-TTS Local is now fully integrated through the shared CSC dispatch path. Retry `33930167` hit the old 10G Slurm memory limit, retry `33930238` exposed a torchaudio/torchcodec save-path failure, retry `33930342` succeeded, and redundant rerun `33930416` was cancelled. `55_moss_tts_local_transformer_clone.wav` is now synced locally.
- A second prototype wave now exists in-repo for `kugelaudio/kugelaudio-0-open`, `jordand/echo-tts-base`, `bosonai/higgs-audio-v2-generation-3B-base`, `ResembleAI/chatterbox-turbo`, `mistralai/Voxtral-4B-TTS-2603`, and the VibeVoice family. Those runners are present under `scripts/mahti/`, but they are not yet wired into the shared `submit_model.sh` / `run_model.slurm` path and have not yet produced local comparison WAVs.
- VibeVoice is no longer fully blocked: a community-mirror prototype path now exists, but the official Microsoft repo is still gone and the Hugging Face weights remain research-use-only.

## Approved execution plan

### Objectives

1. Preserve the prior Copilot/user intent in repo docs.
2. Audit the existing local plus CSC TTS coverage so expansion builds on what already exists.
3. Add substantially more high-value TTS and voice-cloning models, including explicit mode coverage where a model supports more than one generation style.
4. Use Mahti/Puhti intelligently so long-lived installs and model state do not keep overfilling scratch.
5. Get more outputs back into `experiments/tts_naturalness/generated/all_samples/` so local listening comparison is materially better than the current limited set.

### Current repo state at takeover

- `.gitignore` already includes `.venv-*/`.
- The repo already has a meaningful TTS comparison scaffold:
  - local/browser baselines
  - Hugging Face Space baselines
  - CSC heavyweight runners under `scripts/mahti/`
- `experiments/tts_naturalness/generated/all_samples/` contains 18 files, so the comparison set is still materially smaller than the user wants.
- `scripts/mahti/` already includes heavyweight integrations for:
  - Orpheus
  - Fish Speech
  - F5-TTS
  - XTTS v2
  - GPT-SoVITS
  - OpenVoice V2
  - OmniVoice
  - Qwen3-TTS
  - Faster Qwen3-TTS
  - Kyutai TTS

### Model and mode plan

The expansion will not be random. It will prioritize strong, currently relevant, practically runnable models, with explicit mode coverage where supported.

#### Existing mode gaps to make explicit

- OmniVoice: `clone`, `design`, `auto`
- Qwen3-TTS: `design`, `custom`, `clone`
- Faster Qwen3-TTS: `design`, `custom`, `clone`
- XTTS v2: explicit clone output
- GPT-SoVITS: explicit clone output
- OpenVoice V2: explicit tone-color conversion output
- Chatterbox: base path first, clone path if runtime stabilizes
- VoxCPM2 when added: `tts`, `voice_design`, `controllable_clone`, `ultimate_clone`

#### First-wave additions

1. `openbmb/VoxCPM2`
   - Explicitly requested by the user.
   - Strong because it supports normal TTS, voice design, controllable cloning, and transcript-backed high-fidelity cloning.
   - Apache-2.0 and documented as practical on modern NVIDIA GPUs.
2. `mistralai/Voxtral-4B-TTS-2603`
   - Explicitly requested by the user.
   - Strong current open-weights entrant with a clean A100-targeted inference story.
3. `Zyphra/Zonos-v0.1-transformer`
   - Important expressive/cloning model missing from the current repo.
   - Linux/NVIDIA-first, so a CSC target rather than a local Mac target.
4. `maya-research/maya1`
   - High-interest missing open-weight candidate.
5. `IndexTeam/IndexTTS-2`
   - Strong modern non-cloning comparison point.
6. `HKUSTAudio/Llasa-3B`
   - Mid-size open TTS model that should be realistic on A100.
7. `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
   - Smaller multilingual modern model.
8. `microsoft/VibeVoice-1.5B`
   - Strong interest and useful modern speech-LM-style comparison.
9. `neuphonic/neutts-air`
   - Lower-cost modern TTS addition.
10. `SparkAudio/Spark-TTS-0.5B`
   - Compact practical baseline.

#### Second-wave additions

11. `microsoft/VibeVoice-Realtime-0.5B`
12. `fishaudio/fish-speech-1.5`
13. `fishaudio/s1-mini`
14. `openbmb/VoxCPM-0.5B`
15. `bosonai/higgs-audio-v2-generation-3B-base`
16. `metavoiceio/metavoice-1B-v0.1`
17. `ResembleAI/chatterbox-turbo`
18. `microsoft/speecht5_tts`
19. `nari-labs/Dia-1.6B`
20. `suno/bark`

### Voice-cloning reference policy

- Do not use copyrighted YouTube downloads for cloning.
- Use the user-provided local source path instead:
  - `/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/outputs/video_downloads/haCxgJSFmFc`
- Current files present there at takeover:
  - `haCxgJSFmFc.mp3`
  - `haCxgJSFmFc.mp4`

### CSC storage plan

1. Inspect real runtime values for `CSC_PROJECT`, `PROJAPPL`, `SCRATCH`, `LOCAL_SCRATCH`, `/projappl/$CSC_PROJECT`, and `/scratch/$CSC_PROJECT/$USER`.
2. Keep long-lived environments, repos, checkpoints, and model installs in project-backed storage.
3. Keep transient cache, pip temp, Hugging Face cache, and temporary runner outputs in scratch or local scratch.
4. If scratch is tight, move long-lived state out of scratch and keep only active working data there.
5. Reuse the existing environment knobs in `scripts/mahti/common.sh` rather than inventing new hard-coded paths.

### Batch execution strategy

The approved execution style is to avoid serial waiting where possible.

1. Use parallelizable waves instead of one-model-at-a-time execution.
2. Submit independent CSC jobs in batches where they do not contend on the exact same environment bootstrap step.
3. Prefer grouped submission for:
   - already wired models that just need smoke/full runs
   - independent clone/design/custom variants that can run as separate jobs
4. Keep local doc/audit/integration work moving while CSC jobs are queued or running.
5. Sync outputs back locally as soon as a meaningful batch finishes rather than waiting for every single model to complete.

## Recovered user messages

These are the prompts recovered from the current session plus local VS Code Insiders/Copilot artifacts relevant to this takeover.

### Current session messages

#### Current session message 1

Source: current CLI session

```text
look at my locl files here pelse add to gitgnore the venvs then stage
  current work. youre takig over fro some other agent use context mcp mine
  suing sonnet subagent the most recent conversaion in
  ~/Documents/Personal/Repos/thesisUndergrad  form copilit. write down all
  my user emssages in docs and epsle take over ensure ALLL th modes i
  mentions are imple,ented even add in more you find online so deploy
  researcher sonnet subaget add AS many promisong models pelse and generate
  the audio sample using my mahti and downlaod all the audios ot
  /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_na
  turalness/generated/all_samples make sure to actually look at e.g puhti
  space e.g mys cratch is full but i know theres plenty of storage
  soemwhere else not sure where it is just look and find it. i think most
  of arhcitectrure is thee just need to expand th emodels too adn e.g
  optimie settings if need be    the agent as ive said sint doing enough i want as many models even ones with e.g voice cloning etc.  and for voice cloning we use this https://youtu.be/haCxgJSFmFc?si=2fpwP0lGGyWnxTmQ e.g downlaod like 10 min of this or so adn use that for voice cloning i liek the voice theer  i want as many itesm as possible to do real comparison right now im not happy at all with aht i see e.g  only 30 here?? /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/generated/all_samples remmber you can adn should 1st add to gitgnore the venv  oh wait nvm i added .venv-*/ so its gone alreay but yeah im not happy with the models it selsected we literally have 1 40gb A100 model its so good we cna be running HUGE models but instead were playing in the diret  i literally just went here https://huggingface.co/models?pipeline_tag=text-to-speech adn see how many there are?? we cn do research see th ebest ones adn use them theres no limitatiosn  e.g why sint this in our list of model?? https://huggingface.co/openbmb/VoxCPM2 we literally dont even test the op ones here https://huggingface.co/models?pipeline_tag=text-to-speech&sort=likes really??? i need you to add at LEAST 20 even more models all these e.g https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 remmber what is adi mix of models nad if a model supports voice cloning use that youtube limk (figure out how to download oyutube items then clip it) remmber you can use the internet. so e.g if it supports voice cloning youll generate multiple version one tih voive cloning the other just the normal one
```

#### Current session message 2

Source: current CLI session

```text
yuse this path foe the voice cloone stuff no need to downlaod form youtube /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/outputs/video_downloads/haCxgJSFmFc
```

#### Current session message 3

Source: current CLI session

```text
maek sure to show me your FULL plan exact paln before executing anything full plan including exac models why you chose them etc
```

#### Current session message 4

Source: current CLI session

```text
greta pelse write this down adn all my suer messages too then begin P:S if oyu cna do batch jobs or whatever do it so we dont keep e.g waiting lots of times for jobs ot start or if we cn also start in parallel idk whatevers ht ebst way to do this without wastig time i just want the outputs ehre loclly so i cna see go ahead soldier
```

#### Current session message 5

Source: current CLI session

```text
oi hoppe yopu also have models like https://huggingface.co/parler-tts/parler-tts-large-v1  https://github.com/huggingface/parler-tts?tab=readme-ov-file adn that oif not add them immediately.  thsi too mandatory https://github.com/microsoft/VibeVoice  this too https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md  bascailly try all these hre https://github.com/microsoft/VibeVoice/tree/main/docs  these too MOSS-TTS

Qwen3-TTS
 
Voxtral-TTS

Fish-AudioTTS

Chatterbox-Turbo  this too Echo-TTS  https://huggingface.co/jordand/echo-tts-base  https://huggingface.co/microsoft/VibeVoice-1.5B  https://github.com/RVC-Boss/GPT-SoVITS  https://huggingface.co/sesame/csm-1b both of thse too https://github.com/kyutai-labs/delayed-streams-modeling/  this is must https://huggingface.co/fishaudio/s2-pro  https://github.com/fishaudio/fish-speech  https://github.com/Kugelaudio/kugelaudio-open  read this and chosoe the top ones https://www.reddit.com/r/LocalLLaMA/comments/1r2hbsl/comment/o4xgonb/?share_id=HnJWpHy7LjfYSYzYqrGG-&utm_content=2&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1  EchoTTS (44 kHz) > Qwen3-TTS (24 kHz) > Higgs-Audio-TTS (24 kHz) > Spark-TTS (16 kHz)

  Spark-tts(very depended of the prompt audio so it may be much worse then others)>Qwen3-TTS(very close and work much better across wide range of prompt audio)>Higgs-Audio-TTS(too much variation)>EchoTTS( still sounds quite natural, but it doesn’t always feel connected to the text)

 https://github.com/kyutai-labs/pocket-tts that too  https://huggingface.co/SWivid/F5-TTS  https://github.com/swivid/f5-tts  write doen ths exact message adn add ALLL thse models every single oen ere to your list an continue to generate pelse
```

#### Current session message 6

Source: current CLI session

```text
you ened to do this too 
I use AI TTS models to generate full audio-books. After testing over 15 different models, I found that most lack the expressiveness or the high voice variation I’m looking for. Some are more expressive but very unstable, while others offer high variation but produce inconsistent timbres.

I eventually found a prompt audio that works well with Spark-TTS, and it sounds great. However, it takes many attempts and different settings to find the right result, and it only applies to one specific voice. The biggest issue that for many zero shot audio files it just not sound well in terms of the style or timbre so this limits you to specific voices.
A week ago, I started fine-tuning a model using Unsloth with LoRA and was surprised by how fast it was to train the model and how little VRAM it used. The biggest challenge was finding good dateset and the right training parameters, but once I figured them out, the results were excellent. The improvement in the speaker's style similarity is outstanding. I also tested 30 minutes of audio files and it still work very well, maybe even lower will work. The quality is much more important. It doesn't seem to make the model more unstable, as it misses about the same amount of words as the original model. It is stable enough that I can use STT to find errors and then regenerate those parts with a more stable model to fix most issues.

I also tried writing a training script for Echo TTS with LoRA (not Unsloth), which is a much larger model with better stability than Spark-TTS. I didn't use it as my primary model because it lacks voice variation and the output is noisier, not significantly, but it’s apparent when using earphones. As it is a larger model, it requires more RAM for training, but it stays around the same level as the inference requirements, so any card with 16GB (or even 12GB) of VRAM should handle it. Fine-tuning significantly improved the speaker's style similarity compared to using zero-shot. It now has high voice variation and expressiveness if the audio dateset have it. It is still noisier than Spark-TTS, but because it is much more stable, it is perfect for regenerating the parts where Spark-TTS has issues.
I will add that spark-tts can be very fast if implementing a SGLang implementation, using high batch, bfloat16 along with deterministic inference you can get around 50x RTFx when using an rtx 5070 ti.
You can use VLLM too but I could not get fully deterministic inference work(even with the same seed) so if you use batch 1 or 50 you will get different result but it still have consistent output for same batch if using "export VLLM_BATCH_INVARIANT=1" . Without export VLLM_BATCH_INVARIANT=1 you will get different result even if use the same batch and seed for identical sentences one after another.
Also as spark-tts output only 16khz audio files so it can sound quite muffled but if using audio super resolution model like Flow-High or LavaSR to up-sample it to 48khz, it sound fantastic. The speed of Flow-High is around RTFx of 45x and LavaSR is around 700x. It really depend of the voice, which produce better output but LavaSR is much faster (Flow-High is quite fast too).  remmer what i said make sure to add all thse too to your list and continue "oi hoppe yopu also have models like                                                    
https://huggingface.co/parler-tts/parler-tts-large-v1                                  
https://github.com/huggingface/parler-tts?tab=readme-ov-file adn that oif not add them 
immediately.  thsi too mandatory https://github.com/microsoft/VibeVoice  this too      
https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md  bascailly try  
all these hre https://github.com/microsoft/VibeVoice/tree/main/docs  these too MOSS-TTS
                                                                                       
Qwen3-TTS                                                                              
                                                                                       
Voxtral-TTS                                                                            
                                                                                       
Fish-AudioTTS                                                                          
                                                                                       
Chatterbox-Turbo  this too Echo-TTS  https://huggingface.co/jordand/echo-tts-base      
https://huggingface.co/microsoft/VibeVoice-1.5B  https://github.com/RVC-Boss/GPT-SoVITS
  https://huggingface.co/sesame/csm-1b both of thse too                                
https://github.com/kyutai-labs/delayed-streams-modeling/  this is must                 
https://huggingface.co/fishaudio/s2-pro  https://github.com/fishaudio/fish-speech      
https://github.com/Kugelaudio/kugelaudio-open  read this and chosoe the top ones       
https://www.reddit.com/r/LocalLLaMA/comments/1r2hbsl/comment/o4xgonb/?share_id=HnJWpHy7
LjfYSYzYqrGG-&utm_content=2&utm_medium=android_app&utm_name=androidcss&utm_source=share
&utm_term=1  EchoTTS (44 kHz) > Qwen3-TTS (24 kHz) > Higgs-Audio-TTS (24 kHz) >        
Spark-TTS (16 kHz)                                                                     
                                                                                       
  Spark-tts(very depended of the prompt audio so it may be much worse then             
others)>Qwen3-TTS(very close and work much better across wide range of prompt          
audio)>Higgs-Audio-TTS(too much variation)>EchoTTS( still sounds quite natural, but it 
doesn’t always feel connected to the text)                                             
                                                                                       
 https://github.com/kyutai-labs/pocket-tts that too                                    
https://huggingface.co/SWivid/F5-TTS  https://github.com/swivid/f5-tts  write doen ths 
exact message adn add ALLL thse models every single oen ere to your list an continue to
 generate pelse "
```

#### Current session message 7

Source: current CLI session

```text
dont use local cpu most fo tie time?? use contecxt mcp deploy e.g sonnet subagent to mine see what previous agent did how it did it how it e.g installed stuff an dall and just do all thise so make sure it all works i want things to run on puhti if possble. unless its really quick e.g small kukoro model can easily run ehre for SHORT generation btu evne kukoro for long audis si slow on cpu. (ill be using the bst one to generate hours of audiobooks e.g 10h plus for a single audiobook kina like thsi already doen my locla kukoro that takes in txt adn chunks it generats audio in chunks etc. so thats final stake ane even thoug kukoro is about 100md generation even 1h+ audiobook locallt takes forever whihc is why i want the stuff generated on gpu so i ahve the code to easily do it when i chosoe the bst one. go pelse greta job)
```

#### Current session message 8

Source: current CLI session

```text
continue pelse great job i already see so much new output
```

#### Current session message 9

Source: current CLI session

```text
· 3.9 KiB)
  /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/source_catalog.md adn /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/docs/tts_takeover_context.md remmebr to actually fo through documentaiton see wahst been donen dresearch ed and contiue
```

#### Current session message 10

Source: current CLI session

```text
did you gegerate all?? you cna now access  sesame/csm-1b its unblocked so make sure whatever you havent generatd is i see /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/generated/all_samples deosnt look like it has all
```

#### Current session message 11

Source: current CLI session

```text
kep going keep generating pesle  still so much more to do
```

#### Current session message 12

Source: current CLI session

```text
Fleet deployed: remmber fro this entire task can and ashould sue subaegnts as youve been very slow all remaining work doen with subagents help pelse remmber request was this "oi hoppe yopu also have models like https://huggingface.co/parler-tts/parler-tts-large-v1 https://github.com/huggingface/parler-tts?tab=readme-ov-file adn that oif not add them immediately. thsi too mandatory https://github.com/microsoft/VibeVoice this too https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-tts.md bascailly try all these hre https://github.com/microsoft/VibeVoice/tree/main/docs these too MOSS-TTS Qwen3-TTS Voxtral-TTS Fish-AudioTTS Chatterbox-Turbo this too Echo-TTS https://huggingface.co/jordand/echo-tts-base https://huggingface.co/microsoft/VibeVoice-1.5B https://github.com/RVC-Boss/GPT-SoVITS https://huggingface.co/sesame/csm-1b both of thse too https://github.com/kyutai-labs/delayed-streams-modeling/ this is must https://huggingface.co/fishaudio/s2-pro https://github.com/fishaudio/fish-speech https://github.com/Kugelaudio/kugelaudio-open read this and chosoe the top ones https://www.reddit.com/r/LocalLLaMA/comments/1r2hbsl/comment/o4xgonb/?share_id=HnJWpHy7 LjfYSYzYqrGG-&utm_content=2&utm_medium=android_app&utm_name=androidcss&utm_source=share &utm_term=1 EchoTTS (44 kHz) > Qwen3-TTS (24 kHz) > Higgs-Audio-TTS (24 kHz) > Spark-TTS (16 kHz) Spark-tts(very depended of the prompt audio so it may be much worse then others)>Qwen3-TTS(very close and work much better across wide range of prompt audio)>Higgs-Audio-TTS(too much variation)>EchoTTS( still sounds quite natural, but it doesn’t always feel connected to the text) https://github.com/kyutai-labs/pocket-tts that too https://huggingface.co/SWivid/F5-TTS https://github.com/swivid/f5-tts write doen ths exact message adn add ALLL thse models every single oen ere to your list an continue to generate pelse " including all other models ALL h rest in your plan so add and run
```

### Recovered local Copilot / VS Code Insiders messages

#### Recovered message 1

Source:
- `~/Library/Application Support/Code - Insiders/User/globalStorage/github.copilot-chat/copilotCli/copilotcli.session.metadata.json`
- field: `firstUserMessage`

Note: this recovered message matches the main current-session kickoff prompt and is preserved here because it was explicitly requested that recovered user messages be written down.

```text
look at my locl files here pelse add to gitgnore the venvs then stage
  current work. youre takig over fro some other agent use context mcp mine
  suing sonnet subagent the most recent conversaion in
  ~/Documents/Personal/Repos/thesisUndergrad  form copilit. write down all
  my user emssages in docs and epsle take over ensure ALLL th modes i
  mentions are imple,ented even add in more you find online so deploy
  researcher sonnet subaget add AS many promisong models pelse and generate
  the audio sample using my mahti and downlaod all the audios ot
  /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_na
  turalness/generated/all_samples make sure to actually look at e.g puhti
  space e.g mys cratch is full but i know theres plenty of storage
  soemwhere else not sure where it is just look and find it. i think most
  of arhcitectrure is thee just need to expand th emodels too adn e.g
  optimie settings if need be    the agent as ive said sint doing enough i want as many models even ones with e.g voice cloning etc.  and for voice cloning we use this https://youtu.be/haCxgJSFmFc?si=2fpwP0lGGyWnxTmQ e.g downlaod like 10 min of this or so adn use that for voice cloning i liek the voice theer  i want as many itesm as possible to do real comparison right now im not happy at all with aht i see e.g  only 30 here?? /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/generated/all_samples remmber you can adn should 1st add to gitgnore the venv  oh wait nvm i added .venv-*/ so its gone alreay but yeah im not happy with the models it selsected we literally have 1 40gb A100 model its so good we cna be running HUGE models but instead were playing in the diret  i literally just went here https://huggingface.co/models?pipeline_tag=text-to-speech adn see how many there are?? we cn do research see th ebest ones adn use them theres no limitatiosn  e.g why sint this in our list of model?? https://huggingface.co/openbmb/VoxCPM2 we literally dont even test the op ones here https://huggingface.co/models?pipeline_tag=text-to-speech&sort=likes really??? i need you to add at LEAST 20 even more models all these e.g https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 remmber what is adi mix of models nad if a model supports voice cloning use that youtube limk (figure out how to download oyutube items then clip it) remmber you can use the internet. so e.g if it supports voice cloning youll generate multiple version one tih voive cloning the other just the normal one
```

#### Recovered message 2

Source:
- `~/Library/Application Support/Code - Insiders/User/History/-4c408d8c/entries.json`
- `resource`: `file:///Users/ruwodda/Documents/Personal/Repos/thesisUndergrad/mahti_findings.md`

```text
pelse explore my ssh mahti    adn  document findings ot /Users/ruwodda/Documents/Personal/Repos/thesisUndergrad what fiels are there that belong to em my usr traverse the perosnal adn the larger storage what project was bring done what wer eth eresults what scripts can you see how correct are the scripts etc adn how did the experiemnts go and why??
```

#### Recovered message 3

Source:
- `~/Library/Application Support/Code - Insiders/User/History/77797429/entries.json`
- `resource`: `file:///Users/ruwodda/Documents/Personal/Repos/thesisUndergrad/mahti_exploration.md`

```text
pelse explore my ssh mahti    adn  document findings ot /Users/ruwodda/Documents/Personal/Repos/thesisUndergrad what fiels are there that belong to em my usr traverse the perosnal adn the larger storage what project was bring done what wer eth eresults what scripts can you see how correct are the scripts etc adn how did the experiemnts go and why?? remmebr ssh into mahti to see
```
