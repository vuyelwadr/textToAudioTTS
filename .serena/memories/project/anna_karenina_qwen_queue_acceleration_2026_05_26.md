# Anna Karenina Qwen queue acceleration check - 2026-05-26

User asked if the estimate is too far and whether the run could be made faster by CPU or fewer GPUs.

Findings:
- Current full job `6720307` already requests exactly one A100 GPU: `--gres=gpu:a100:1,nvme:900`. The queue delay is not caused by a multi-GPU request.
- `sprio -j 6720307` showed priority `307`; pending reason is `Priority`.
- `sbatch --test-only` against the existing script showed no earlier estimate for smaller walltime/resource shapes. Same one-GPU request with `36h`, `24h`, `20h`, `18h`, `12h`, `8h`, and `6h` all estimated `2026-05-28T04:59:44`. Dropping NVMe and reducing to `16` CPUs/`64G` did not improve the estimate.
- Multi-GPU is not a better queue option: 2-GPU tests estimated later (`2026-05-28T22:19:24`), and 4-GPU tests failed with `QOSMaxGRESPerJob`.
- Runtime estimate from smoke: two chunks generated in `84.81s`, avg `42.4s/chunk`; for `1139` chunks this is about `13.4h`. The chunk plan estimates `2329.5` minutes audio and smoke RTF `~0.40x`, implying about `15.4h` TTS generation. Audit/video overhead still applies, but final MP4 is CPU-submitted after the GPU job.
- CPU TTS is not practical: even conservative `5x-20x` CPU slowdown implies roughly `2.8-11.2` days for TTS only, before audit/video overhead, and it would not use the GPU-optimized Qwen CUDA graph path.

Decision recorded in `tasks/anna_karenina_qwen_video_20260525/verification.md` and synced to Mahti. Local/remote SHA-256 after sync: `dcf935a33865edae86c8eb602df734719d6bfcd3703a88004283d27a4c5b3b90`.

Recommendation: keep the existing one-GPU job. Changing walltime/resources does not improve tested queue start; CPU is impractical; 2 GPUs starts later; 4 GPUs is disallowed.