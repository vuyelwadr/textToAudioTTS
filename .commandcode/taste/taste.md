# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# workflow
- Do not edit local repo files unless absolutely necessary, and do not revert anything since you are not alone in the codebase. Confidence: 0.70
- For CPU-only video/audio/post-processing jobs, request 4-8 cores. Confidence: 0.70
- Execute audiobook/TTS generation tasks immediately rather than preparing scripts and asking for permission, including SSHing directly into clusters to submit jobs instead of preparing handoff documentation. Confidence: 0.95
- Find existing reference voice samples by checking reference audio paths in previous job configurations and slurm scripts. Confidence: 0.70
- For audio/TTS jobs on remote clusters, place scripts directly on the cluster alongside other job files rather than creating them only locally. Confidence: 0.70

