#!/usr/bin/env python3
"""
Comprehensive Audio Quality Analysis for TTS Naturalness Experiment
Analyzes all_samples folder for problematic audio files
"""

import subprocess
import json
import os
from pathlib import Path


def run_ffprobe(filepath):
    """Get audio file metadata"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate,size",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None


def get_volume_stats(filepath):
    """Get volume statistics"""
    cmd = ["ffmpeg", "-i", filepath, "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    stats = {}
    for line in output.split("\n"):
        if "mean_volume:" in line:
            stats["mean_volume"] = float(
                line.split("mean_volume:")[1].split("dB")[0].strip()
            )
        if "max_volume:" in line:
            stats["max_volume"] = float(
                line.split("max_volume:")[1].split("dB")[0].strip()
            )
    return stats


def get_silence_stats(filepath):
    """Get silence detection stats"""
    cmd = [
        "ffmpeg",
        "-i",
        filepath,
        "-af",
        "silencedetect=noise=-50dB:d=0.5",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    silence_periods = []
    current_start = None

    for line in output.split("\n"):
        if "silence_start:" in line:
            current_start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and current_start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            duration = end - current_start
            silence_periods.append(
                {"start": current_start, "end": end, "duration": duration}
            )
            current_start = None

    return silence_periods


def analyze_file(filepath):
    """Comprehensive analysis of a single audio file"""
    filename = os.path.basename(filepath)

    # Basic metadata
    probe = run_ffprobe(filepath)
    if not probe:
        return {"filename": filename, "error": "Failed to probe file"}

    format_info = probe.get("format", {})
    streams = probe.get("streams", [{}])[0]

    duration = float(format_info.get("duration", 0))
    size = int(format_info.get("size", 0))

    # Volume stats
    volume = get_volume_stats(filepath)

    # Silence stats
    silence_periods = get_silence_stats(filepath)
    total_silence = sum(p["duration"] for p in silence_periods)
    silence_ratio = total_silence / duration if duration > 0 else 0

    return {
        "filename": filename,
        "duration": duration,
        "size_bytes": size,
        "mean_volume_db": volume.get("mean_volume", -100),
        "max_volume_db": volume.get("max_volume", -100),
        "total_silence_sec": total_silence,
        "silence_ratio": silence_ratio,
        "silence_periods": len(silence_periods),
        "longest_silence": max([p["duration"] for p in silence_periods], default=0),
    }


def classify_quality(analysis):
    """Classify file quality based on analysis"""
    issues = []
    severity = "OK"

    # Check for very low volume (likely inaudible/poor quality)
    if analysis["mean_volume_db"] < -30:
        issues.append(
            f"VERY LOW VOLUME: mean={analysis['mean_volume_db']:.1f}dB (inaudible/whisper)"
        )
        severity = "RUBBISH"
    elif analysis["mean_volume_db"] < -25:
        issues.append(f"LOW VOLUME: mean={analysis['mean_volume_db']:.1f}dB")
        if severity == "OK":
            severity = "POOR"

    # Check for excessive silence
    if analysis["silence_ratio"] > 0.5:
        issues.append(
            f"EXCESSIVE SILENCE: {analysis['silence_ratio'] * 100:.1f}% of file is silent"
        )
        severity = "RUBBISH"
    elif analysis["silence_ratio"] > 0.3:
        issues.append(
            f"HIGH SILENCE: {analysis['silence_ratio'] * 100:.1f}% of file is silent"
        )
        if severity == "OK":
            severity = "POOR"

    # Check for very long trailing silence
    if analysis["longest_silence"] > 3:
        issues.append(f"LONG SILENCE PERIOD: {analysis['longest_silence']:.1f}s")
        if severity == "OK":
            severity = "POOR"

    # Check for abnormally long duration (possible loop/bug)
    if analysis["duration"] > 20:
        issues.append(f"ABNORMAL DURATION: {analysis['duration']:.1f}s (possible loop)")
        if severity == "OK":
            severity = "WARNING"

    # Check for very short files
    if analysis["duration"] < 5:
        issues.append(f"VERY SHORT: {analysis['duration']:.1f}s")
        if severity == "OK":
            severity = "WARNING"

    return severity, issues


def main():
    samples_dir = "/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/experiments/tts_naturalness/generated/all_samples"

    # Get all audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3"]:
        audio_files.extend(Path(samples_dir).glob(ext))

    audio_files.sort()

    results = {"RUBBISH": [], "POOR": [], "WARNING": [], "OK": []}

    print("=" * 100)
    print("TTS NATURALNESS EXPERIMENT - AUDIO QUALITY ANALYSIS")
    print("=" * 100)
    print(f"\nAnalyzing {len(audio_files)} audio files...\n")

    for filepath in audio_files:
        analysis = analyze_file(str(filepath))
        severity, issues = classify_quality(analysis)

        result = {"analysis": analysis, "issues": issues}
        results[severity].append(result)

        # Print progress
        print(
            f"[{severity:8}] {analysis['filename'][:50]:<50} "
            f"Dur:{analysis['duration']:5.1f}s "
            f"Vol:{analysis['mean_volume_db']:6.1f}dB "
            f"Sil:{analysis['silence_ratio'] * 100:5.1f}%"
        )

    # Print detailed report
    print("\n" + "=" * 100)
    print("DETAILED FINDINGS")
    print("=" * 100)

    if results["RUBBISH"]:
        print("\n🔴 RUBBISH FILES ( unusable / pure garbage ):")
        print("-" * 100)
        for r in results["RUBBISH"]:
            a = r["analysis"]
            print(f"\n  📁 {a['filename']}")
            print(
                f"     Duration: {a['duration']:.2f}s | Mean Volume: {a['mean_volume_db']:.1f}dB | "
                f"Max Volume: {a['max_volume_db']:.1f}dB"
            )
            print(
                f"     Silence: {a['total_silence_sec']:.2f}s ({a['silence_ratio'] * 100:.1f}%) | "
                f"Periods: {a['silence_periods']} | Longest: {a['longest_silence']:.2f}s"
            )
            for issue in r["issues"]:
                print(f"     ⚠️  {issue}")

    if results["POOR"]:
        print("\n🟠 POOR QUALITY FILES ( usable but degraded ):")
        print("-" * 100)
        for r in results["POOR"]:
            a = r["analysis"]
            print(f"\n  📁 {a['filename']}")
            print(
                f"     Duration: {a['duration']:.2f}s | Mean Volume: {a['mean_volume_db']:.1f}dB"
            )
            for issue in r["issues"]:
                print(f"     ⚠️  {issue}")

    if results["WARNING"]:
        print("\n🟡 WARNING FILES ( unusual but may be intentional ):")
        print("-" * 100)
        for r in results["WARNING"]:
            a = r["analysis"]
            print(f"\n  📁 {a['filename']}")
            for issue in r["issues"]:
                print(f"     ⚠️  {issue}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"\n  Total Files Analyzed: {len(audio_files)}")
    print(f"  🔴 RUBBISH (unusable):    {len(results['RUBBISH'])} files")
    print(f"  🟠 POOR (degraded):       {len(results['POOR'])} files")
    print(f"  🟡 WARNING (unusual):     {len(results['WARNING'])} files")
    print(f"  🟢 OK (good quality):     {len(results['OK'])} files")

    if results["RUBBISH"]:
        print("\n  \n📋 LIST OF RUBBISH FILES TO EXCLUDE FROM EVALUATION:")
        print("  " + "-" * 80)
        for r in results["RUBBISH"]:
            print(f"    - {r['analysis']['filename']}")


if __name__ == "__main__":
    main()
