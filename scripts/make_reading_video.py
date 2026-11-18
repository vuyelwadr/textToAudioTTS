#!/usr/bin/env python3
"""Build a reading video from an audio file and matching source text.

This script is intended as a common final step for both local Kokoro output
and the web TTS downloader output. It creates:

1. A paper-like orange background image.
2. Timed text overlays based on the source text and audio duration.
3. An MP4 video rendered with ffmpeg.

The timing is estimated from text length, so it is good enough for readable
"study video" output without needing a forced-alignment model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def require_binary(name: str) -> str:
    binary_path = shutil.which(name)
    if not binary_path:
        raise RuntimeError(f"Required binary '{name}' was not found on PATH")
    return binary_path


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<![.!?])\n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_into_phrases(text: str, max_chars: int) -> list[str]:
    normalized = normalize_text(text)
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    phrases: list[str] = []

    for paragraph in paragraphs:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        current = ""

        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                phrases.append(current)
                current = ""

            wrapped_parts = textwrap.wrap(sentence, width=max_chars, break_long_words=False, break_on_hyphens=False)
            if wrapped_parts:
                phrases.extend(part.strip() for part in wrapped_parts if part.strip())
            else:
                phrases.append(sentence)

        if current:
            phrases.append(current)

    return phrases


def probe_audio_duration(audio_path: Path) -> float:
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise RuntimeError(f"Audio duration must be positive, got {duration}")
    return duration


def build_segments(phrases: list[str], total_duration: float, gap_seconds: float) -> list[Segment]:
    if not phrases:
        raise RuntimeError("No readable text segments were produced from the input text")

    weights = [max(1, len(re.findall(r"\S+", phrase))) for phrase in phrases]
    total_gaps = gap_seconds * max(0, len(phrases) - 1)
    usable_duration = max(0.5, total_duration - total_gaps)
    min_duration = min(1.8, usable_duration / len(phrases))
    reserved_duration = min_duration * len(phrases)
    distributable = max(0.0, usable_duration - reserved_duration)
    weight_total = sum(weights)

    segments: list[Segment] = []
    cursor = 0.0
    for index, (phrase, weight) in enumerate(zip(phrases, weights)):
        bonus = distributable * (weight / weight_total) if weight_total else 0.0
        duration = min_duration + bonus
        end = total_duration if index == len(phrases) - 1 else min(total_duration, cursor + duration)
        segments.append(Segment(index=index, start=cursor, end=end, text=phrase))
        cursor = min(total_duration, end + gap_seconds)

    if segments:
        segments[-1].end = total_duration

    return segments


def wrap_subtitle_text(text: str, width: int) -> str:
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(lines) if lines else text


def clamp_color(value: int) -> int:
    return max(0, min(255, value))


def build_paper_background_image(width: int, height: int, seed: int) -> Image.Image:
    randomizer = random.Random(seed)
    base_r, base_g, base_b = 214, 168, 114
    center_x = width / 2.0
    center_y = height / 2.0
    pixels = bytearray(width * height * 3)
    pixel_index = 0

    for y_pos in range(height):
        wave = int(7 * math.sin(y_pos / 33.0))
        band = 4 if y_pos % 61 == 0 else 0
        for x_pos in range(width):
            noise = randomizer.randint(-8, 8)
            fiber = int(5 * math.sin((x_pos + y_pos) / 52.0))
            edge_x = abs(x_pos - center_x) / center_x
            edge_y = abs(y_pos - center_y) / center_y
            vignette = int((edge_x + edge_y) * 10)
            speck = 10 if randomizer.random() < 0.0015 else 0

            red = clamp_color(base_r + wave + fiber + noise + band + speck - vignette)
            green = clamp_color(base_g + wave + fiber + noise + band + speck - vignette)
            blue = clamp_color(base_b + wave + fiber + noise + band + speck - vignette)

            pixels[pixel_index] = red
            pixels[pixel_index + 1] = green
            pixels[pixel_index + 2] = blue
            pixel_index += 3

    return Image.frombytes("RGB", (width, height), bytes(pixels)).convert("RGBA")


DEFAULT_FONT_PATH_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
    Path("/Library/Fonts/Georgia.ttf"),
    Path.home() / "Library" / "Fonts" / "Georgia.ttf",
    Path("/usr/share/fonts/truetype/msttcorefonts/Georgia.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/georgia.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSerif.ttf"),
)

DEFAULT_FONT_NAME_CANDIDATES = (
    "Georgia",
    "Times New Roman",
    "DejaVu Serif",
    "DejaVuSerif.ttf",
)


def choose_default_font_argument() -> str:
    for candidate in DEFAULT_FONT_PATH_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    for candidate_name in DEFAULT_FONT_NAME_CANDIDATES:
        try:
            ImageFont.truetype(candidate_name, 72)
            return candidate_name
        except OSError:
            continue

    tried = ", ".join([str(path) for path in DEFAULT_FONT_PATH_CANDIDATES] + list(DEFAULT_FONT_NAME_CANDIDATES))
    raise RuntimeError(f"Unable to find a default scalable serif font. Tried: {tried}")


def resolve_font(font_name: str, font_size: int) -> ImageFont.ImageFont:
    requested_font_name = font_name.strip()
    allow_default_family_fallback = requested_font_name.lower() in {"", "auto", "default"}
    if allow_default_family_fallback:
        font_name = choose_default_font_argument()
    else:
        font_name = requested_font_name
    font_path = Path(font_name)
    candidate_paths: list[Path] = []
    if font_path.exists():
        candidate_paths.append(font_path)
    else:
        font_dirs = [
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
        for font_dir in font_dirs:
            for extension in (".ttf", ".otf", ".ttc"):
                candidate_paths.append(font_dir / f"{font_name}{extension}")

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            try:
                return ImageFont.truetype(str(candidate_path), font_size)
            except OSError:
                continue

    fallback_names = (font_name,) if not allow_default_family_fallback else (font_name, *DEFAULT_FONT_NAME_CANDIDATES)
    for candidate_name in fallback_names:
        try:
            return ImageFont.truetype(candidate_name, font_size)
        except OSError:
            continue

    raise RuntimeError(f"Unable to resolve a scalable font for {font_name!r} at size {font_size}")


def wrap_text_for_card(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join(current_words + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        candidate_width = bbox[2] - bbox[0]
        if candidate_width <= max_width or not current_words:
            current_words.append(word)
        else:
            lines.append(" ".join(current_words))
            current_words = [word]

    if current_words:
        lines.append(" ".join(current_words))

    return lines


def render_segment_cards(directory: Path, segments: list[Segment], width: int, height: int, font_name: str, font_size: int, seed: int) -> list[Path]:
    base_image = build_paper_background_image(width, height, seed)
    font = resolve_font(font_name, font_size)
    max_text_width = int(width * 0.72)
    spacing = 14
    card_paths: list[Path] = []

    for segment in segments:
        frame = base_image.copy()
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        wrapped_lines = wrap_text_for_card(segment.text, draw, font, max_text_width)
        wrapped_text = "\n".join(wrapped_lines)
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=spacing, align="center")
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        panel_padding_x = 70
        panel_padding_y = 48
        panel_width = text_width + (panel_padding_x * 2)
        panel_height = text_height + (panel_padding_y * 2)
        panel_left = int((width - panel_width) / 2)
        panel_top = height - panel_height - 120
        panel_right = panel_left + panel_width
        panel_bottom = panel_top + panel_height

        draw.rounded_rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            radius=32,
            fill=(248, 242, 231, 88),
            outline=(243, 233, 217, 180),
            width=2,
        )

        text_x = int((width - text_width) / 2)
        text_y = int(panel_top + ((panel_height - text_height) / 2) - text_bbox[1])
        draw.multiline_text(
            (text_x, text_y),
            wrapped_text,
            font=font,
            fill=(43, 31, 22, 255),
            spacing=spacing,
            align="center",
        )

        rendered_frame = Image.alpha_composite(frame, overlay).convert("RGB")
        card_path = directory / f"segment_{segment.index:04d}.png"
        rendered_frame.save(card_path, format="PNG")
        card_paths.append(card_path)

    return card_paths


def write_concat_manifest(path: Path, card_paths: list[Path], segments: list[Segment]) -> None:
    lines: list[str] = []
    for card_path, segment in zip(card_paths, segments):
        lines.append(f"file '{card_path.as_posix()}'\n")
        lines.append(f"duration {max(0.1, segment.end - segment.start):.3f}\n")
    if card_paths:
        lines.append(f"file '{card_paths[-1].as_posix()}'\n")
    path.write_text("".join(lines), encoding="utf-8")


def render_video(audio_path: Path, concat_manifest_path: Path, output_path: Path, duration: float) -> None:
    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_manifest_path),
        "-i",
        str(audio_path),
        "-vf",
        "fps=25,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-t",
        f"{duration:.3f}",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paper-style reading video from audio and text")
    parser.add_argument("--audio", required=True, type=Path, help="Rendered audio file, for example a Kokoro or web TTS MP3/WAV")
    parser.add_argument("--text", required=True, type=Path, help="Source text file that matches the audio")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument(
        "--font",
        default="auto",
        help="Subtitle font family or path. 'auto' selects a verified scalable serif font.",
    )
    parser.add_argument("--font-size", type=int, default=54, help="Subtitle font size")
    parser.add_argument("--max-chars", type=int, default=120, help="Maximum characters per timed text segment")
    parser.add_argument("--gap", type=float, default=0.18, help="Gap between subtitle segments in seconds")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for paper texture generation")
    parser.add_argument("--keep-temp", action="store_true", help="Keep generated background and subtitle files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.audio.exists():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 1
    if not args.text.exists():
        print(f"Text file not found: {args.text}", file=sys.stderr)
        return 1

    try:
        duration = probe_audio_duration(args.audio)
        phrases = split_into_phrases(read_text_file(args.text), max_chars=args.max_chars)
        segments = build_segments(phrases, total_duration=duration, gap_seconds=args.gap)
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_temp:
        temp_dir_path = args.output.parent / f"{args.output.stem}_video_assets"
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        cleanup_needed = False
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="reading_video_", dir=str(args.output.parent))
        temp_dir_path = Path(temp_dir.name)
        cleanup_needed = True

    concat_manifest_path = temp_dir_path / "frames.txt"
    metadata_path = temp_dir_path / "timings.json"

    try:
        card_paths = render_segment_cards(
            temp_dir_path,
            segments=segments,
            width=args.width,
            height=args.height,
            font_name=args.font,
            font_size=args.font_size,
            seed=args.seed,
        )
        write_concat_manifest(concat_manifest_path, card_paths, segments)
        metadata_path.write_text(json.dumps([asdict(segment) for segment in segments], indent=2), encoding="utf-8")
        render_video(args.audio, concat_manifest_path, args.output, duration=duration)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"Video build failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if cleanup_needed:
            temp_dir.cleanup()

    print(f"Created reading video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
