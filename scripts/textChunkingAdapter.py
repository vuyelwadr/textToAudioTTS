#!/usr/bin/env python3
"""
Text Chunking Adapter for Web TTS
Optimized chunking for web-based TTS services with large character limits
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[1]
INDABA_DIR = REPO_ROOT / "library" / "credoMutwa" / "indaba"
DEFAULT_INPUT_FILE = INDABA_DIR / "indaba1.1.txt"
DEFAULT_OUTPUT_DIR = INDABA_DIR / "ttsOnline"


class WebTTSChunker:
    """
    Text chunking adapter optimized for web TTS services with large character limits
    """

    def __init__(
        self,
        max_chunk_size: int = 6000,
        min_chunk_size: int = 500,
        output_dir: str = None,
    ):
        """
        Initialize the web TTS chunker

        Args:
            max_chunk_size: Maximum characters per chunk (default: 6000 for web TTS)
            min_chunk_size: Minimum characters per chunk (default: 500)
            output_dir: Directory to save chunks and manifest
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.output_dir = output_dir or "ttsOnline"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Sentence-ending punctuation
        self.sentence_endings = r"[.!?]+"

        # Clause separators for better chunking
        self.clause_separators = r"[;,:]"

        # Sentinel marking a real paragraph break (blank line in the source).
        # Inserted before any other newline handling so it survives regardless
        # of whether the paragraph happens to end mid-sentence or not, then
        # used by chunk_text_optimized() to prefer paragraph-aligned chunk
        # boundaries and to restore visible paragraph breaks in chunk text.
        self.PARAGRAPH_MARKER = " "

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better chunking and TTS processing

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        # Mark real paragraph breaks (blank line) with a sentinel before any
        # other newline handling, so genuine paragraph boundaries can be found
        # later regardless of where sentence-ending punctuation happens to fall.
        text = re.sub(r"[ \t]*\n[ \t]*\n[ \t\n]*", self.PARAGRAPH_MARKER, text)

        # Any remaining newlines are just line-wraps inside a paragraph; join them.
        text = re.sub(r"\n", " ", text)

        # Clean up excessive whitespace
        text = re.sub(r" +", " ", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.!?])", r"\1", text)
        text = re.sub(r"([;,:])\s+", r"\1 ", text)

        # Tidy whitespace directly around the paragraph marker itself
        text = re.sub(r"\s*" + self.PARAGRAPH_MARKER + r"\s*", self.PARAGRAPH_MARKER, text)

        return text.strip(self.PARAGRAPH_MARKER + " \t\n")

    def find_paragraph_boundaries(self, text: str) -> List[int]:
        """
        Find positions right after each paragraph-marker sentinel.

        Args:
            text: Normalized text (already containing PARAGRAPH_MARKER)

        Returns:
            List of character positions where a paragraph ends
        """
        return [m.end() for m in re.finditer(self.PARAGRAPH_MARKER, text)]

    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find positions of sentence boundaries in text

        Args:
            text: Text to analyze

        Returns:
            List of character positions where sentences end
        """
        boundaries = []
        for match in re.finditer(self.sentence_endings, text):
            # Include some context after the punctuation
            end_pos = match.end()
            # Skip if at end of text
            if end_pos < len(text):
                # Find next non-whitespace character
                while end_pos < len(text) and text[end_pos].isspace():
                    end_pos += 1
            boundaries.append(end_pos)
        return boundaries

    def find_clause_boundaries(self, text: str) -> List[int]:
        """
        Find positions of clause boundaries for secondary splitting points

        Args:
            text: Text to analyze

        Returns:
            List of character positions where clauses end
        """
        boundaries = []
        for match in re.finditer(self.clause_separators, text):
            boundaries.append(match.end())
        return boundaries

    def create_chunk(
        self,
        text: str,
        chunk_id: int,
        total_chunks: int,
        ends_at_paragraph: bool = False,
    ) -> Dict:
        """
        Create a chunk dictionary with metadata

        Args:
            text: Text content for the chunk
            chunk_id: Sequential ID of the chunk
            total_chunks: Total number of chunks
            ends_at_paragraph: Whether this chunk's boundary lands on a real
                paragraph break (used downstream to insert a pause between
                this chunk's audio and the next one)

        Returns:
            Chunk dictionary with metadata
        """
        # Restore any paragraph breaks that fall *inside* this chunk (when
        # several short paragraphs fit in one chunk) as real blank lines, so
        # the saved chunk text still shows paragraph structure for video
        # rendering and gives the TTS model a natural pacing cue.
        text = text.replace(self.PARAGRAPH_MARKER, "\n\n")

        # Create a unique hash for the chunk content
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

        # Generate filename with zero-padding for proper sorting
        filename = f"chunk_{chunk_id:04d}_{content_hash}.txt"

        # Calculate estimated duration (rough estimate: 150 words per minute)
        word_count = len(text.split())
        estimated_duration = word_count / 150 * 60  # seconds

        return {
            "chunk_id": chunk_id,
            "filename": filename,
            "text": text,
            "character_count": len(text),
            "word_count": word_count,
            "estimated_duration_seconds": estimated_duration,
            "content_hash": content_hash,
            "created_at": datetime.now().isoformat(),
            "total_chunks": total_chunks,
            "ends_at_paragraph": ends_at_paragraph,
        }

    def chunk_text_optimized(self, text: str) -> List[Dict]:
        """
        Chunk text optimized for web TTS with large character limits

        Args:
            text: Text to chunk

        Returns:
            List of chunk dictionaries
        """
        print(f"Starting text chunking with max size: {self.max_chunk_size} characters")

        # Normalize the text first
        normalized_text = self.normalize_text(text)
        print(f"Text normalized. Length: {len(normalized_text)} characters")

        if len(normalized_text) <= self.max_chunk_size:
            print("Text is small enough for a single chunk")
            return [self.create_chunk(normalized_text, 0, 1, ends_at_paragraph=True)]

        # Find all splitting points
        sentence_boundaries = self.find_sentence_boundaries(normalized_text)
        clause_boundaries = self.find_clause_boundaries(normalized_text)
        paragraph_boundaries = self.find_paragraph_boundaries(normalized_text)

        chunks = []
        start_pos = 0
        chunk_id = 0

        while start_pos < len(normalized_text):
            # Calculate the ideal end position
            ideal_end = start_pos + self.max_chunk_size

            # If we're near the end, just take what's left
            if ideal_end >= len(normalized_text):
                chunk_text = normalized_text[start_pos:].strip()
                if chunk_text:
                    chunks.append(self.create_chunk(chunk_text, chunk_id, None, ends_at_paragraph=True))
                break

            # Find the best breaking point before or at ideal_end
            best_break = None
            ends_at_paragraph = False

            # Prefer a real paragraph break nearest to (but not past) ideal_end,
            # so chunk boundaries line up with paragraph edges whenever the
            # size budget allows it -- this is what lets us insert a natural
            # pause between chunks that fall on a paragraph boundary.
            for boundary in paragraph_boundaries:
                if start_pos + self.min_chunk_size <= boundary <= ideal_end:
                    best_break = boundary
                    ends_at_paragraph = True

            # Otherwise fall back to sentence boundaries (preferred over clause)
            if best_break is None:
                for boundary in sentence_boundaries:
                    if start_pos < boundary <= ideal_end:
                        best_break = boundary

            # If no sentence boundary found, look for clause boundaries
            if best_break is None:
                for boundary in clause_boundaries:
                    if start_pos < boundary <= ideal_end:
                        best_break = boundary

            # If still no good break point, use the ideal_end
            if best_break is None:
                best_break = ideal_end
                # Try to break at a space to avoid cutting words
                if best_break < len(normalized_text):
                    space_pos = normalized_text.rfind(" ", start_pos, best_break)
                    if space_pos > start_pos + self.min_chunk_size:
                        best_break = space_pos

            # Extract the chunk
            chunk_text = normalized_text[start_pos:best_break].strip(self.PARAGRAPH_MARKER + " \t\n")

            # Only create chunk if it meets minimum size requirements
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    self.create_chunk(chunk_text, chunk_id, None, ends_at_paragraph=ends_at_paragraph)
                )
                chunk_id += 1

            start_pos = best_break

        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        print(f"Created {len(chunks)} chunks")
        print(
            f"Chunk sizes: {[c['character_count'] for c in chunks[:10]]}{'...' if len(chunks) > 10 else ''}"
        )

        return chunks

    def save_chunks(
        self,
        chunks: List[Dict],
        prefix: str = "indaba",
        source_file: str | None = None,
        source_text_sha256: str | None = None,
    ) -> str:
        """
        Save chunks to files and create manifest

        Args:
            chunks: List of chunk dictionaries
            prefix: Prefix for chunk files

        Returns:
            Path to the manifest file
        """
        print(f"Saving {len(chunks)} chunks to {self.output_dir}")

        # Save individual chunk files
        for chunk in chunks:
            filepath = os.path.join(self.output_dir, chunk["filename"])
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chunk["text"])

        # Create manifest
        manifest = {
            "metadata": {
                "total_chunks": len(chunks),
                "total_characters": sum(c["character_count"] for c in chunks),
                "total_words": sum(c["word_count"] for c in chunks),
                "estimated_total_duration_seconds": sum(
                    c["estimated_duration_seconds"] for c in chunks
                ),
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size,
                "created_at": datetime.now().isoformat(),
                "prefix": prefix,
                "source_file": source_file or "unknown",
                "source_text_sha256": source_text_sha256 or "unknown",
            },
            "chunks": [],
        }

        # Add chunk info to manifest (without the full text to save space)
        for chunk in chunks:
            chunk_info = {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "character_count": chunk["character_count"],
                "word_count": chunk["word_count"],
                "estimated_duration_seconds": chunk["estimated_duration_seconds"],
                "content_hash": chunk["content_hash"],
                "created_at": chunk["created_at"],
                "ends_at_paragraph": chunk.get("ends_at_paragraph", False),
            }
            manifest["chunks"].append(chunk_info)

        # Save manifest
        manifest_path = os.path.join(self.output_dir, f"{prefix}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Manifest saved to: {manifest_path}")
        print(
            f"Total duration: {manifest['metadata']['estimated_total_duration_seconds'] / 60:.1f} minutes"
        )

        return manifest_path

    def create_processing_plan(self, manifest_path: str, num_workers: int = 4) -> Dict:
        """
        Create a processing plan for parallel TTS generation

        Args:
            manifest_path: Path to the manifest file
            num_workers: Number of parallel workers

        Returns:
            Processing plan dictionary
        """
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        chunks = manifest["chunks"]
        total_chunks = len(chunks)

        # Divide chunks among workers
        chunks_per_worker = (total_chunks + num_workers - 1) // num_workers

        processing_plan = {
            "metadata": {
                "num_workers": num_workers,
                "total_chunks": total_chunks,
                "chunks_per_worker": chunks_per_worker,
                "created_at": datetime.now().isoformat(),
            },
            "workers": [],
        }

        for worker_id in range(num_workers):
            start_chunk = worker_id * chunks_per_worker
            end_chunk = min(start_chunk + chunks_per_worker, total_chunks)

            if start_chunk >= total_chunks:
                break

            worker_chunks = chunks[start_chunk:end_chunk]

            worker_plan = {
                "worker_id": worker_id,
                "chunk_range": [start_chunk, end_chunk - 1],
                "num_chunks": len(worker_chunks),
                "chunks": [c["filename"] for c in worker_chunks],
            }

            processing_plan["workers"].append(worker_plan)

        # Save processing plan
        plan_path = os.path.join(self.output_dir, "processing_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(processing_plan, f, indent=2, ensure_ascii=False)

        print(f"Processing plan saved to: {plan_path}")
        print(
            f"Distributed {total_chunks} chunks across {len(processing_plan['workers'])} workers"
        )

        return processing_plan

    def process_file(
        self, input_file: str, prefix: str = "indaba", num_workers: int = 4
    ) -> Tuple[str, str]:
        """
        Process a text file into chunks ready for web TTS

        Args:
            input_file: Path to input text file
            prefix: Prefix for output files
            num_workers: Number of parallel workers to plan for

        Returns:
            Tuple of (manifest_path, processing_plan_path)
        """
        print(f"Processing file: {input_file}")

        # Read input file
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        source_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

        print(f"Read {len(text)} characters from {input_file}")

        # Create chunks
        chunks = self.chunk_text_optimized(text)

        # Save chunks and manifest
        manifest_path = self.save_chunks(
            chunks,
            prefix,
            source_file=Path(input_file).name,
            source_text_sha256=source_text_sha256,
        )

        # Create processing plan
        processing_plan = self.create_processing_plan(manifest_path, num_workers)

        # Print summary
        print(f"\nChunking Summary:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total characters: {sum(c['character_count'] for c in chunks):,}")
        print(f"  Total words: {sum(c['word_count'] for c in chunks):,}")
        print(
            f"  Estimated duration: {sum(c['estimated_duration_seconds'] for c in chunks) / 60:.1f} minutes"
        )
        print(
            f"  Average chunk size: {sum(c['character_count'] for c in chunks) / len(chunks):.0f} characters"
        )
        print(f"  Output directory: {self.output_dir}")

        return manifest_path, os.path.join(self.output_dir, "processing_plan.json")


def main():
    """
    Main function to process the indaba1.1.txt file
    """
    parser = argparse.ArgumentParser(description="Text Chunking Adapter for Web TTS")
    parser.add_argument("--input", type=str, default=None, help="Input text file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--max-chunk-size", type=int, default=6000, help="Max chunk size"
    )
    parser.add_argument(
        "--min-chunk-size", type=int, default=500, help="Min chunk size"
    )
    parser.add_argument("--num-workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--prefix", type=str, default="indaba1.1", help="Output prefix")
    args = parser.parse_args()

    # Configuration
    input_file = args.input if args.input else str(DEFAULT_INPUT_FILE)
    output_dir = args.output_dir if args.output_dir else str(DEFAULT_OUTPUT_DIR)

    # Initialize chunker with web-optimized settings
    chunker = WebTTSChunker(
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        output_dir=output_dir,
    )

    # Process the file
    manifest_path, plan_path = chunker.process_file(
        input_file=input_file,
        prefix=args.prefix,
        num_workers=args.num_workers,
    )

    print(f"\nFiles created:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Processing plan: {plan_path}")
    print(f"  Chunk files: {output_dir}/chunk_*.txt")

    print(f"\nReady for parallel TTS processing!")


if __name__ == "__main__":
    main()
