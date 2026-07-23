"""Read-only safety net for train_manager_run1.log: reconstructs a resume-compatible
checkpoint from the log's stdout, so today's run (already executing the pre-checkpoint
version of train_manager.py, which never writes checkpoint.json itself) can still be
resumed on Monday via `--resume-from train_manager_output_run1/checkpoint.json`.

Only includes images from the *last fully-merged batch* (i.e. everything strictly before
the last "--- updated training prompt ---" marker) -- any trailing images processed after
that but before a cutoff are deliberately left OUT and will just be redone on resume, so
the running_prompt/notes pair written here is always internally consistent (matches what
train_manager.py itself would have had in memory at that checkpoint).

Never touches the live training process -- only reads its log file.
"""
import json
import re
import time
from pathlib import Path

LOG = Path("train_manager_run1.log")
OUT_DIR = Path("train_manager_output_run1")
PROMPT_TXT = OUT_DIR / "checkpoint_prompt.txt"
CHECKPOINT_JSON = OUT_DIR / "checkpoint.json"
MARKER = "--- updated training prompt ---\n"
POLL_SECONDS = 90

IMAGE_RE = re.compile(
    r"=== \[(\d+)/(\d+)\] (\S+) \((\w+)\) ===.*?\n  initial=(\S+)  final=(\S+)  accepted=(\S+)",
    re.DOTALL,
)


def parse_num(s: str):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def build_checkpoint(text: str):
    marker_positions = [m.start() for m in re.finditer(re.escape(MARKER), text)]
    if not marker_positions:
        return None
    last_marker_pos = marker_positions[-1]

    after = text[last_marker_pos + len(MARKER):]
    end_m = re.search(r"\n(?:=== \[|Final training prompt written to)", after)
    running_prompt = (after[: end_m.start()] if end_m else after).strip()

    before = text[:last_marker_pos]
    notes = []
    for m in IMAGE_RE.finditer(before):
        _, total, image_id, agent, initial, final, accepted = m.groups()
        notes.append({
            "image_id": image_id, "agent": agent, "lower_is_better": agent == "countgd",
            "initial_score": parse_num(initial), "final_score": parse_num(final),
            "accepted": accepted == "True", "adjustments": [],
            "output_characteristics": "(reconstructed from log by checkpoint_extract.py -- detail not preserved)",
        })
    total = int(m.group(2)) if notes else 0
    return running_prompt, notes, total


def main():
    last_written = None
    while True:
        if LOG.exists():
            text = LOG.read_text(errors="replace")
            result = build_checkpoint(text)
            if result:
                running_prompt, notes, total = result
                if running_prompt and running_prompt != last_written:
                    PROMPT_TXT.write_text(
                        f"[checkpoint saved {time.strftime('%Y-%m-%d %H:%M:%S')} -- "
                        f"{len(notes)}/{total} images processed, last fully-merged batch checkpoint below]\n\n"
                        f"{running_prompt}\n"
                    )
                    CHECKPOINT_JSON.write_text(json.dumps({
                        "completed_ids": [n["image_id"] for n in notes],
                        "running_prompt": running_prompt,
                        "notes": notes,
                    }, indent=2))
                    last_written = running_prompt
                    print(f"[{time.strftime('%H:%M:%S')}] checkpoint.json updated -- {len(notes)}/{total} images")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
