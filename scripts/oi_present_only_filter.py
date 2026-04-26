"""
OpenImages V7 one-sided clean subset:
- Filter: keep only families where is_prompt_absent=False (i.e. prompt category
  has at least one GT annotation in the image).
- Rationale: OI bbox annotations are exhaustive WITHIN a (image, category) pair
  that has any annotation. The "absent" prompts in the federated eval are not
  backed by negative image-level labels in the converted COCO json, so they
  contribute the analogous "UNKNOWN" inflation seen in LVIS.
- Limitation: this is one-sided (present-only); we lose the absent-prompt
  evaluation. But it cleanly demonstrates the same finding as LVIS.
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_JSONL = REPO_ROOT / "data_raw/a_project_2/outputs/oi_1000_gdino.jsonl"
OUT_JSONL = REPO_ROOT / "data_raw/a_project_2/outputs/oi_1000_gdino_present_only.jsonl"


def main():
    n_in, n_out = 0, 0
    n_present, n_absent = 0, 0
    images_kept = set()
    with open(PILOT_JSONL, "r") as f_in, open(OUT_JSONL, "w") as f_out:
        for line in f_in:
            n_in += 1
            fam = json.loads(line)
            if fam["is_prompt_absent"]:
                n_absent += 1
            else:
                n_present += 1
                f_out.write(line)
                n_out += 1
                images_kept.add(fam["image_id"])

    meta = {
        "source_jsonl": str(PILOT_JSONL),
        "filter": "is_prompt_absent == False (present-only)",
        "num_families_in": n_in,
        "num_families_out": n_out,
        "num_present": n_present,
        "num_absent": n_absent,
        "num_unique_images": len(images_kept),
    }
    with open(str(OUT_JSONL) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"in={n_in} (present={n_present}, absent={n_absent}), kept={n_out}, images={len(images_kept)}")
    print(f"written: {OUT_JSONL}")


if __name__ == "__main__":
    main()
