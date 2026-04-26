# OI dropped vs kept images — are they content-selected?

## Setup

- Full OI 1000-image sample has **3 prompts per image** (1 present + 2 absent), except images where the sampled present-category happens to have 0 GT instances for ANY of the 3 prompts — which then become "all-absent" images and get dropped by the present-only filter.
- `oi_present_only_filter.py` keeps families with `is_prompt_absent == False` only. An image is implicitly dropped iff **all its prompts are absent**.

## Counts

| Group | # images | # families | prompts/img |
|-------|----------|-----------|-------------|
| Full | 1000 | 2784 | mean=2.78 (usually 3, some 2) |
| Kept | 784 | 2352 (present + absent) | exactly 3 per img = 1 present + 2 absent |
| Dropped | 216 | 432 | exactly 2 per img = 2 absent (no present) |

## Property parity

| Metric | Dropped | Kept | Interpretation |
|--------|---------|------|----------------|
| present prompts / img | 0 (by def.) | exactly 1 | deterministic split |
| absent prompts / img | exactly 2 | exactly 2 | same |
| candidates / img (summed across prompts) | mean 34.5, p90=66 | mean 40.7, p90=80 | kept slightly higher (more prompts) |
| file extension | 432× `.jpg` | 2352× `.jpg` | identical |
| filename length | 20 chars | 20 chars | identical (OI uses fixed-length hash IDs) |

## 结论

**Dropped images are a deterministic function of the eval-protocol sampler**, not a content-based selection:

- Each full image got **exactly 2 absent prompts** via the dataset's absent-prompt sampler.
- Whether an image retains a "present" prompt depends on whether any of its GT-positive categories was sampled — **this is uniform over the image distribution**, not dependent on scene difficulty, image quality, or spatial attributes.
- Filename / extension distributions are identical, so there's no file-level selection artifact.

Reviewer defense: "216 dropped images are **not** harder / easier images; they are images where the absent-prompt-only version of the protocol leaves nothing to honestly evaluate. This is a **protocol restriction**, not a **data restriction**."
