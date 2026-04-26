#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "templates"


SEED_TEMPLATES = [
    "a visible {category} in the scene",
    "a photo of a {category}",
    "a {category} object",
    "the {category} in this image",
    "a photo containing a {category}",
    "a small {category} near the center",
    "a {category} in the scene",
    "a close-up of a {category}",
]

DESCRIPTIVE_PREFIXES = [
    "a clear photo of",
    "an image showing",
    "a picture with",
    "a scene featuring",
    "a cropped view of",
    "a single",
    "a full view of",
    "a partial view of",
    "a front view of",
    "a side view of",
    "a distant",
    "a nearby",
    "a large",
    "a tiny",
    "a bright",
    "a dark",
    "a blurry",
    "a sharp",
    "a real",
    "an actual",
    "a natural",
    "a man-made",
    "a parked",
    "a moving",
    "a standing",
    "a sitting",
    "a lying",
    "a hanging",
    "a stacked",
    "a clean view of",
    "a messy view of",
    "a real-world",
    "a photographic",
    "a natural image of",
    "a surveillance-style view of",
    "a zoomed view of",
    "a wide-angle view with",
    "a low-resolution view of",
    "a high-resolution view of",
]

CONTEXT_TEMPLATES = [
    "a {category} on the ground",
    "a {category} on a table",
    "a {category} next to a wall",
    "a {category} in front of a wall",
    "a {category} near a road",
    "a {category} beside a person",
    "a {category} inside a room",
    "a {category} outdoors",
    "a {category} under daylight",
    "a {category} in the background",
    "a {category} in the foreground",
    "a {category} near the edge of the image",
    "a {category} at the left side",
    "a {category} at the right side",
    "a {category} at the top of the image",
    "a {category} at the bottom of the image",
    "a {category} among other objects",
    "a {category} partly hidden",
    "a {category} with visible outline",
    "a {category} with clear shape",
    "a {category} near furniture",
    "a {category} near vegetation",
    "a {category} near water",
    "a {category} near a vehicle",
    "a {category} near a building",
    "a {category} near an animal",
    "a {category} near food",
    "a {category} near equipment",
    "a {category} near the camera",
    "a {category} far from the camera",
    "a {category} seen from above",
    "a {category} seen from below",
    "a {category} in profile",
    "a {category} facing the camera",
    "a {category} turned away",
    "a {category} with occlusion",
    "a {category} without occlusion",
    "a {category} in clutter",
    "a {category} in open space",
    "a {category} near text or signs",
    "a {category} near shadows",
    "a {category} in low contrast",
    "a {category} in high contrast",
    "a {category} with strong color",
    "a {category} with neutral color",
    "a {category} at small scale",
    "a {category} at medium scale",
    "a {category} at large scale",
    "a {category} cropped by the frame",
    "a {category} fully inside the frame",
    "a {category} overlapping another object",
    "a {category} separated from other objects",
    "a {category} on a shelf",
    "a {category} on a seat",
    "a {category} on a floor",
    "a {category} on grass",
    "a {category} on pavement",
    "a {category} on a plate",
    "a {category} in a container",
    "a {category} in a vehicle",
    "a {category} attached to something",
    "a {category} held by someone",
    "a {category} worn by someone",
    "a {category} used by someone",
    "a {category} near the image center",
    "a {category} away from the image center",
    "a {category} in the upper left area",
    "a {category} in the upper right area",
    "a {category} in the lower left area",
    "a {category} in the lower right area",
    "a {category} in a busy background",
    "a {category} in a plain background",
    "a {category} near a doorway",
    "a {category} near a window",
    "a {category} near a curb",
    "a {category} near a sign",
    "a {category} near a pole",
    "a {category} near a fence",
    "a {category} near a tree",
    "a {category} near a counter",
    "a {category} near a bed",
    "a {category} near a chair",
    "a {category} near a bicycle",
    "a {category} near a car",
    "a {category} near a bus",
    "a {category} near a train",
    "a {category} under shade",
    "a {category} under artificial light",
    "a {category} in a photo frame",
    "a {category} in a mirror-like surface",
    "a {category} behind another object",
    "a {category} in front of another object",
    "a {category} at an unusual angle",
    "a {category} with unusual appearance",
    "a {category} partly outside the image",
    "a {category} mostly visible",
    "a {category} barely visible",
]

SCENE_TEMPLATES = [
    "a crowded scene with a {category}",
    "a sparse scene with a {category}",
    "a street scene containing a {category}",
    "an indoor scene containing a {category}",
    "an outdoor scene containing a {category}",
    "a kitchen scene with a {category}",
    "a living area with a {category}",
    "a sports scene with a {category}",
    "a traffic scene with a {category}",
    "a market scene with a {category}",
    "a park scene with a {category}",
    "a vehicle scene with a {category}",
    "a household scene with a {category}",
]

AMBIGUOUS_TEMPLATES = [
    "perhaps a {category}",
    "what looks like a {category}",
    "maybe a {category}",
    "possibly a {category}",
    "a likely {category}",
    "a suspected {category}",
    "an object resembling a {category}",
    "something shaped like a {category}",
    "something similar to a {category}",
    "an item that could be a {category}",
    "a candidate {category}",
    "a possible instance of {category}",
    "a {category} that is hard to see",
    "a {category} that may be absent",
]

TARGET_TEMPLATES = [
    "a visible instance of {category}",
    "one visible {category}",
    "multiple {category} instances",
    "a group of {category} objects",
    "a {category} cluster",
    "a {category} region",
    "a {category} target",
    "a {category} detection target",
    "find the {category}",
    "locate the {category}",
    "identify any {category}",
    "look for a {category}",
    "show me a {category}",
    "where is the {category}",
    "the main {category}",
    "the nearest {category}",
    "the most visible {category}",
    "a {category} candidate box",
    "a {category} proposal",
]

NEGATIVE_TEMPLATES = [
    "no obvious {category}",
    "no clear {category}",
    "no visible {category}",
    "there is no {category} here",
    "this scene does not contain a {category}",
    "I do not see a {category}",
    "without a {category}",
    "a scene lacking a {category}",
    "an image without any {category}",
    "background regions not containing a {category}",
]


def build_templates() -> list[str]:
    templates: list[str] = []
    templates.extend(SEED_TEMPLATES)
    for prefix in DESCRIPTIVE_PREFIXES:
        if prefix.endswith(("of", "showing", "with", "featuring")):
            templates.append(f"{prefix} a {{category}}")
        else:
            templates.append(f"{prefix} {{category}}")
    templates.extend(CONTEXT_TEMPLATES)
    templates.extend(SCENE_TEMPLATES)
    templates.extend(AMBIGUOUS_TEMPLATES)
    templates.extend(TARGET_TEMPLATES)
    templates.extend(NEGATIVE_TEMPLATES)

    unique: list[str] = []
    seen = set()
    for template in templates:
        if "{category}" not in template:
            raise ValueError(f"Template is missing {{category}}: {template}")
        if template not in seen:
            unique.append(template)
            seen.add(template)
    if len(unique) < 200:
        raise ValueError(f"Need at least 200 templates, got {len(unique)}")
    return unique[:200]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    templates = build_templates()
    for count in (50, 100, 200):
        path = OUT_DIR / f"freeform_{count}.txt"
        path.write_text("\n".join(templates[:count]) + "\n", encoding="utf-8")
        print(f"wrote {path} ({count} templates)")


if __name__ == "__main__":
    main()
