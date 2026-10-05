from pathlib import Path
import xml.etree.ElementTree as ET

from persian_sea.types import Cue, SignSegment


def read_sign_segments(
    eaf_path: str | Path,
    accepted_tiers: list[str],
) -> tuple[list[SignSegment], str]:
    eaf_path = Path(eaf_path)
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"No TIME_ORDER in {eaf_path}")

    time_slots: dict[str, float] = {}
    for ts in time_order.findall("TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        value = ts.get("TIME_VALUE")
        if ts_id and value is not None:
            time_slots[ts_id] = float(value) / 1000.0

    selected_tier = None
    selected_tier_name = None

    for tier_name in accepted_tiers:
        for tier in root.findall("TIER"):
            if tier.get("TIER_ID") == tier_name:
                selected_tier = tier
                selected_tier_name = tier_name
                break
        if selected_tier is not None:
            break

    if selected_tier is None:
        available = [t.get("TIER_ID") for t in root.findall("TIER")]
        raise ValueError(
            f"None of ELAN tiers {accepted_tiers} found in {eaf_path}. "
            f"Available tiers: {available}"
        )

    segments: list[SignSegment] = []
    for annotation in selected_tier.findall("ANNOTATION"):
        elem = next(iter(annotation), None)
        if elem is None:
            continue

        ref1 = elem.attrib.get("TIME_SLOT_REF1")
        ref2 = elem.attrib.get("TIME_SLOT_REF2")
        if ref1 not in time_slots or ref2 not in time_slots:
            continue

        text_elem = elem.find("ANNOTATION_VALUE")
        text = text_elem.text if text_elem is not None and text_elem.text else ""

        seg = SignSegment(
            start=time_slots[ref1],
            end=time_slots[ref2],
            text=text,
        )
        try:
            seg.validate()
        except ValueError:
            continue
        segments.append(seg)

    segments.sort(key=lambda x: (x.start, x.end))
    return segments, selected_tier_name or ""


def _remove_existing_tier(root: ET.Element, tier_id: str) -> None:
    for tier in list(root.findall("TIER")):
        if tier.get("TIER_ID") == tier_id:
            root.remove(tier)


def write_aligned_subtitle_tier(
    source_eaf: str | Path,
    cues: list[Cue],
    output_eaf: str | Path,
    tier_id: str = "SUBTITLE_SHIFTED",
) -> None:
    source_eaf = Path(source_eaf)
    output_eaf = Path(output_eaf)

    tree = ET.parse(source_eaf)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"No TIME_ORDER in {source_eaf}")

    _remove_existing_tier(root, tier_id)

    ling_type = "default-lt"
    for tier in root.findall("TIER"):
        ref = tier.get("LINGUISTIC_TYPE_REF")
        if ref:
            ling_type = ref
            break

    new_tier = ET.Element(
        "TIER",
        {"TIER_ID": tier_id, "LINGUISTIC_TYPE_REF": ling_type},
    )

    for i, cue in enumerate(cues):
        ts1 = f"PSEA_SUB_{i}_1"
        ts2 = f"PSEA_SUB_{i}_2"

        time_order.append(
            ET.Element(
                "TIME_SLOT",
                {
                    "TIME_SLOT_ID": ts1,
                    "TIME_VALUE": str(int(round(cue.start * 1000))),
                },
            )
        )
        time_order.append(
            ET.Element(
                "TIME_SLOT",
                {
                    "TIME_SLOT_ID": ts2,
                    "TIME_VALUE": str(int(round(cue.end * 1000))),
                },
            )
        )

        annotation = ET.Element("ANNOTATION")
        alignable = ET.Element(
            "ALIGNABLE_ANNOTATION",
            {
                "ANNOTATION_ID": f"psea_sub_{i}",
                "TIME_SLOT_REF1": ts1,
                "TIME_SLOT_REF2": ts2,
            },
        )
        value = ET.Element("ANNOTATION_VALUE")
        value.text = cue.text
        alignable.append(value)
        annotation.append(alignable)
        new_tier.append(annotation)

    root.append(new_tier)
    output_eaf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_eaf, encoding="utf-8", xml_declaration=True)
