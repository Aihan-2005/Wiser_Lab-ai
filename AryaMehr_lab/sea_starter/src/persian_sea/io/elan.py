from pathlib import Path
import xml.etree.ElementTree as ET

from persian_sea.types import SignSegment


def available_tiers(eaf_path: str | Path) -> list[str]:
    root = ET.parse(eaf_path).getroot()
    return [
        t.get("TIER_ID")
        for t in root.findall("TIER")
        if t.get("TIER_ID")
    ]


def read_sign_segments(
    eaf_path: str | Path,
    accepted_tiers: list[str],
) -> tuple[list[SignSegment], str]:
    eaf_path = Path(eaf_path)
    root = ET.parse(eaf_path).getroot()

    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"No TIME_ORDER in {eaf_path}")

    slots: dict[str, float] = {}
    for ts in time_order.findall("TIME_SLOT"):
        slot_id = ts.get("TIME_SLOT_ID")
        value = ts.get("TIME_VALUE")
        if slot_id and value is not None:
            slots[slot_id] = float(value) / 1000.0

    selected_tier = None
    selected_name = None
    for requested in accepted_tiers:
        for tier in root.findall("TIER"):
            if tier.get("TIER_ID") == requested:
                selected_tier = tier
                selected_name = requested
                break
        if selected_tier is not None:
            break

    if selected_tier is None:
        raise ValueError(
            f"None of {accepted_tiers} exists in {eaf_path}. "
            f"Available tiers: {available_tiers(eaf_path)}"
        )

    segments: list[SignSegment] = []
    for annotation in selected_tier.findall("ANNOTATION"):
        elem = next(iter(annotation), None)
        if elem is None:
            continue

        ref1 = elem.attrib.get("TIME_SLOT_REF1")
        ref2 = elem.attrib.get("TIME_SLOT_REF2")
        if ref1 not in slots or ref2 not in slots:
            continue

        value = elem.find("ANNOTATION_VALUE")
        text = value.text.strip() if value is not None and value.text else ""

        segment = SignSegment(slots[ref1], slots[ref2], text)
        try:
            segment.validate()
        except ValueError:
            continue
        segments.append(segment)

    segments.sort(key=lambda s: (s.start, s.end))
    return segments, selected_name or ""
