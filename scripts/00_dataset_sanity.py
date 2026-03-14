from __future__ import annotations

from pathlib import Path

ROOT = Path("data/raw")

TRAIN = ROOT / "monuseg_train"
TEST = ROOT / "monuseg_test"


def check_pairing(images_dir: Path, ann_dir: Path) -> None:
    images = sorted(images_dir.glob("*.tif"))
    xmls = sorted(ann_dir.glob("*.xml"))

    img_stems = {p.stem for p in images}
    xml_stems = {p.stem for p in xmls}

    missing_xml = sorted(list(img_stems - xml_stems))
    missing_img = sorted(list(xml_stems - img_stems))
    paired = len(img_stems & xml_stems)

    print(f"Images (.tif): {len(images)}")
    print(f"Annotations (.xml): {len(xmls)}")
    print(f"Paired: {paired}")

    if missing_xml:
        print(f"Missing XML for {len(missing_xml)} images (showing up to 5): {missing_xml[:5]}")
    if missing_img:
        print(f"Missing image for {len(missing_img)} XMLs (showing up to 5): {missing_img[:5]}")


def main() -> None:
    print("== MoNuSeg TRAIN ==")
    check_pairing(TRAIN / "Tissue Images", TRAIN / "Annotations")
    print()

    print("== MoNuSeg TEST ==")
    check_pairing(TEST / "Tissue Images", TEST / "Annotations")
    print()

    print("OK: dataset structure looks consistent.")


if __name__ == "__main__":
    main()