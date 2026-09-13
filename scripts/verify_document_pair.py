#!/usr/bin/env python3
"""I check my generated document pair, not rendered appearance or arbitrary OOXML."""

import argparse
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, BadZipFile

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
PATTERNS = {
    "placeholder": re.compile(r"\b(?:TODO|TBD|FIXME)\b|\{\{[^}]+\}\}|\[\[[^]]+\]\]"),
    "credential": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
                             r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bAKIA[A-Z0-9]{16}\b|"
                             r"\bBearer\s+[A-Za-z0-9_./+=-]{20,}", re.I),
}


def xml(package, name):
    entry = package.getinfo(name)
    if entry.file_size > 8 * 1024 * 1024:
        raise ValueError("I refuse an XML part larger than 8 MiB")
    data = package.read(name)
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("I do not admit XML entity declarations")
    return ET.fromstring(data)


def relations(package, part):
    path = PurePosixPath(part)
    root = xml(package, str(path.parent / "_rels" / (path.name + ".rels")))
    result = {}
    for rel in root.findall(f"{{{REL}}}Relationship"):
        identifier = rel.attrib["Id"]
        if identifier in result:
            raise ValueError("I found duplicate relationship IDs")
        target = rel.attrib["Target"]
        if rel.get("TargetMode") == "External":
            result[identifier] = (rel.attrib["Type"], None)
            continue
        target = posixpath.normpath(posixpath.join(str(path.parent), target))
        if target.startswith("../") or target.startswith("/") or "\\" in target:
            raise ValueError("I refuse an escaping package relationship")
        result[identifier] = (rel.attrib["Type"], target)
    return result


def scan_text(package, errors, label):
    for name in package.namelist():
        if not name.endswith(".xml"):
            continue
        root = xml(package, name)
        paragraphs = [node for node in root.iter()
                      if node.tag in (f"{{{NS['a']}}}p", f"{{{NS['w']}}}p")]
        text = "\n".join("".join(node.text or "" for node in paragraph.iter()
                        if node.tag in (f"{{{NS['a']}}}t", f"{{{NS['w']}}}t"))
                         for paragraph in paragraphs)
        for kind, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {kind} pattern in {name}")


def check_slides(path, expected, errors):
    with ZipFile(path) as package:
        root = xml(package, "ppt/presentation.xml")
        size = root.find("p:sldSz", NS)
        width, height = int(size.attrib["cx"]), int(size.attrib["cy"])
        if width <= 0 or height <= 0:
            raise ValueError("I need a positive slide surface")
        slides = root.findall("p:sldIdLst/p:sldId", NS)
        if len(slides) != expected:
            errors.append("deck: actual slide count differs from manifest")
        links = relations(package, "ppt/presentation.xml")
        seen = set()
        for number, slide in enumerate(slides, 1):
            kind, part = links[slide.attrib[f"{{{NS['r']}}}id"]]
            if not kind.endswith("/slide") or not part or part in seen:
                raise ValueError("I need unique internal slide relationships")
            seen.add(part)
            label = f"slide {number}"
            document = xml(package, part)
            frames = []
            tree = document.find("p:cSld/p:spTree", NS)
            if tree is None:
                raise ValueError("I need a slide shape tree")
            for shape in tree:
                tag = shape.tag.rsplit("}", 1)[-1]
                if tag in ("nvGrpSpPr", "grpSpPr", "extLst"):
                    continue
                if tag not in ("sp", "pic", "cxnSp"):
                    errors.append(f"{label}: unsupported shape {tag}; I cannot verify its geometry")
                    continue
                transform = shape.find("p:spPr/a:xfrm", NS)
                if transform is None or int(transform.get("rot", "0")) % 21600000:
                    errors.append(f"{label}: missing or rotated geometry")
                    continue
                offset, extent = transform.find("a:off", NS), transform.find("a:ext", NS)
                x, y = int(offset.attrib["x"]), int(offset.attrib["y"])
                w, h = int(extent.attrib["cx"]), int(extent.attrib["cy"])
                if min(x, y, w, h) < 0 or x + w > width or y + h > height:
                    errors.append(f"{label}: shape escapes the slide surface")
                if "".join(shape.itertext()).strip() and shape.find("p:txBody", NS) is not None:
                    if "".join(n.text or "" for n in shape.findall(".//a:t", NS)).strip():
                        if w <= 0 or h <= 0:
                            errors.append(f"{label}: text has an empty frame")
                        frames.append((x, y, x + w, y + h))
            for index, a in enumerate(frames):
                for b in frames[index + 1:]:
                    if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                        errors.append(f"{label}: text-bearing frames overlap")
            notes = [target for kind, target in relations(package, part).values()
                     if kind.endswith("/notesSlide")]
            if len(notes) != 1 or not notes[0]:
                errors.append(f"{label}: I need one internal speaker-notes part")
                continue
            note = xml(package, notes[0])
            bodies = [shape for shape in note.findall(".//p:sp", NS)
                      if any(ph.get("type") == "body" for ph in shape.findall(".//p:ph", NS))]
            if not any("".join(n.text or "" for n in body.findall(".//a:t", NS)).strip() for body in bodies):
                errors.append(f"{label}: speaker notes are empty")
        scan_text(package, errors, "deck")
        return len(slides)


def check_narrative(path, errors):
    with ZipFile(path) as package:
        styles = xml(package, "word/styles.xml")
        levels = {}
        for style in styles.findall("w:style", NS):
            name = style.find("w:name", NS)
            match = re.fullmatch(r"heading ([1-9])", name.get(f"{{{NS['w']}}}val", ""), re.I) if name is not None else None
            if match:
                levels[style.attrib[f"{{{NS['w']}}}styleId"]] = int(match[1])
        root = xml(package, "word/document.xml")
        previous, count = 0, 0
        for paragraph in root.findall(".//w:p", NS):
            style = paragraph.find("w:pPr/w:pStyle", NS)
            level = levels.get(style.get(f"{{{NS['w']}}}val")) if style is not None else None
            if level is None:
                continue
            count += 1
            if level > 6 or level > previous + 1:
                errors.append("narrative: heading levels skip a level or exceed six")
            if not "".join(n.text or "" for n in paragraph.findall(".//w:t", NS)).strip():
                errors.append("narrative: empty heading")
            previous = level
        if not count:
            errors.append("narrative: no headings")
        scan_text(package, errors, "narrative")
        return count


def verify(manifest_path):
    errors = []
    report = {"schema": "nanolang/document-pair-acceptance@1", "accepted": False,
              "visual_qa": "not performed", "errors": errors}
    try:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "nanolang/developer-document-pair@1":
            raise ValueError("I need a document-pair manifest")
        expected = manifest["slides"]
        if type(expected) is not int or expected < 1:
            raise ValueError("I need a positive integer slide count")
        paths = []
        for key in ("local_artifact", "narrative"):
            value = manifest[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"I need the {key} artifact path")
            path = Path(value)
            paths.append(path if path.is_absolute() else manifest_path.parent / path)
        report["slides"] = check_slides(paths[0], expected, errors)
        report["headings"] = check_narrative(paths[1], errors)
    except (OSError, BadZipFile, ET.ParseError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append(f"I could not verify the document pair ({type(error).__name__})")
    report["accepted"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.manifest)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("I accepted the mechanical checks" if report["accepted"] else "I rejected the document pair")
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
