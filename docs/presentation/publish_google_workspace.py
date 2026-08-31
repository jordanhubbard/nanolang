#!/usr/bin/env python3
"""Publish the overview pair to Google Workspace without persisting tokens.

Obtains a bearer token with `gcloud auth print-access-token` immediately before
use. The token stays in process memory and is never printed, logged, or written
to a receipt. Updates the existing Slides resource in place and creates a Google
Doc from the local .docx when no Doc ID is supplied.
"""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
SLIDES_ID = "1V9mt1JpEst_2ucC0eJIIw7dff64MrtFRfut8MSiukeE"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
SLIDES_GOOGLE = "application/vnd.google-apps.presentation"
DOCS_GOOGLE = "application/vnd.google-apps.document"
DRIVE = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3"


def _ssl() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _token() -> str:
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "gcloud auth print-access-token failed").strip()
        raise SystemExit(message)
    token = result.stdout.strip()
    if not token or token.count(".") < 2 and not token.startswith("ya29."):
        # Still accept opaque tokens; refuse empty.
        if not token:
            raise SystemExit("gcloud returned an empty access token")
    return token


def _request(
    token: str,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    raw: bool = False,
) -> tuple[int, dict[str, str], bytes]:
    request_headers = {"Authorization": f"Bearer {token}"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, context=_ssl()) as response:
            body = response.read()
            return response.status, dict(response.headers), body
    except urllib.error.HTTPError as error:
        detail = error.read()[:800]
        raise SystemExit(f"Drive API {method} {url.split('?', 1)[0]} failed: {error.code} {detail!r}") from None


def _json(token: str, method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=UTF-8"
    status, _headers, body = _request(token, method, url, headers=headers, data=data)
    if not body:
        return {"http_status": status}
    parsed = json.loads(body.decode("utf-8"))
    parsed["http_status"] = status
    return parsed


def _resumable_update(token: str, file_id: str, path: Path, source_mime: str) -> dict:
    meta_url = (
        f"{UPLOAD}/files/{urllib.parse.quote(file_id)}"
        "?uploadType=resumable&supportsAllDrives=true"
    )
    start_headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": source_mime,
        "X-Upload-Content-Length": str(path.stat().st_size),
    }
    start_data = json.dumps({}).encode("utf-8")
    _status, response_headers, _body = _request(
        token, "PATCH", meta_url, headers=start_headers, data=start_data
    )
    location = response_headers.get("Location") or response_headers.get("location")
    if not location:
        raise SystemExit("Drive resumable update did not return a Location header")
    media = path.read_bytes()
    _status, _headers, body = _request(
        token,
        "PUT",
        location,
        headers={"Content-Type": source_mime, "Content-Length": str(len(media))},
        data=media,
    )
    return json.loads(body.decode("utf-8")) if body else {}


def _resumable_create(
    token: str, *, name: str, parents: list[str], source_mime: str, google_mime: str, path: Path
) -> dict:
    meta_url = f"{UPLOAD}/files?uploadType=resumable&supportsAllDrives=true"
    metadata = {"name": name, "mimeType": google_mime, "parents": parents}
    start_headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": source_mime,
        "X-Upload-Content-Length": str(path.stat().st_size),
    }
    _status, response_headers, _body = _request(
        token, "POST", meta_url, headers=start_headers, data=json.dumps(metadata).encode("utf-8")
    )
    location = response_headers.get("Location") or response_headers.get("location")
    if not location:
        raise SystemExit("Drive resumable create did not return a Location header")
    media = path.read_bytes()
    _status, _headers, body = _request(
        token,
        "PUT",
        location,
        headers={"Content-Type": source_mime, "Content-Length": str(len(media))},
        data=media,
    )
    return json.loads(body.decode("utf-8")) if body else {}


def _ensure_org_reader(token: str, file_id: str) -> dict:
    listed = _json(
        token,
        "GET",
        f"{DRIVE}/files/{urllib.parse.quote(file_id)}/permissions"
        "?fields=permissions(id,type,domain,role,allowFileDiscovery)&supportsAllDrives=true",
    )
    permissions = listed.get("permissions") or []
    already = any(
        item.get("type") == "domain"
        and item.get("domain") == "nvidia.com"
        and item.get("role") == "reader"
        for item in permissions
        if isinstance(item, dict)
    )
    if not already:
        _json(
            token,
            "POST",
            f"{DRIVE}/files/{urllib.parse.quote(file_id)}/permissions?supportsAllDrives=true",
            {
                "type": "domain",
                "domain": "nvidia.com",
                "role": "reader",
                "allowFileDiscovery": False,
            },
        )
        listed = _json(
            token,
            "GET",
            f"{DRIVE}/files/{urllib.parse.quote(file_id)}/permissions"
            "?fields=permissions(id,type,domain,role,allowFileDiscovery)&supportsAllDrives=true",
        )
    public = [
        {
            "type": item.get("type"),
            "domain": item.get("domain"),
            "role": item.get("role"),
            "allowFileDiscovery": item.get("allowFileDiscovery"),
        }
        for item in (listed.get("permissions") or [])
        if isinstance(item, dict)
    ]
    return {"permissions": public}


def _export(token: str, file_id: str, *, kind: str, dest: Path) -> int:
    if kind == "pptx":
        url = f"https://docs.google.com/presentation/d/{urllib.parse.quote(file_id)}/export/pptx"
    elif kind == "docx":
        url = (
            f"{DRIVE}/files/{urllib.parse.quote(file_id)}/export?"
            + urllib.parse.urlencode({"mimeType": DOCX_MIME})
        )
    else:
        raise SystemExit(f"unknown export kind {kind}")
    _status, _headers, body = _request(token, "GET", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return len(body)


def _pptx_pages(path: Path) -> tuple[int, int]:
    with zipfile.ZipFile(path) as archive:
        slides = [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
        notes = [name for name in archive.namelist() if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
    return len(slides), len(notes)


def _docx_headings(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    count = 0
    for para in root.iter(f"{w}p"):
        style = para.find(f"{w}pPr/{w}pStyle")
        if style is not None:
            value = style.attrib.get(f"{w}val", "")
            if value.lower().startswith("heading"):
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slides-id", default="", help="authorized Google Slides file ID; omit to create one")
    parser.add_argument("--doc-id", default="", help="update this Google Doc in place when set")
    parser.add_argument(
        "--pptx",
        default=str(HERE / "nanolang-developer-overview.pptx"),
    )
    parser.add_argument(
        "--docx",
        default=str(HERE / "nanolang-developer-overview.docx"),
    )
    args = parser.parse_args()
    pptx = Path(args.pptx)
    docx = Path(args.docx)
    if not pptx.is_file() or not docx.is_file():
        raise SystemExit("local PPTX and DOCX must both exist before publication")

    token = _token()
    slides_meta = {}
    if args.slides_id:
        slides_meta = _json(
            token, "GET",
            f"{DRIVE}/files/{urllib.parse.quote(args.slides_id)}"
            "?fields=id,name,mimeType,parents,version,capabilities/canEdit&supportsAllDrives=true",
        )
        if slides_meta.get("mimeType") != SLIDES_GOOGLE:
            raise SystemExit(f"slides resource is not a Google presentation: {slides_meta.get('mimeType')}")
        if not (slides_meta.get("capabilities") or {}).get("canEdit"):
            raise SystemExit("authenticated principal cannot edit the existing Slides resource")
        updated_slides = _resumable_update(token, args.slides_id, pptx, PPTX_MIME)
        slides_id = args.slides_id
        parents = [str(item) for item in (slides_meta.get("parents") or []) if item]
    else:
        updated_slides = _resumable_create(
            token, name="NanoLang Developer Overview (3.5 edition)", parents=[],
            source_mime=PPTX_MIME, google_mime=SLIDES_GOOGLE, path=pptx,
        )
        slides_id = str(updated_slides.get("id") or "")
        if not slides_id:
            raise SystemExit("Drive create did not return a Slides file id")
        parents = [str(item) for item in (updated_slides.get("parents") or []) if item]
    slides_access = {"publication": "owner-authenticated"}

    if args.doc_id:
        updated_doc = _resumable_update(token, args.doc_id, docx, DOCX_MIME)
        doc_id = args.doc_id
        renamed = _json(
            token,
            "PATCH",
            f"{DRIVE}/files/{urllib.parse.quote(doc_id)}?supportsAllDrives=true",
            {"name": "NanoLang Developer Narrative (3.5 edition)"},
        )
        if renamed.get("name"):
            updated_doc["name"] = renamed["name"]
    else:
        updated_doc = _resumable_create(
            token,
            name="NanoLang Developer Narrative (3.5 edition)",
            parents=parents,
            source_mime=DOCX_MIME,
            google_mime=DOCS_GOOGLE,
            path=docx,
        )
        doc_id = str(updated_doc.get("id") or "")
        if not doc_id:
            raise SystemExit("Drive create did not return a document id")
    doc_access = _ensure_org_reader(token, doc_id)

    export_root = HERE.parents[1] / "_build" / "nanolang-developer-overview"
    exported_pptx = export_root / "published-export.pptx"
    exported_docx = export_root / "published-export.docx"
    pptx_bytes = _export(token, slides_id, kind="pptx", dest=exported_pptx)
    docx_bytes = _export(token, doc_id, kind="docx", dest=exported_docx)
    local_slides, local_notes = _pptx_pages(pptx)
    exported_slides, exported_notes = _pptx_pages(exported_pptx)
    local_headings = _docx_headings(docx)
    exported_headings = _docx_headings(exported_docx)

    receipt = {
        "slides_id": slides_id,
        "slides_url": f"https://docs.google.com/presentation/d/{slides_id}/edit",
        "slides_version": updated_slides.get("version") or slides_meta.get("version"),
        "slides_mime": updated_slides.get("mimeType") or SLIDES_GOOGLE,
        "doc_id": doc_id,
        "doc_url": f"https://docs.google.com/document/d/{doc_id}/edit",
        "doc_name": updated_doc.get("name"),
        "exported_pptx_bytes": pptx_bytes,
        "exported_docx_bytes": docx_bytes,
        "local_slides": local_slides,
        "exported_slides": exported_slides,
        "local_notes": local_notes,
        "exported_notes": exported_notes,
        "local_headings": local_headings,
        "exported_headings": exported_headings,
        "slides_access": slides_access,
        "doc_access": doc_access,
    }
    receipt_path = export_root / "publish-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    json.dump(receipt, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if exported_slides != local_slides or exported_notes != local_notes:
        raise SystemExit("exported Slides page or notes count does not match the local PPTX")
    if exported_headings < 1:
        raise SystemExit("exported Google Doc has no headings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
