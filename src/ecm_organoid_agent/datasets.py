from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
import mimetypes
from pathlib import Path
import shutil
import zipfile
import tarfile
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx


@dataclass(frozen=True)
class PublicDatasetSpec:
    dataset_id: str
    title: str
    source: str
    url: str
    landing_page: str
    description: str
    tags: tuple[str, ...]
    expected_format: str
    access_mode: str


def curated_public_datasets() -> list[PublicDatasetSpec]:
    return [
        PublicDatasetSpec(
            dataset_id="mendeley_hydrogel_gelma_pegda",
            title="Hydrogel Characterization Data for GelMA and PEGDA",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/cx8cfgrhnn/files/1f87f3e2-a72d-4a8f-8227-428fdc5c7f0f/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/cx8cfgrhnn/1",
            description="Hydrogel characterization package including rheology, compression, and related mechanics measurements for GelMA and PEGDA.",
            tags=("hydrogel", "rheology", "compression", "gelma", "pegda"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_hydrogel_rheology_swelling",
            title="Hydrogel Rheology, Optical Values, and Swelling",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/c3rxvdgbrs/files/8a46f7fb-76dd-42a2-b992-1ea054f95f7e/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/c3rxvdgbrs/1",
            description="Hydrogel rheology, optical, and swelling dataset suitable for calibration priors and benchmark validation.",
            tags=("hydrogel", "rheology", "swelling"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_rcphc1_ma_hydrogel",
            title="Methacrylated Human Recombinant Collagen Peptide Hydrogel Stiffness Dataset",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/nvkd84dzdf/files/8cefa893-5c78-4da6-98d6-3c0bb93e0d07/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/nvkd84dzdf",
            description="Hydrogel stiffness-oriented dataset for a collagen-peptide derived matrix.",
            tags=("hydrogel", "collagen", "stiffness"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_3dtfm_compressibility",
            title="Testing Data for Effects of Hydrogel Compressibility in 3D Traction Force Microscopy",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/nmfxvtbrbd/files/7488db8b-15f7-4ef4-a0c3-96e541e62749/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/nmfxvtbrbd",
            description="Benchmark-like hydrogel compressibility dataset useful for inverse-analysis and identifiability exercises.",
            tags=("hydrogel", "compressibility", "3dtfm", "benchmark"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_gelatin_hydrogel_ink_rheology",
            title="Temperature-Dependent Stress Yielding Behavior of a Gelatin-Based Hydrogel Ink",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/2z25ttxv6b/files/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/2z25ttxv6b",
            description="Rheology-focused dataset for a gelatin-based hydrogel ink with temperature-dependent yielding behavior, useful for extending viscoelastic prior coverage.",
            tags=("hydrogel", "rheology", "gelatin", "yielding", "3d-printing"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_tissue_adhesive_hydrogel_fatigue",
            title="Dataset of Fatigue Tests of Tissue Adhesive Hydrogel",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/vg4f7s667f/files/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/vg4f7s667f/1",
            description="Hydrogel monotonic, static, cyclic, and crack-growth fatigue data, useful for extending mechanics coverage beyond quasi-static stiffness.",
            tags=("hydrogel", "fatigue", "cyclic-loading", "fracture", "mechanics"),
            expected_format="zip",
            access_mode="manual",
        ),
        PublicDatasetSpec(
            dataset_id="mendeley_lost_circulation_hydrogel_rheology",
            title="Optimization of an In-Situ Polymerized and Crosslinked Hydrogel Formulation",
            source="Mendeley Data",
            url="https://data.mendeley.com/public-files/datasets/mrrr7rk5pr/files/file_downloaded",
            landing_page="https://data.mendeley.com/datasets/mrrr7rk5pr",
            description="Hydrogel formulation dataset with time/frequency/stress sweep rheology and DOE analysis, useful for expanding crosslink-sensitive prior coverage.",
            tags=("hydrogel", "rheology", "frequency-sweep", "stress-sweep", "doe"),
            expected_format="zip",
            access_mode="manual",
        ),
    ]


def dataset_workspace(project_dir: Path) -> Path:
    path = project_dir / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_manifest_path(project_dir: Path) -> Path:
    return dataset_workspace(project_dir) / "manifest.json"


def list_public_dataset_specs(*, query: str = "") -> dict[str, Any]:
    normalized = query.strip().lower()
    specs = curated_public_datasets()
    if normalized:
        specs = [
            spec
            for spec in specs
            if normalized in spec.title.lower()
            or normalized in spec.description.lower()
            or any(normalized in tag.lower() for tag in spec.tags)
        ]
    return {
        "count": len(specs),
        "datasets": [asdict(spec) for spec in specs],
    }


def acquire_public_dataset(
    *,
    project_dir: Path,
    dataset_id: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    spec = next((item for item in curated_public_datasets() if item.dataset_id == dataset_id), None)
    if spec is None:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    workspace = dataset_workspace(project_dir)
    dataset_dir = workspace / dataset_id
    raw_dir = dataset_dir / "raw"
    extracted_dir = dataset_dir / "extracted"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    if spec.access_mode == "manual":
        payload = {
            "dataset_id": spec.dataset_id,
            "title": spec.title,
            "source": spec.source,
            "landing_page": spec.landing_page,
            "archive_path": None,
            "dataset_dir": str(dataset_dir),
            "extracted_dir": str(extracted_dir),
            "download_meta": {"mode": "manual"},
            "file_count": 0,
            "files": [],
            "downloaded_at": _utc_now_iso(),
            "status": "manual_download_required",
            "manual_download_instructions": (
                "Open the landing page in a browser, download the source files manually, "
                f"and place them into {raw_dir}. Then run the dataset normalization step."
            ),
        }
        _update_manifest(project_dir, payload)
        return payload

    archive_path = raw_dir / _infer_download_filename(spec)
    download_meta = _download_file(spec.url, archive_path, timeout_seconds=timeout_seconds)
    extracted_files = _extract_if_archive(archive_path, extracted_dir)

    payload = {
        "dataset_id": spec.dataset_id,
        "title": spec.title,
        "source": spec.source,
        "landing_page": spec.landing_page,
        "archive_path": str(archive_path),
        "dataset_dir": str(dataset_dir),
        "extracted_dir": str(extracted_dir),
        "download_meta": download_meta,
        "file_count": len(extracted_files),
        "files": [str(path) for path in extracted_files],
        "downloaded_at": _utc_now_iso(),
        "status": "downloaded",
    }
    _update_manifest(project_dir, payload)
    return payload


def ingest_manual_dataset(
    *,
    project_dir: Path,
    dataset_id: str,
) -> dict[str, Any]:
    spec = next((item for item in curated_public_datasets() if item.dataset_id == dataset_id), None)
    if spec is None:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    workspace = dataset_workspace(project_dir)
    dataset_dir = workspace / dataset_id
    raw_dir = dataset_dir / "raw"
    extracted_dir = dataset_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    if not raw_files:
        raise FileNotFoundError(f"No manually downloaded files found in {raw_dir}")

    extracted_files: list[Path] = []
    for raw_file in raw_files:
        extracted_files.extend(_extract_if_archive(raw_file, extracted_dir))
    extracted_files = sorted({path.resolve() for path in extracted_files})

    payload = {
        "dataset_id": spec.dataset_id,
        "title": spec.title,
        "source": spec.source,
        "landing_page": spec.landing_page,
        "archive_path": None,
        "dataset_dir": str(dataset_dir),
        "extracted_dir": str(extracted_dir),
        "download_meta": {"mode": "manual_ingest"},
        "file_count": len(extracted_files),
        "files": [str(path) for path in extracted_files],
        "downloaded_at": _utc_now_iso(),
        "status": "manual_ingested",
    }
    _update_manifest(project_dir, payload)
    return payload


def auto_register_loose_archives(project_dir: Path) -> list[dict[str, Any]]:
    workspace = dataset_workspace(project_dir)
    registered = []
    for path in sorted(workspace.glob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        dataset_id = _slugify_dataset_name(path.stem)
        dataset_dir = workspace / dataset_id
        raw_dir = dataset_dir / "raw"
        extracted_dir = dataset_dir / "extracted"
        raw_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        extracted_files = _extract_if_archive(target, extracted_dir)
        payload = {
            "dataset_id": dataset_id,
            "title": path.stem,
            "source": "manual_local_upload",
            "landing_page": None,
            "archive_path": str(target),
            "dataset_dir": str(dataset_dir),
            "extracted_dir": str(extracted_dir),
            "download_meta": {"mode": "manual_local_upload"},
            "file_count": len(extracted_files),
            "files": [str(file) for file in extracted_files],
            "downloaded_at": _utc_now_iso(),
            "status": "manual_local_ingested",
        }
        _update_manifest(project_dir, payload)
        registered.append(payload)
    return registered


def normalize_dataset_directory(
    *,
    project_dir: Path,
    dataset_id: str,
) -> dict[str, Any]:
    workspace = dataset_workspace(project_dir)
    dataset_dir = workspace / dataset_id
    extracted_dir = dataset_dir / "extracted"
    normalized_dir = dataset_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in extracted_dir.rglob("*") if path.is_file())
    normalized_rows = [_summarize_dataset_file(path) for path in files]
    calibration_rows = []
    for path in files:
        if path.suffix.lower() == ".xlsx":
            calibration_rows.extend(parse_xlsx_to_calibration_rows(path))
    payload = {
        "dataset_id": dataset_id,
        "normalized_dir": str(normalized_dir),
        "file_count": len(normalized_rows),
        "files": normalized_rows,
        "calibration_row_count": len(calibration_rows),
        "normalized_at": _utc_now_iso(),
        "notes": [
            "This is a first-pass directory normalization, not a row-level mechanics schema conversion.",
            "Use the extracted metadata to choose which xlsx/csv files should be parsed into calibration tables next.",
        ],
    }
    (normalized_dir / "normalized_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if calibration_rows:
        calibration_path = normalized_dir / "calibration_records.csv"
        with calibration_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset_id",
                    "file_name",
                    "sheet_name",
                    "sample_id",
                    "material_family",
                    "measurement_type",
                    "condition",
                    "field",
                    "value",
                ],
            )
            writer.writeheader()
            writer.writerows(calibration_rows)
        payload["calibration_csv"] = str(calibration_path)
    return payload


def _download_file(url: str, destination: Path, *, timeout_seconds: int) -> dict[str, Any]:
    headers = {"User-Agent": "autogen-ecm-organoid-assistant/0.1"}
    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        return {"content_type": content_type, "verify": True}
    except httpx.HTTPError:
        pass

    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True, verify=False) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    return {"content_type": content_type, "verify": False}


def _extract_if_archive(archive_path: Path, extracted_dir: Path) -> list[Path]:
    extracted = False
    if archive_path.suffix.lower() == ".zip" and zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted_dir)
        extracted = True
    elif archive_path.suffix.lower() in {".tgz", ".gz"} or archive_path.name.endswith(".tar.gz"):
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as archive:
                archive.extractall(extracted_dir)
            extracted = True

    if not extracted:
        target = extracted_dir / archive_path.name
        shutil.copy2(archive_path, target)
    return sorted(path for path in extracted_dir.rglob("*") if path.is_file())


def _infer_download_filename(spec: PublicDatasetSpec) -> str:
    parsed = urlparse(spec.url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return f"{spec.dataset_id}{suffix}"
    if spec.expected_format == "zip":
        return f"{spec.dataset_id}.zip"
    if spec.expected_format == "tgz":
        return f"{spec.dataset_id}.tgz"
    content_type, _ = mimetypes.guess_type(spec.url)
    guessed = mimetypes.guess_extension(content_type or "") if content_type else None
    return f"{spec.dataset_id}{guessed or '.bin'}"


def _update_manifest(project_dir: Path, payload: dict[str, Any]) -> None:
    manifest_path = dataset_manifest_path(project_dir)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"datasets": []}

    datasets = [row for row in manifest.get("datasets", []) if row.get("dataset_id") != payload["dataset_id"]]
    datasets.append(payload)
    manifest["datasets"] = datasets
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize_dataset_file(path: Path) -> dict[str, Any]:
    rel_parts = path.parts
    stem = path.stem
    return {
        "path": str(path),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "material_guess": _material_guess_from_name(stem),
        "measurement_guess": _measurement_guess_from_name(stem, rel_parts),
        "condition_guess": _condition_guess_from_name(stem),
    }


def parse_xlsx_to_calibration_rows(path: Path) -> list[dict[str, Any]]:
    workbook = read_xlsx_workbook(path)
    rows: list[dict[str, Any]] = []
    dataset_id = _slugify_dataset_name(path.parents[2].name if len(path.parents) >= 3 else path.parent.name)
    for sheet_name, table in workbook.items():
        rows.extend(
            _sheet_to_calibration_rows(
                dataset_id=dataset_id,
                path=path,
                sheet_name=sheet_name,
                table=table,
            )
        )
    return rows


def read_xlsx_workbook(path: Path) -> dict[str, list[list[str]]]:
    ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(t.text or "" for t in item.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                for item in shared_root
            ]

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root
            if rel.tag.endswith("Relationship")
        }

        workbook: dict[str, list[list[str]]] = {}
        for sheet in workbook_root.find("x:sheets", ns):
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{ns['r']}}}id"]
            target = rel_map[rel_id]
            sheet_path = f"xl/{target}" if not target.startswith("xl/") else target
            sheet_root = ET.fromstring(archive.read(sheet_path))
            table: list[list[str]] = []
            for row in sheet_root.find("x:sheetData", ns):
                parsed_cells = []
                for cell in row:
                    value_node = cell.find("x:v", ns)
                    value = value_node.text if value_node is not None else ""
                    if cell.attrib.get("t") == "s" and value:
                        value = shared_strings[int(value)]
                    parsed_cells.append((cell.attrib.get("r", ""), str(value).strip()))
                if not parsed_cells:
                    continue
                last_col = max(_column_index(ref) for ref, _ in parsed_cells)
                values = [""] * (last_col + 1)
                for ref, value in parsed_cells:
                    values[_column_index(ref)] = value
                table.append(values)
            workbook[sheet_name] = table
        return workbook


def _sheet_to_calibration_rows(
    *,
    dataset_id: str,
    path: Path,
    sheet_name: str,
    table: list[list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header_index = _detect_header_row(table)
    if header_index is None:
        return rows
    header = [_clean_header_name(value) for value in table[header_index]]
    measurement_type = _measurement_guess_from_name(sheet_name, tuple(path.parts))
    material_family = _material_guess_from_name(path.name)

    for row in table[header_index + 1 :]:
        if not any(cell.strip() for cell in row):
            continue
        sample_id = row[0].strip() if row and row[0].strip() else path.stem
        condition = _condition_guess_from_name(sample_id or path.stem)
        for idx, value in enumerate(row):
            if idx >= len(header):
                continue
            field = header[idx]
            if not field or not value.strip():
                continue
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "file_name": path.name,
                    "sheet_name": sheet_name,
                    "sample_id": sample_id,
                    "material_family": material_family,
                    "measurement_type": measurement_type,
                    "condition": json.dumps(condition, ensure_ascii=False),
                    "field": field,
                    "value": value,
                }
            )
    return rows


def _detect_header_row(table: list[list[str]]) -> int | None:
    for idx, row in enumerate(table[:8]):
        normalized = " ".join(cell.strip().lower() for cell in row if cell.strip())
        if any(keyword in normalized for keyword in ("sample", "density", "modulus", "weight", "strain", "stress")):
            return idx
    return None


def _clean_header_name(value: str) -> str:
    import re

    return re.sub(r"\s+", "_", value.strip().lower()).strip("_")


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    result = 0
    for char in letters:
        result = result * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(result - 1, 0)


def _material_guess_from_name(name: str) -> str:
    lowered = name.lower()
    if "gelma" in lowered:
        return "GelMA"
    if "pegda" in lowered:
        return "PEGDA"
    return "unknown"


def _measurement_guess_from_name(name: str, parts: tuple[str, ...]) -> str:
    lowered = name.lower()
    joined = " ".join(part.lower() for part in parts)
    if "compression" in lowered or "compression" in joined:
        return "compression"
    if "characterization" in lowered:
        return "summary_characterization"
    return "unknown"


def _condition_guess_from_name(name: str) -> dict[str, Any]:
    lowered = name.lower()
    condition: dict[str, Any] = {}
    for token in lowered.replace("_", " ").split():
        if token.endswith("p") and token[:-1].isdigit():
            condition.setdefault("percentage_tokens", []).append(token)
        if token.endswith("s") and token[:-1].isdigit():
            condition.setdefault("time_seconds", []).append(token)
    return condition


def _slugify_dataset_name(name: str) -> str:
    import re

    normalized = re.sub(r"[^\w.-]+", "_", name.strip().lower()).strip("._")
    return normalized or "dataset"


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
