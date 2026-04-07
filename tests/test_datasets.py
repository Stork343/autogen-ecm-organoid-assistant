from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecm_organoid_agent.datasets import (
    acquire_public_dataset,
    curated_public_datasets,
    dataset_manifest_path,
    dataset_workspace,
    ingest_manual_dataset,
    list_public_dataset_specs,
    normalize_dataset_directory,
    parse_xlsx_to_calibration_rows,
    read_xlsx_workbook,
)


class DatasetWorkflowTests(unittest.TestCase):
    def test_curated_public_datasets_returns_specs(self) -> None:
        specs = curated_public_datasets()
        self.assertGreater(len(specs), 0)
        self.assertTrue(specs[0].dataset_id)

    def test_list_public_dataset_specs_filters_by_query(self) -> None:
        payload = list_public_dataset_specs(query="pegda")
        self.assertIn("datasets", payload)
        self.assertGreaterEqual(payload["count"], 1)

    def test_acquire_public_dataset_registers_manual_download_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            workspace = dataset_workspace(project_dir)
            self.assertTrue(workspace.exists())

            payload = acquire_public_dataset(
                project_dir=project_dir,
                dataset_id=curated_public_datasets()[0].dataset_id,
                timeout_seconds=180,
            )
            self.assertEqual(payload["dataset_id"], curated_public_datasets()[0].dataset_id)
            self.assertEqual(payload["status"], "manual_download_required")
            manifest = json.loads(dataset_manifest_path(project_dir).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["datasets"]), 1)

    def test_ingest_manual_dataset_expands_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            dataset_dir = dataset_workspace(project_dir) / curated_public_datasets()[0].dataset_id
            raw_dir = dataset_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.csv").write_text("time,stress\n0,0\n1,1\n", encoding="utf-8")

            payload = ingest_manual_dataset(project_dir=project_dir, dataset_id=curated_public_datasets()[0].dataset_id)
            self.assertEqual(payload["status"], "manual_ingested")
            self.assertGreaterEqual(payload["file_count"], 1)

    def test_read_xlsx_workbook_and_normalize_manual_dataset(self) -> None:
        project_dir = PROJECT_ROOT
        xlsx_path = project_dir / "datasets" / "hydrogel_characterization_data" / "extracted" / "Hydrogel Characterization Data" / "GelMA_Characterization_Data.xlsx"
        workbook = read_xlsx_workbook(xlsx_path)
        self.assertIn("YoungsModulus", workbook)
        rows = parse_xlsx_to_calibration_rows(xlsx_path)
        self.assertGreater(len(rows), 0)
        payload = normalize_dataset_directory(project_dir=project_dir, dataset_id="hydrogel_characterization_data")
        self.assertGreater(payload["calibration_row_count"], 0)
        self.assertIn("calibration_csv", payload)


if __name__ == "__main__":
    unittest.main()
