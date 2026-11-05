"""Central configuration: paths and verified source provenance.

Keeping provenance (URL, pinned commit, checksum) in code means every run can
prove which exact bytes it ingested.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Source A: IBM Telco Customer Churn (verified 2026-11-04) -----------------
IBM_REPO = "https://github.com/IBM/telco-customer-churn-on-icp4d"
IBM_COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"  # master HEAD; repo archived
IBM_CSV_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{IBM_COMMIT}/data/Telco-Customer-Churn.csv"
)
IBM_CSV_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
IBM_CSV_BYTES = 970_457
IBM_LICENSE = "Apache-2.0 (repository license)"


@dataclass(frozen=True)
class Paths:
    """All filesystem locations, rooted at a data directory.

    Tests pass a tmp directory so they never touch real data.
    """

    data_dir: Path

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def ibm_csv(self) -> Path:
        return self.raw / "ibm" / "Telco-Customer-Churn.csv"

    @property
    def profiles_json(self) -> Path:
        return self.raw / "profiles" / "customer_profiles.json"

    @property
    def events_dir(self) -> Path:
        return self.raw / "events"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def curated(self) -> Path:
        return self.processed / "curated"

    @property
    def features(self) -> Path:
        """Phase 2 outputs: split files, fitted preprocessor, feature report."""
        return self.processed / "features"

    @property
    def sqlite_db(self) -> Path:
        return self.processed / "churn.db"

    @property
    def summary_md(self) -> Path:
        return self.processed / "ingestion_summary.md"

    @property
    def summary_json(self) -> Path:
        return self.processed / "ingestion_summary.json"


DEFAULT_PATHS = Paths(PROJECT_ROOT / "data")
