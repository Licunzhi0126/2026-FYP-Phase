"""Data contract shared by V6/V7. Arrays are latent truth unless named counts."""
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

@dataclass
class DatasetView:
    version: str
    source_path: Path
    p_true: pd.DataFrame
    m_true: pd.DataFrame
    genes: pd.DataFrame
    elements: pd.DataFrame
    p_element_state: pd.DataFrame
    m_element_state: pd.DataFrame
    p_grn: pd.DataFrame
    m_grn: pd.DataFrame
    regions: pd.DataFrame | None = None
    capabilities: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

@dataclass
class AuditFinding:
    module: str
    name: str
    status: str
    observation: str
    interpretation: str
    # 'observe' != validated biological pass/fail
    severity: str = "info"
