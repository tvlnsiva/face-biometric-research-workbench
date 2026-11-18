"""Typed data contracts and error envelopes for the Research Workbench API (`v1`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class WorkbenchError(Exception):
    """Structured application error carrying HTTP status code and machine-readable code."""

    def __init__(
        self,
        message: str,
        code: str = "WORKBENCH_ERROR",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = int(status_code)
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative metadata for a single algorithm parameter rendered dynamically by the UI."""

    name: str
    label: str
    control_type: str  # "slider" | "number" | "select" | "boolean" | "multiselect" | "object"
    value_type: str    # "float" | "int" | "string" | "bool" | "list" | "dict"
    default: Any
    description: str
    impact: str        # "accuracy" | "performance" | "both"
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    options: list[dict[str, Any]] | None = None
    visible_when: dict[str, Any] | None = None
    read_only: bool = False
    experimental_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlgorithmSpec:
    """Authoritative metadata record for a registered research algorithm or model."""

    algorithm_id: str
    name: str
    category: str  # "detection" | "landmark_alignment" | "quality" | "recognition" | "matching" | "retrieval"
    description: str
    paper_reference: str
    source_implementation: str
    model_id: str | None
    checkpoint_path: str | None
    model_sha256: str | None
    input_requirements: dict[str, Any]
    output_schema: dict[str, Any]
    supported_devices: list[str]
    supported_precisions: list[str]
    supported_runtimes: list[str]
    dynamic_batch_supported: bool
    license_provenance: dict[str, Any]
    availability_status: str  # "AVAILABLE" | "MISSING_LOCAL_WEIGHTS"
    parameters: list[ParameterSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parameters"] = [p.to_dict() for p in self.parameters]
        return d


@dataclass
class PipelineValidationResult:
    """Pre-execution validation outcome for multi-stage pipeline configurations."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    """Complete reproducible record for a Workbench or historical experiment."""

    experiment_id: str
    name: str
    source_origin: str  # "historical_readonly" | "workbench_run" | "saved_configuration"
    milestone: str
    category: str
    timestamp_utc: str
    status: str         # "completed" | "failed" | "cancelled" | "saved_config"
    dataset_id: str
    dataset_split: str
    dataset_manifest_sha256: str | None
    algorithms: dict[str, Any]
    model_hashes: dict[str, str | None]
    parameters: dict[str, Any]
    software_environment: dict[str, Any]
    hardware: dict[str, Any]
    execution_provider: str
    precision: str
    batch_size: int
    seed: int | None
    metrics: dict[str, Any]
    curves: dict[str, Any] = field(default_factory=dict)
    stage_graph: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    notes: str = ""
    error_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
