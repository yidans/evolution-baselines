"""Read-only access to a ``controller-hidden-benchmark-pack.v1`` link-prediction pack."""
from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

INSTANCE_SCHEMA = "local-link-prediction-instance.v1"
SOLUTION_SCHEMA = "local-link-prediction-solution.v1"
CONFIG_SCHEMA = "local-link-prediction-config.v1"


@dataclass(frozen=True)
class InstanceSpec:
    scope: str  # "public" or "hidden"
    family_id: str
    instance_id: str
    instance_seed: int
    role: str  # "quality" or "runtime_envelope"
    config_path: Path
    bundle: Path
    noise_band: float = 0.0
    scale: str | None = None

    @property
    def key(self) -> str:
        """Hidden families reuse instance ids (inst_001...), so runs are keyed per family."""
        return f"{self.family_id}/{self.instance_id}"

    def qualification_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "family_id": self.family_id,
            "instance_id": self.instance_id,
            "instance_seed": self.instance_seed,
            "role": self.role,
            "noise_band": self.noise_band,
        }
        if self.scale:
            spec["scale"] = self.scale
        return spec


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


class LinkPredictionPack:
    """The v2 pack: public development bundle, controller bundle, hidden registry."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "public_contract.json").is_file():
            raise FileNotFoundError(f"not a benchmark pack: {self.root}")
        self.public_contract: dict[str, Any] = _read(self.root / "public_contract.json")
        self.bundle = self.root / "benchmark_bundle"
        self.dev_bundle = self.root / "development_benchmark_bundle"
        self.dev_manifest: dict[str, Any] = _read(self.dev_bundle / "benchmark_manifest.json")

    @cached_property
    def hidden_registry(self) -> dict[str, Any]:
        """Read private inputs only for an explicit operator diagnostic."""
        return _read(self.root / "hidden_registry.json")

    # ----------------------------------------------------------------- contract
    @property
    def contract_id(self) -> str:
        return str(self.public_contract["contract_id"])

    @property
    def baseline_ids(self) -> list[str]:
        return [str(name) for name in self.public_contract["baselines"]]

    @property
    def seeds(self) -> list[int]:
        return [int(seed) for seed in self.public_contract["seed_policy"]["seeds"]]

    @property
    def timeout_seconds(self) -> float:
        return float(self.public_contract["compute_budget"]["timeout_seconds_per_run"])

    @property
    def instance_required(self) -> list[str]:
        return [str(field) for field in self.public_contract["interface"]["instance_required"]]

    @property
    def primary_metric(self) -> str:
        return str(self.public_contract["metrics"]["primary"])

    @property
    def minimum_relative_improvement(self) -> float:
        return float(self.public_contract["metrics"]["minimum_relative_improvement"])

    @property
    def minimum_win_rate(self) -> float:
        return float(self.public_contract["metrics"].get("minimum_win_rate", 0.5))

    def required_family_count(self) -> int:
        """Mirror ``ProjectQualification``: all but at most one family, never fewer than two."""
        families = len(self.dev_manifest["development_families"])
        if self.public_contract.get("benchmark_bundle_origin") == "scientist_designed":
            return families
        return max(2, families - 1)

    # ---------------------------------------------------------------- instances
    def public_specs(self) -> list[InstanceSpec]:
        specs: list[InstanceSpec] = []
        for family in self.dev_manifest["development_families"]:
            for item in family["instances"]:
                specs.append(
                    InstanceSpec(
                        scope="public",
                        family_id=str(family["id"]),
                        instance_id=str(item["id"]),
                        instance_seed=int(item["instance_seed"]),
                        role=str(item["role"]),
                        config_path=self.dev_bundle / item["config"],
                        bundle=self.dev_bundle,
                        noise_band=float(item.get("noise_band", 0.0)),
                        scale=item.get("scale"),
                    )
                )
        return specs

    def hidden_specs(self) -> list[InstanceSpec]:
        specs: list[InstanceSpec] = []
        for family in self.hidden_registry["families"]:
            for item in family["instances"]:
                specs.append(
                    InstanceSpec(
                        scope="hidden",
                        family_id=str(family["id"]),
                        instance_id=str(item["instance_id"]),
                        instance_seed=int(item["instance_seed"]),
                        role="quality",
                        config_path=self.root / item["instance_config_path"],
                        bundle=self.bundle,
                    )
                )
        return specs

    def hidden_family_tags(self) -> dict[str, str]:
        return {
            str(family["id"]): str(family.get("private_family_tag") or family["id"])
            for family in self.hidden_registry["families"]
        }

    def load_instance(self, spec: InstanceSpec) -> dict[str, Any]:
        """Copy of the generator's logic: a precomputed split, never resampled."""
        config = _read(spec.config_path)
        if config.get("schema_version") != CONFIG_SCHEMA:
            raise ValueError(f"wrong configuration schema in {spec.config_path}")
        if int(config["split_seed"]) != spec.instance_seed:
            raise ValueError(f"instance seed differs from precomputed split seed: {spec}")
        path = (spec.bundle / config["split_path"]).resolve()
        path.relative_to(spec.bundle.resolve())
        instance = _read(path)
        if instance.get("schema_version") != INSTANCE_SCHEMA:
            raise ValueError(f"wrong instance schema in {path}")
        return instance

    def project(self, instance: dict[str, Any]) -> dict[str, Any]:
        """What a method may see: the controller projection to ``instance_required``."""
        return {field: instance[field] for field in self.instance_required}

    # ------------------------------------------------------------------ scoring
    @cached_property
    def score(self) -> Callable[[dict[str, Any], Any], dict[str, float]]:
        """The pack's own role-blind scorer, loaded from ``benchmark_bundle/scorers/score.py``."""
        return runpy.run_path(str(self.dev_bundle / "scorers" / "score.py"))["score"]

    def baseline_source(self, baseline_id: str) -> str:
        path = self.dev_bundle / "baselines" / baseline_id / "solution.py"
        if not path.is_file():
            raise FileNotFoundError(f"unknown pack baseline {baseline_id!r}")
        return path.read_text()

    def public_reference_dir(self) -> Path:
        candidates = sorted(p for p in (self.root / "public").glob("*/reference") if p.is_dir())
        if not candidates:
            raise FileNotFoundError("pack has no public reference directory")
        return candidates[0]

    def operator_heuristics_source(self) -> str:
        """The sender's full heuristic file; controller-private, never shown to a model."""
        return (self.root / "operator_sources" / "baselines.py").read_text()
