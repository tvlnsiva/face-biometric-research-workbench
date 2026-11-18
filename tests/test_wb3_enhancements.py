"""Automated Regression & Parity Verification Suite for Milestone WB3 (`tests/test_wb3_enhancements.py`).

Covers:
  - P-1: Global Cross-Stage Configuration Delta & Two-Run Comparison (independent Python vs. JS parity
    across all 5 stages, parameter types, conditional `visible_when`, locked params, disabled stages,
    algorithm switching, and incompatible-run warnings).
  - P-2: Genuine/Impostor Score Distribution & Threshold Explorer (all 12 boundary/tie/zero-denominator
    fixtures cross-checked between Python and Node.js, real LFW stratified verification pipeline execution,
    strict calibration vs. held-out test partition separation, legacy backward compatibility, and JSON/CSV/YAML
    export preservation).
  - P-3: Continuous Scientific Image Overlay Opacity (0%, 25%, 50%, 75%, 100% compositing parity and
    Quality Lab landmark/box viewer metadata contract).
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.common.config_loader import get_project_root
from src.workbench.datasets_io import DatasetCatalog
from src.workbench.executors import WorkbenchExecutionEngine, compute_empirical_operating_point
from src.workbench.history_store import ExperimentHistoryStore
from src.workbench.job_runner import ExperimentJobRunner
from src.workbench.model_manager import ModelResourceManager
from src.workbench.registry import get_full_registry_payload


def _run_node_json(js_code: str) -> Any:
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js binary not on PATH")
    proc = subprocess.run(
        [node_bin, "--input-type=module", "-"],
        input=js_code,
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node eval failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    return json.loads(proc.stdout.strip())


def _python_expected_stage_modifications(
    algo_spec: dict[str, Any],
    values: dict[str, Any],
    stage_enabled: bool = True,
) -> tuple[int, list[str]]:
    if not stage_enabled:
        return 0, []
    params = algo_spec.get("parameters", [])
    defaults = {p["name"]: p["default"] for p in params}
    mod_names: list[str] = []
    for p in params:
        if p.get("read_only"):
            continue
        vw = p.get("visible_when")
        if vw:
            visible = True
            for dep_k, exp_v in vw.items():
                actual_dep = values.get(dep_k, defaults.get(dep_k))
                if actual_dep != exp_v:
                    visible = False
                    break
            if not visible:
                continue
        cur = values.get(p["name"], p["default"])
        if p["value_type"] == "float":
            if abs(float(cur) - float(p["default"])) > 1e-9:
                mod_names.append(p["name"])
        elif p["value_type"] == "int":
            if int(round(float(cur))) != int(round(float(p["default"]))):
                mod_names.append(p["name"])
        elif p["value_type"] == "bool":
            if bool(cur) is not bool(p["default"]):
                mod_names.append(p["name"])
        else:
            if cur != p["default"]:
                mod_names.append(p["name"])
    return len(mod_names), mod_names


def test_wb3_p1_configuration_delta_and_two_run_comparison_parity():
    reg = get_full_registry_payload()
    algos_by_id = {a["algorithm_id"]: a for a in reg["algorithms"]}

    # Python independent reference calculation across multiple scenarios
    scenarios = {
        "default_clean": {
            "detectorId": "yunet_2023mar",
            "detectorParamsByModel": {"yunet_2023mar": {}},
            "alignmentParams": {},
            "qualityParams": {},
            "recognizerId": "adaface_ir50_webface4m",
            "recognizerParamsByModel": {"adaface_ir50_webface4m": {}},
            "retrievalIndexId": "ivf_pq_index",
            "retrievalParamsByIndex": {"ivf_pq_index": {}},
        },
        "five_stages_one_mod_each": {
            "detectorId": "yunet_2023mar",
            "detectorParamsByModel": {"yunet_2023mar": {"score_threshold": 0.77}},
            "alignmentParams": {"transform_mode": "affine_least_squares_6dof"},
            "qualityParams": {"reject_quantile": 0.20},
            "recognizerId": "adaface_ir50_webface4m",
            "recognizerParamsByModel": {"adaface_ir50_webface4m": {"flip_tta": True}},
            "retrievalIndexId": "ivf_pq_index",
            "retrievalParamsByIndex": {"ivf_pq_index": {"nprobe": 32}},
        },
        "multi_mod_and_hidden_conditional": {
            "detectorId": "scrfd_500m_research",
            "detectorParamsByModel": {
                "scrfd_500m_research": {
                    "max_dim": 640,
                    "canvas_mode": "stride32_tight",
                    "nms_mode": "hard_greedy",
                    "soft_nms_sigma": 0.22,  # hidden because nms_mode == hard_greedy!
                }
            },
            "alignmentParams": {"interpolation": "cubic", "zoom_factor": 0.95},
            "qualityParams": {},
            "recognizerId": "mobilefacenet_w600k_research",
            "recognizerParamsByModel": {"mobilefacenet_w600k_research": {"batch_size": 16, "l2_normalize": False}},
            "retrievalIndexId": "hnsw_index",
            "retrievalParamsByIndex": {"hnsw_index": {"ef_search": 96}},
        },
    }

    js_script = f"""
    import {{ computeCrossStageDelta, compareTwoExperimentConfigs }} from './frontend/js/components/config_delta.js';
    const algos = {json.dumps(algos_by_id)};
    const scenarios = {json.dumps(scenarios)};
    const out = {{}};
    for (const [k, st] of Object.entries(scenarios)) {{
      out[k] = computeCrossStageDelta(algos, st, {{ scope: 'global' }});
    }}
    const cmp = compareTwoExperimentConfigs(
      {{
        experiment_id: 'EXP-WB-TEST-A',
        category: 'pipeline',
        dataset_id: 'lfw_research_split',
        dataset_split: 'Calib 1-5 -> Test 6-10',
        parameters: {{
          detection: {{ algorithm_id: 'yunet_2023mar', params: {{ score_threshold: 0.60, top_k: 5000 }} }},
          alignment: {{ algorithm_id: 'five_point_canonical_aligner', params: {{ transform_mode: 'similarity_umeyama_4dof' }} }},
          recognition: {{ algorithm_id: 'adaface_ir50_webface4m', params: {{ precision: 'fp16', flip_tta: false }} }}
        }}
      }},
      {{
        experiment_id: 'EXP-WB-TEST-B',
        category: 'pipeline',
        dataset_id: 'nasa_p11_verification',
        dataset_split: '120 smoke pairs',
        parameters: {{
          detection: {{ algorithm_id: 'scrfd_500m_research', params: {{ score_threshold: 0.50, canvas_mode: 'stride32_tight' }} }},
          alignment: {{ algorithm_id: 'five_point_canonical_aligner', params: {{ transform_mode: 'similarity_umeyama_4dof' }} }},
          recognition: {{ algorithm_id: 'adaface_ir50_webface4m', params: {{ precision: 'fp32', flip_tta: true }} }}
        }}
      }},
      algos
    );
    console.log(JSON.stringify({{ deltas: out, cmp }}));
    """
    res = _run_node_json(js_script)

    for sc_name, st in scenarios.items():
        d_cnt, _ = _python_expected_stage_modifications(
            algos_by_id[st["detectorId"]], st["detectorParamsByModel"].get(st["detectorId"], {})
        )
        a_cnt, _ = _python_expected_stage_modifications(
            algos_by_id["five_point_canonical_aligner"], st["alignmentParams"]
        )
        q_cnt, _ = _python_expected_stage_modifications(
            algos_by_id["hybrid_classical_and_norm_quality"], st["qualityParams"]
        )
        r_cnt, _ = _python_expected_stage_modifications(
            algos_by_id[st["recognizerId"]], st["recognizerParamsByModel"].get(st["recognizerId"], {})
        )
        ret_cnt, _ = _python_expected_stage_modifications(
            algos_by_id[st["retrievalIndexId"]], st["retrievalParamsByIndex"].get(st["retrievalIndexId"], {})
        )
        expected_stage_counts = {
            "detection": d_cnt,
            "alignment": a_cnt,
            "quality": q_cnt,
            "recognition": r_cnt,
            "retrieval": ret_cnt,
        }
        js_delta = res["deltas"][sc_name]
        assert js_delta["stageCounts"] == expected_stage_counts, f"Mismatch in {sc_name}"
        assert js_delta["totalModified"] == sum(expected_stage_counts.values())

    cmp_out = res["cmp"]
    assert cmp_out["datasetMismatch"] is True
    assert cmp_out["modelFamilyMismatch"] is True
    assert len(cmp_out["warnings"]) >= 2
    assert cmp_out["counts"]["changed"] >= 3
    assert cmp_out["counts"]["identical"] >= 4
    assert cmp_out["counts"]["added"] >= 1
    assert cmp_out["counts"]["absent"] >= 1


def test_wb3_p2_all_12_boundary_tie_and_zero_denominator_cases_python_vs_js():
    gen = [0.15, 0.30, 0.45, 0.60, 0.85]
    imp = [-0.25, 0.05, 0.15, 0.30, 0.52]

    cases = [
        ("min_permitted_threshold", gen, imp, -2.0),
        ("max_permitted_threshold", gen, imp, 1.0),
        ("exact_score_tie_both", gen, imp, 0.30),
        ("threshold_equal_genuine_only", gen, imp, 0.60),
        ("threshold_equal_impostor_only", gen, imp, 0.52),
        ("threshold_midpoint_adjacent", gen, imp, 0.375),
        ("no_genuine_scores", [], imp, 0.20),
        ("no_impostor_scores", gen, [], 0.20),
        ("all_genuine_accepted", gen, imp, 0.10),
        ("all_genuine_rejected", gen, imp, 0.90),
        ("all_impostor_accepted", gen, imp, -0.30),
        ("all_impostor_rejected", gen, imp, 0.55),
    ]

    js_cases_payload = [{"id": cid, "g": g, "i": im_s, "t": tau} for cid, g, im_s, tau in cases]
    js_script = f"""
    import {{ computeEmpiricalOperatingPoint }} from './frontend/js/components/threshold_explorer.js';
    const cases = {json.dumps(js_cases_payload)};
    const out = {{}};
    for (const c of cases) {{
      out[c.id] = computeEmpiricalOperatingPoint(c.g, c.i, c.t, true);
    }}
    console.log(JSON.stringify(out));
    """
    js_results = _run_node_json(js_script)

    for cid, g, im_s, tau in cases:
        py_op = compute_empirical_operating_point(g, im_s, tau, higher_is_more_similar=True)
        js_op = js_results[cid]
        for key in ("tp", "fn", "fp", "tn", "genuine_count", "impostor_count", "total_pairs"):
            assert js_op[key] == py_op[key], f"{cid} mismatch on {key}: JS={js_op[key]} vs PY={py_op[key]}"
        for rate_key in ("tar", "frr", "far", "tnr"):
            if py_op[rate_key] is None:
                assert js_op[rate_key] is None, f"{cid} expected null on {rate_key}, got {js_op[rate_key]}"
            else:
                assert abs(float(js_op[rate_key]) - float(py_op[rate_key])) <= 1e-9, f"{cid} {rate_key} drift"


@pytest.mark.skipif(
    not (
        (get_project_root() / "datasets/evaluation/lfw_research_only/pairs.txt").is_file()
        and (get_project_root() / "models/detection/face_detection_yunet_2023mar.onnx").is_file()
        and (get_project_root() / "models/recognition/public_resnet50/adaface_ir50_webface4m.onnx").is_file()
    ),
    reason="LFW dataset or YuNet/AdaFace ONNX weights not installed locally (see README.md).",
)
def test_wb3_p2_lfw_pipeline_score_distributions_partition_isolation_legacy_and_exports(tmp_path: Path):
    models = ModelResourceManager()
    datasets = DatasetCatalog()
    history = ExperimentHistoryStore(runs_rel_dir=str(tmp_path))
    # Create tmp history store backed by isolated temp dir while still ingesting historical read-only baselines
    history.runs_dir = tmp_path
    history.runs_dir.mkdir(parents=True, exist_ok=True)
    engine = WorkbenchExecutionEngine(models, datasets)
    runner = ExperimentJobRunner(engine, history)

    try:
        sub = runner.submit(
            job_type="pipeline_run",
            name="WB3 Test LFW Stratified Score Distribution Run",
            config={
                "dataset_id": "lfw_research_split",
                "pair_limit": 24,
                "detection": {"algorithm_id": "yunet_2023mar", "enable_landmarks": True, "params": {}},
                "alignment": {"algorithm_id": "five_point_canonical_aligner", "params": {}},
                "quality": {"enabled": True, "params": {}},
                "recognition": {"enabled": True, "algorithm_id": "adaface_ir50_webface4m", "strict_weights": True, "params": {}},
                "normalization": {"enabled": True},
                "matching": {"metric": "cosine", "threshold": 0.2169},
                "retrieval": {"enabled": False, "algorithm_id": "exact_flat_index", "params": {}},
            },
            save_experiment=True,
        )
        job_id = sub["job_id"]
        for _ in range(120):
            j = runner.get_job(job_id)
            if j["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.25)

        assert j["status"] == "completed", f"Pipeline job failed: {j.get('error')}"
        exp_id = j["experiment_id"]
        assert exp_id is not None

        saved_rec = history.get_experiment(exp_id)
        sd = saved_rec["curves"]["score_distributions"]
        assert sd["schema_version"] == "wb3.v1"
        assert sd["experiment_id"] == exp_id
        assert sd["dataset_id"] == "lfw_research_split"
        assert set(sd["partitions"].keys()) == {"held_out_test", "calibration"}

        test_part = sd["partitions"]["held_out_test"]
        calib_part = sd["partitions"]["calibration"]
        assert test_part["is_held_out_test"] is True and test_part["is_calibration"] is False
        assert calib_part["is_calibration"] is True and calib_part["is_held_out_test"] is False

        # Verify calibration & held-out test pair IDs are 100% disjoint (never mixed!)
        test_pair_ids = {p["pair_id"] for p in test_part["pair_refs"]}
        calib_pair_ids = {p["pair_id"] for p in calib_part["pair_refs"]}
        assert test_pair_ids.isdisjoint(calib_pair_ids)
        assert len(test_part["genuine_scores"]) + len(test_part["impostor_scores"]) == test_part["total_pairs"]

        # Verify JSON, CSV, and YAML exports preserve threshold & empirical operating-point fields
        j_bytes, _, _ = history.export_experiment(exp_id, "json")
        j_doc = json.loads(j_bytes.decode("utf-8"))
        assert "score_distributions" in j_doc["curves"]

        c_bytes, _, _ = history.export_experiment(exp_id, "csv")
        csv_rows = list(csv.DictReader(io.StringIO(c_bytes.decode("utf-8"))))
        csv_keys = {r["metric_key"] for r in csv_rows}
        assert {"decision_threshold", "tp_at_manual_thresh", "fp_at_manual_thresh", "tar_at_manual_thresh", "far_at_manual_thresh"}.issubset(
            csv_keys
        )

        y_bytes, _, _ = history.export_experiment(exp_id, "yaml")
        y_doc = yaml.safe_load(y_bytes.decode("utf-8"))
        assert "empirical_operating_point" in y_doc
        assert y_doc["empirical_operating_point"]["total_pairs"] == test_part["total_pairs"]

        # Verify older historical/pre-WB3 records without `score_distributions` still load normally
        legacy_hist = history.get_experiment("HIST-P3-EXP02-ADAFACE_IR50_FP16-SCRFD_5PT_LANDMARKS")
        assert "score_distributions" not in legacy_hist.get("curves", {})
        assert legacy_hist["source_origin"] == "historical_readonly"
    finally:
        models.unload(unload_all=True)


def test_wb3_p3_overlay_opacity_compositing_and_quality_viewer_landmarks():
    # Verify exact compositing parity across 0%, 25%, 50%, 75%, 100%
    base_rgb = [32, 144, 212]
    ov_rgb = [240, 48, 80]
    js_out = _run_node_json(
        f"""
        import {{ computeOverlayAlphaComposite }} from './frontend/js/components/image_viewer.js';
        const levels = [0, 25, 50, 75, 100];
        const res = {{}};
        for (const p of levels) {{
          res[p] = computeOverlayAlphaComposite({json.dumps(base_rgb)}, {json.dumps(ov_rgb)}, p);
        }}
        console.log(JSON.stringify(res));
        """
    )
    assert js_out["0"] == base_rgb  # 0% preserves 100% original pixel values (MAE == 0)
    assert js_out["100"] == ov_rgb
    for pct in (25, 50, 75):
        alpha = pct / 100.0
        expected = [round((1.0 - alpha) * b + alpha * o) for b, o in zip(base_rgb, ov_rgb)]
        assert js_out[str(pct)] == expected

    # Verify Quality Lab returns `landmarks_points` and `confidence` for Workspace 05 viewer overlays
    sample_img = get_project_root() / "datasets/samples/dev_single_sally_ride.jpg"
    yunet_onnx = get_project_root() / "models/detection/face_detection_yunet_2023mar.onnx"
    ada_onnx = get_project_root() / "models/recognition/public_resnet50/adaface_ir50_webface4m.onnx"
    if not (sample_img.is_file() and yunet_onnx.is_file() and ada_onnx.is_file()):
        return
    models = ModelResourceManager()
    datasets = DatasetCatalog()
    engine = WorkbenchExecutionEngine(models, datasets)
    try:
        q_res = engine.run_quality_lab(
            image_refs=["datasets/samples/dev_single_sally_ride.jpg"],
            detector_id="yunet_2023mar",
            recognizer_id="adaface_ir50_webface4m",
        )
        assert len(q_res["items"]) >= 1
        first = q_res["items"][0]
        assert "bounding_box" in first
        assert "landmarks_points" in first and len(first["landmarks_points"]) == 5
        assert "confidence" in first and 0.0 < first["confidence"] <= 1.0
    finally:
        models.unload(unload_all=True)
