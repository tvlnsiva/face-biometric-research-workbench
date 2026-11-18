"""Automated unit & algorithmic correctness tests for Milestone P3 research modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.common.config_loader import resolve_project_path
from src.matching.retrieval_indexes import (
    ExactFlatIndex,
    HNSWIndex,
    IVFFlatIndex,
    IVFPQIndex,
    compute_recall_at_k,
)
from src.p3_research.algorithmic_variants import (
    ARCFACE_5PT_112,
    GuardedFixedBatchEmbedder,
    aggregate_subject_template,
    align_face_research,
    apply_downscale_upscale,
    apply_gamma,
    apply_gaussian_blur,
    apply_jpeg_compression,
    compute_extended_quality_features,
    compute_pair_scores,
    evaluate_calibrated_verification,
    gaussian_soft_nms,
)
from src.recognition.base_embedder import l2_normalize_embeddings
from src.recognition.candidate_embedders import MobileFaceNetEmbedder, SFaceEmbedder


def test_soft_nms_retains_decayed_adjacent_face() -> None:
    dets = np.array(
        [
            [10.0, 10.0, 110.0, 110.0, 0.95],
            [35.0, 10.0, 135.0, 110.0, 0.90],  # IoU ~ 0.60 -> hard NMS@0.40 drops it, Soft-NMS decays & keeps
            [300.0, 300.0, 380.0, 380.0, 0.80],
        ],
        dtype=np.float32,
    )
    keep, decayed = gaussian_soft_nms(dets, sigma=0.50, score_thresh=0.35)
    assert 0 in keep and 2 in keep and 1 in keep
    pos_1 = keep.index(1)
    assert 0.35 <= float(decayed[pos_1]) < 0.90


def test_alignment_variants_and_degradations_shapes_and_signal() -> None:
    rng = np.random.default_rng(123)
    img = rng.integers(30, 220, size=(250, 250, 3), dtype=np.uint8)
    # Synthetic textured pattern
    img[::4, :, :] = 240
    lms = ARCFACE_5PT_112 * 2.0 + 15.0

    for tmode in ["similarity_umeyama_4dof", "affine_least_squares_6dof", "similarity_zoom_out_10pct"]:
        for interp in ["nearest", "linear", "cubic", "lanczos4"]:
            crop = align_face_research(img, lms, transform_mode=tmode, interpolation=interp)
            assert crop.shape == (112, 112, 3)
            assert crop.dtype == np.uint8

    clean = align_face_research(img, lms)
    blurred = apply_gaussian_blur(clean, 2.5)
    lowres = apply_downscale_upscale(clean, 20)
    dark = apply_gamma(clean, 2.5)
    jpeg10 = apply_jpeg_compression(clean, 10)

    q_clean = compute_extended_quality_features(clean, lms, raw_embedding_norm=24.0)
    q_blur = compute_extended_quality_features(blurred, lms, raw_embedding_norm=14.0)
    q_jpeg = compute_extended_quality_features(jpeg10, lms, raw_embedding_norm=18.0)

    assert q_blur["sharpness_laplacian_var"] < q_clean["sharpness_laplacian_var"]
    assert q_jpeg["jpeg_blockiness_ratio"] > q_clean["jpeg_blockiness_ratio"]
    assert lowres.shape == (112, 112, 3) and dark.shape == (112, 112, 3)


def test_cosine_and_normalized_l2_exact_roc_equivalence() -> None:
    rng = np.random.default_rng(42)
    raw_a = rng.normal(0.0, 1.0, size=(400, 128)).astype(np.float32) * rng.uniform(5.0, 30.0, size=(400, 1)).astype(np.float32)
    labels = np.array([(i % 2) == 0 for i in range(400)], dtype=bool)
    unit_a = l2_normalize_embeddings(raw_a)
    noisy_gen_dir = l2_normalize_embeddings(unit_a + rng.normal(0.0, 0.08, size=(400, 128)).astype(np.float32))
    raw_b = np.where(
        labels[:, None],
        noisy_gen_dir * rng.uniform(5.0, 30.0, size=(400, 1)).astype(np.float32),
        l2_normalize_embeddings(rng.normal(0.0, 1.0, size=(400, 128)).astype(np.float32)) * rng.uniform(14.0, 18.0, size=(400, 1)).astype(np.float32),
    )

    cos_s = compute_pair_scores(raw_a, raw_b, "cosine")
    nl2_s = compute_pair_scores(raw_a, raw_b, "normalized_l2")
    ul2_s = compute_pair_scores(raw_a, raw_b, "unnormalized_l2")

    ev_cos = evaluate_calibrated_verification(cos_s[:200], labels[:200], cos_s[200:], labels[200:])
    ev_nl2 = evaluate_calibrated_verification(nl2_s[:200], labels[:200], nl2_s[200:], labels[200:])
    ev_ul2 = evaluate_calibrated_verification(ul2_s[:200], labels[:200], ul2_s[200:], labels[200:])

    # Monotone transformation on unit sphere -> identical rank order -> identical ROC AUC!
    assert abs(ev_cos["test_roc_auc"] - ev_nl2["test_roc_auc"]) < 1e-9
    # Unnormalized L2 suffers from radial scale variation between gallery and probe
    assert ev_ul2["test_roc_auc"] < ev_cos["test_roc_auc"]


def test_template_aggregation_weights_clean_over_noisy() -> None:
    rng = np.random.default_rng(7)
    true_dir = l2_normalize_embeddings(rng.normal(0.0, 1.0, size=(1, 64)).astype(np.float32))[0]
    # Shot 0 & 1: low-norm noisy shots; Shot 2: high-norm clean shot aligned with true_dir
    shot0 = (0.3 * true_dir + 0.9 * l2_normalize_embeddings(rng.normal(0, 1, (1, 64)).astype(np.float32))[0]) * 8.0
    shot1 = (0.3 * true_dir + 0.9 * l2_normalize_embeddings(rng.normal(0, 1, (1, 64)).astype(np.float32))[0]) * 9.0
    shot2 = true_dir * 26.0
    stack = np.stack([shot0, shot1, shot2], axis=0).astype(np.float32)

    t_unif = aggregate_subject_template(stack, mode="uniform_mean")
    t_norm = aggregate_subject_template(stack, mode="learned_norm_weighted")
    assert float(np.dot(t_norm, true_dir)) > float(np.dot(t_unif, true_dir))


@pytest.mark.skipif(
    not (
        resolve_project_path("models/recognition/face_recognition_sface_2021dec.onnx").is_file()
        and resolve_project_path("models/recognition/research_only/mobilefacenet_w600k_mbf.onnx").is_file()
    ),
    reason="SFace / MobileFaceNet ONNX weights not installed locally (see README.md).",
)
def test_guarded_fixed_batch_embedder_p2_aud_06() -> None:
    sface = SFaceEmbedder(runtime="onnxruntime", device="cpu", config_path="configs/p2_models.yaml")
    with pytest.raises(ValueError, match="dynamic_batch_supported=False"):
        _ = GuardedFixedBatchEmbedder(sface, fixed_batch_size=8)

    mbf = MobileFaceNetEmbedder(runtime="onnxruntime", device="cpu", batch_size=4, config_path="configs/p2_models.yaml")
    guarded = GuardedFixedBatchEmbedder(mbf, fixed_batch_size=8)
    assert mbf.batch_size == 4  # Never mutated at construction!

    rng = np.random.default_rng(99)
    crops = [rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8) for _ in range(5)]
    out_g = guarded.extract(crops)
    assert mbf.batch_size == 4  # Restored cleanly after extraction!
    assert out_g.embeddings.shape == (5, 512)
    out_ref = mbf.extract(crops)
    cos = np.sum(out_g.embeddings * out_ref.embeddings, axis=1)
    assert float(np.min(cos)) > 0.99999


def test_retrieval_indexes_flat_ivf_pq_hnsw() -> None:
    rng = np.random.default_rng(2026)
    n_gal, dim, n_q = 2000, 64, 40
    gal = l2_normalize_embeddings(rng.normal(0.0, 1.0, size=(n_gal, dim)).astype(np.float32))
    queries = l2_normalize_embeddings(gal[:n_q] + rng.normal(0.0, 0.04, size=(n_q, dim)).astype(np.float32))

    flat_cpu = ExactFlatIndex(dim, metric="ip", backend="numpy_cpu")
    flat_cpu.add(gal)
    _, gt_ids = flat_cpu.search(queries, top_k=10)
    assert np.all(gt_ids[:, 0] == np.arange(n_q))

    if torch.cuda.is_available():
        flat_gpu16 = ExactFlatIndex(dim, metric="ip", backend="torch_cuda_fp16")
        flat_gpu16.add(gal)
        _, gpu_ids = flat_gpu16.search(queries, top_k=10)
        assert compute_recall_at_k(gt_ids, gpu_ids)["recall_at_1"] == 1.0

    ivf = IVFFlatIndex(dim, nlist=32, seed=42)
    ivf.train_and_add(gal)
    _, ivf_ids = ivf.search(queries, top_k=10, nprobe=8)
    assert compute_recall_at_k(gt_ids, ivf_ids)["recall_at_1"] >= 0.95

    ivfpq = IVFPQIndex(dim, nlist=32, m_subspaces=8, k_centroids=64, seed=42)
    ivfpq.train_and_add(gal)
    assert ivfpq.code_bytes_only() == n_gal * 8
    _, pq_ids = ivfpq.search(queries, top_k=10, nprobe=16)
    assert compute_recall_at_k(gt_ids, pq_ids)["recall_at_5"] >= 0.90

    hnsw = HNSWIndex(dim, m=16, ef_construction=80, seed=42)
    hnsw.build(gal)
    _, h_ids = hnsw.search(queries, top_k=10, ef_search=64)
    assert compute_recall_at_k(gt_ids, h_ids)["recall_at_1"] >= 0.95
