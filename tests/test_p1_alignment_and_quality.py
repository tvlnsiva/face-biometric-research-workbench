"""Unit tests for Phase 3 — Face Alignment & Classical Quality Assessment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.alignment.face_aligner import FaceAligner, InvalidLandmarksError
from src.common.config_loader import resolve_project_path
from src.common.types import BoundingBox, FacialLandmarks
from src.detection.yunet_detector import YuNetDetector
from src.pipeline import decode_bgr_image
from src.quality.face_quality import FaceQualityAssessor


@pytest.mark.skipif(
    not (
        resolve_project_path("datasets/samples/dev_single_sally_ride.jpg").is_file()
        and resolve_project_path("models/detection/face_detection_yunet_2023mar.onnx").is_file()
    ),
    reason="Sample image or YuNet ONNX weights not installed locally (see README.md).",
)
def test_face_alignment_output_dimensions_and_crop_coordinate_preservation() -> None:
    img_bgr, _, _, _ = decode_bgr_image(
        resolve_project_path("datasets/samples/dev_single_sally_ride.jpg")
    )
    det = YuNetDetector().detect(img_bgr).detections[0]

    aligner_112 = FaceAligner(output_size_wh=(112, 112))
    aligned_112 = aligner_112.align(img_bgr, det.landmarks, source_bbox=det.bbox)
    assert aligned_112.aligned_bgr.shape == (112, 112, 3)
    assert aligned_112.aligned_bgr.dtype == np.uint8
    assert not np.shares_memory(aligned_112.aligned_bgr, img_bgr)

    x1, y1, x2, y2 = aligned_112.crop_bounds_xyxy
    assert 0 <= x1 < x2 <= img_bgr.shape[1]
    assert 0 <= y1 < y2 <= img_bgr.shape[0]
    assert aligned_112.inter_ocular_px > 40.0

    # Verify configurable output size scales template proportionally
    aligner_160 = FaceAligner(output_size_wh=(160, 160))
    aligned_160 = aligner_160.align(img_bgr, det.landmarks, source_bbox=det.bbox)
    assert aligned_160.aligned_bgr.shape == (160, 160, 3)


@pytest.mark.parametrize(
    "bad_landmarks",
    [
        None,
        np.zeros((4, 2), dtype=np.float32),
        np.full((5, 2), np.nan, dtype=np.float32),
        np.array([[10, 10], [10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32),
        # Collinear points
        np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32),
        # Inter-ocular below minimum threshold (4px)
        np.array([[50.0, 50.0], [51.0, 50.0], [50.5, 55.0], [49.0, 60.0], [52.0, 60.0]], dtype=np.float32),
    ],
)
def test_face_aligner_rejects_invalid_or_degenerate_landmarks(bad_landmarks: object) -> None:
    dummy_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    aligner = FaceAligner()
    with pytest.raises(InvalidLandmarksError):
        aligner.align(dummy_img, bad_landmarks)  # type: ignore[arg-type]


def test_face_aligner_rejects_wrong_landmark_ordering() -> None:
    pts = ((40.0, 50.0), (80.0, 50.0), (60.0, 70.0), (45.0, 90.0), (75.0, 90.0))
    wrong_order = FacialLandmarks(
        points=pts,
        order=("left_eye_subject", "right_eye_subject", "nose_tip", "right_mouth_subject", "left_mouth_subject"),
    )
    aligner = FaceAligner()
    dummy_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    with pytest.raises(InvalidLandmarksError):
        aligner.align(dummy_img, wrong_order)


@pytest.mark.skipif(
    not (
        resolve_project_path("datasets/samples/dev_single_sally_ride.jpg").is_file()
        and resolve_project_path("models/detection/face_detection_yunet_2023mar.onnx").is_file()
    ),
    reason="Sample image or YuNet ONNX weights not installed locally (see README.md).",
)
def test_face_quality_measurements_and_configurable_warnings() -> None:
    img_bgr, _, _, _ = decode_bgr_image(
        resolve_project_path("datasets/samples/dev_single_sally_ride.jpg")
    )
    det = YuNetDetector().detect(img_bgr).detections[0]
    aligned = FaceAligner().align(img_bgr, det.landmarks, source_bbox=det.bbox)

    assessor = FaceQualityAssessor()
    q_good = assessor.assess(img_bgr, det.bbox, aligned.aligned_bgr)
    assert q_good.is_valid_image is True
    assert q_good.bbox_width_px > 100.0
    assert q_good.bbox_height_px > 100.0
    assert 0.01 < q_good.relative_face_area < 1.0
    assert q_good.sharpness_laplacian_var > 20.0
    assert 20.0 < q_good.brightness_mean < 240.0
    assert q_good.contrast_std > 10.0

    # Induce blur and dark exposure to confirm warnings trigger without discarding the face
    blurred_dark = (cv2.GaussianBlur(aligned.aligned_bgr, (21, 21), 8.0) * 0.1).astype(np.uint8)
    tiny_bbox = BoundingBox(x=10.0, y=10.0, w=20.0, h=20.0)
    q_warn = assessor.assess(img_bgr, tiny_bbox, blurred_dark)
    assert q_warn.is_valid_image is True
    assert "face_dimensions_below_minimum" in q_warn.warnings
    assert "low_sharpness_possible_blur" in q_warn.warnings
    assert "low_brightness_underexposed" in q_warn.warnings
    assert "low_contrast" in q_warn.warnings
