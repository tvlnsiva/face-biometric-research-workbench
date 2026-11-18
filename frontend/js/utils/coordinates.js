/**
 * Pure Coordinate Transformation & Viewport Projection Math (`frontend/js/utils/coordinates.js`).
 *
 * Guarantees exact mapping between:
 *   1. Original unresized image pixel coordinates `(origX, origY) in [0..natW, 0..natH]`
 *   2. Viewport canvas CSS pixel coordinates `(screenX, screenY)` under arbitrary fit, zoom & pan
 *   3. Aligned face crop local coordinates `(cropX, cropY) in [0..cropW, 0..cropH]`
 *
 * Zero DOM dependencies — tested directly in `tests/js_coordinates_check.mjs`.
 */

export function computeFitTransform(natWidth, natHeight, viewWidth, viewHeight, paddingPx = 16) {
  const nw = Math.max(1, Number(natWidth) || 1);
  const nh = Math.max(1, Number(natHeight) || 1);
  const vw = Math.max(32, Number(viewWidth) || 320);
  const vh = Math.max(32, Number(viewHeight) || 240);
  const availW = Math.max(16, vw - 2 * paddingPx);
  const availH = Math.max(16, vh - 2 * paddingPx);

  const fitScale = Math.min(availW / nw, availH / nh);
  const offsetX = (vw - nw * fitScale) * 0.5;
  const offsetY = (vh - nh * fitScale) * 0.5;

  return {
    natWidth: nw,
    natHeight: nh,
    viewWidth: vw,
    viewHeight: vh,
    scale: fitScale,
    offsetX,
    offsetY,
  };
}

export function computeNative1to1Transform(natWidth, natHeight, viewWidth, viewHeight) {
  const nw = Math.max(1, Number(natWidth) || 1);
  const nh = Math.max(1, Number(natHeight) || 1);
  const vw = Math.max(32, Number(viewWidth) || 320);
  const vh = Math.max(32, Number(viewHeight) || 240);
  return {
    natWidth: nw,
    natHeight: nh,
    viewWidth: vw,
    viewHeight: vh,
    scale: 1.0,
    offsetX: (vw - nw) * 0.5,
    offsetY: (vh - nh) * 0.5,
  };
}

export function origToScreen(origX, origY, transform) {
  return {
    x: transform.offsetX + origX * transform.scale,
    y: transform.offsetY + origY * transform.scale,
  };
}

export function screenToOrig(screenX, screenY, transform) {
  const s = Math.max(transform.scale, 1e-9);
  const ox = (screenX - transform.offsetX) / s;
  const oy = (screenY - transform.offsetY) / s;
  return {
    x: ox,
    y: oy,
    inBounds: ox >= 0 && ox <= transform.natWidth && oy >= 0 && oy <= transform.natHeight,
    clampedX: Math.min(Math.max(0, ox), transform.natWidth),
    clampedY: Math.min(Math.max(0, oy), transform.natHeight),
  };
}

export function projectBoxToScreen(bbox, transform) {
  const tl = origToScreen(bbox.x, bbox.y, transform);
  return {
    x: tl.x,
    y: tl.y,
    w: bbox.w * transform.scale,
    h: bbox.h * transform.scale,
  };
}

export function projectPointsToScreen(points, transform) {
  return (points || []).map(([px, py]) => {
    const p = origToScreen(px, py, transform);
    return [p.x, p.y];
  });
}

export function zoomAroundScreenPoint(transform, zoomFactor, anchorScreenX, anchorScreenY, minScale = 0.05, maxScale = 24.0) {
  const prevScale = transform.scale;
  const nextScale = Math.min(maxScale, Math.max(minScale, prevScale * zoomFactor));
  const origAtAnchorX = (anchorScreenX - transform.offsetX) / prevScale;
  const origAtAnchorY = (anchorScreenY - transform.offsetY) / prevScale;
  return {
    ...transform,
    scale: nextScale,
    offsetX: anchorScreenX - origAtAnchorX * nextScale,
    offsetY: anchorScreenY - origAtAnchorY * nextScale,
  };
}

export function hitTestFaceBoxes(origX, origY, detections) {
  if (!Array.isArray(detections)) return -1;
  // Prefer smallest containing box when boxes overlap
  let bestIdx = -1;
  let bestArea = Infinity;
  detections.forEach((d, i) => {
    const b = d.bounding_box || d;
    if (origX >= b.x && origX <= b.x + b.w && origY >= b.y && origY <= b.y + b.h) {
      const area = b.w * b.h;
      if (area < bestArea) {
        bestArea = area;
        bestIdx = i;
      }
    }
  });
  return bestIdx;
}
