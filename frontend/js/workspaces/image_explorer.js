/**
 * Workspace 2 — Central Image Explorer (`frontend/js/workspaces/image_explorer.js`).
 *
 * Supports single/multi-image loading, directory scanning, dataset browsing, native-resolution
 * inspection, side-by-side synchronized comparison, face-instance selection, and strict separation
 * between Original Image -> Detected Face -> Aligned Crop -> Recognition Embedding/Template.
 */

import { api } from "../api.js";
import { ScientificImageViewer } from "../components/image_viewer.js";
import { showToast, store } from "../state.js";
import { withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderImageExplorer(container, navigateTo) {
  let sideBySide = false;
  let currentInspection = null;
  let currentFacePipeline = null;

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>02 · Image Explorer &amp; Face-Instance Inspector</h2>
        <p class="wb-workspace-desc">
          Load individual images, local directories, or permitted datasets. Clearly separates
          Original Frame → Detected Face Box → Aligned 112×112 Crop → Embedding Vector (never person identity).
        </p>
      </div>
      <div class="wb-toolbar-actions">
        <label class="wb-btn wb-btn-secondary wb-btn-sm" style="cursor:pointer;" title="Import local JPG/PNG/BMP/WEBP image(s) into session uploads (<=16 MiB, <=8192 px; never transmitted externally)" data-tip-title="Upload Session Image(s)" data-tip-body="Validates and saves local image files into reports/workbench_runs/uploads/ (max 16 MiB, max 8192 px / 36 MP).">
          + Load Local Image(s)
          <input type="file" id="ie-file-input" accept="image/*" multiple style="display:none;" />
        </label>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-sbs-toggle" type="button" title="Toggle synchronized dual-viewer comparison (Shift+click a thumbnail to set Image B)" data-tip-title="Side-by-Side Synced Viewer" data-tip-body="Splits the stage into two viewers with synchronized zoom and pan. Shift+click any thumbnail to load Comparison Image B.">Mode: Single View</button>
        <button class="wb-btn wb-btn-sm" id="ie-run-process-btn" type="button" title="Run active face detector and quality/crop inspection on the selected image">Detect &amp; Inspect Faces</button>
      </div>
    </div>

    <!-- Source Selector Bar -->
    <div class="wb-panel">
      <div class="wb-panel-body wb-filter-grid-ie">
        <div>
          <label class="wb-label" for="ie-dataset-select">Load From Permitted Dataset</label>
          <select id="ie-dataset-select" class="wb-select" title="Populate the working image strip from a registered local dataset">
            <option value="">— Select Dataset —</option>
            ${(store.datasets || [])
              .map((d) => `<option value="${esc(d.dataset_id)}">${esc(d.name)}</option>`)
              .join("")}
          </select>
        </div>
        <div>
          <label class="wb-label" for="ie-dir-input">Or Scan Local Image Directory Path</label>
          <input id="ie-dir-input" class="wb-input wb-mono" placeholder="e.g. datasets/evaluation/detection" value="datasets/evaluation/detection" title="Project-relative directory inside permitted roots (datasets/* or reports/workbench_runs/uploads)" data-tip-title="Permitted Directory Scanner" data-tip-body="Scans supported images inside datasets/ or reports/workbench_runs/uploads/. Paths outside permitted roots are blocked (HTTP 403)." />
        </div>
        <div>
          <label class="wb-label" for="ie-detector-select">Active Detector for Overlay</label>
          <select id="ie-detector-select" class="wb-select" title="Select YuNet (CPU) or SCRFD-500M (GPU/CPU) for bounding-box and landmark extraction">
            <option value="yunet_2023mar" ${store.exp.detectorId === "yunet_2023mar" ? "selected" : ""}>YuNet (2023mar CPU)</option>
            <option value="scrfd_500m_research" ${store.exp.detectorId === "scrfd_500m_research" ? "selected" : ""}>SCRFD-500M (GPU/CPU)</option>
          </select>
        </div>
        <button class="wb-btn wb-btn-secondary" id="ie-scan-btn" type="button" title="Scan the specified permitted directory and load matched images into the working strip">Load Directory</button>
      </div>
    </div>

    <!-- Loaded Image Strip with Prev/Next -->
    <div class="wb-panel">
      <div class="wb-panel-header">
        <span>Working Image Set (<span id="ie-count">${store.exp.explorerImages.length}</span> images)</span>
        <div class="wb-toolbar-actions">
          <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-prev-btn" type="button" title="Select previous image in working set">◀ Prev</button>
          <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-next-btn" type="button" title="Select next image in working set">Next ▶</button>
        </div>
      </div>
      <div class="wb-panel-body">
        <div id="ie-thumb-strip" class="wb-thumb-grid"></div>
      </div>
    </div>

    <!-- Main Viewer(s) + Metadata / Face Instance Inspector -->
    <div class="wb-split-layout">
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>1. Original Image Metadata &amp; Provenance</span></div>
          <div class="wb-panel-body" id="ie-meta-box">Loading metadata…</div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>2. Selected Face Instance Hierarchy</span></div>
          <div class="wb-panel-body" id="ie-face-detail-box">
            <div class="wb-help">Click "Detect &amp; Inspect Faces" or click any face bounding box in the viewer.</div>
          </div>
        </div>
      </div>

      <div>
        <div id="ie-viewers-grid">
          <div id="ie-viewer-primary"></div>
          <div id="ie-viewer-secondary" class="hidden"></div>
        </div>
      </div>
    </div>
  `;

  let primaryViewer = null;
  let secondaryViewer = null;

  primaryViewer = new ScientificImageViewer(container.querySelector("#ie-viewer-primary"), {
    title: "Primary Native Image Viewer",
    heightPx: 460,
    onSelectFace: (idx) => {
      store.exp.selectedFaceIndex = idx;
      renderFaceDetail(idx);
    },
    onTransformChange: (t) => {
      if (sideBySide && secondaryViewer) secondaryViewer.applySyncedTransform(t);
    },
  });

  secondaryViewer = new ScientificImageViewer(container.querySelector("#ie-viewer-secondary"), {
    title: "Comparison Image B Viewer (Synced Zoom/Pan)",
    heightPx: 460,
    onTransformChange: (t) => {
      if (sideBySide && primaryViewer) primaryViewer.applySyncedTransform(t);
    },
  });

  function renderThumbStrip() {
    const strip = container.querySelector("#ie-thumb-strip");
    container.querySelector("#ie-count").textContent = String(store.exp.explorerImages.length);
    strip.innerHTML = store.exp.explorerImages
      .map((ref) => {
        const name = ref.split("/").pop();
        const isAct = ref === store.exp.selectedImageRef;
        return `
          <div class="wb-thumb-card ${isAct ? "active" : ""}" data-ref="${esc(ref)}" title="${esc(ref)}">
            <img src="${api.rawImageUrl(ref)}" alt="${esc(name)}" loading="lazy" />
            <div class="wb-mono" style="font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(name)}</div>
          </div>
        `;
      })
      .join("");

    strip.querySelectorAll(".wb-thumb-card").forEach((card) => {
      card.addEventListener("click", async (e) => {
        const ref = card.getAttribute("data-ref");
        if (sideBySide && e.shiftKey) {
          store.exp.selectedSecondaryImageRef = ref;
          await loadSecondaryImage();
        } else {
          store.exp.selectedImageRef = ref;
          store.exp.selectedFaceIndex = 0;
          renderThumbStrip();
          await inspectAndRenderPrimary(true);
        }
      });
    });
  }

  async function loadSecondaryImage() {
    if (!sideBySide) return;
    const refB = store.exp.selectedSecondaryImageRef;
    const metaB = await api.inspectImage(refB);
    await secondaryViewer.loadImageUrl(api.rawImageUrl(refB), metaB);
    secondaryViewer.setScene({ detections: [], gtBoxes: metaB.ground_truth_boxes || [] });
  }

  async function inspectAndRenderPrimary(autoDetect = true) {
    try {
      const ref = store.exp.selectedImageRef;
      const meta = await api.inspectImage(ref);
      currentInspection = meta;
      await primaryViewer.loadImageUrl(api.rawImageUrl(ref), meta);
      primaryViewer.setScene({
        detections: [],
        gtBoxes: meta.ground_truth_boxes || [],
        unmatchedGtBoxes: [],
      });
      renderMetaBox(meta, null);
      if (autoDetect) {
        await runDetectionAndCropInspection();
      }
    } catch (err) {
      showToast(`Image inspection failed: ${err.message}`, true);
    }
  }

  function renderMetaBox(meta, detCount) {
    const metaBox = container.querySelector("#ie-meta-box");
    if (!metaBox) return;
    const fq = meta.frame_quality_summary || {};
    metaBox.innerHTML = `
      <table class="wb-table">
        <tbody>
          <tr><th>Image ID</th><td class="wb-mono">${esc(meta.image_id)}</td></tr>
          <tr><th>Relative Ref</th><td class="wb-mono" style="word-break:break-all;">${esc(meta.image_ref)}</td></tr>
          <tr><th>Dimensions</th><td class="wb-mono">${meta.width} × ${meta.height} × ${meta.channels} (${esc(meta.format)})</td></tr>
          <tr><th>File Size</th><td class="wb-mono">${(meta.file_size_bytes / 1024).toFixed(1)} KiB</td></tr>
          <tr><th>SHA-256</th><td class="wb-mono" style="font-size:10.5px;word-break:break-all;">${esc(meta.sha256)}</td></tr>
          <tr><th>Frame Sharpness / Lum</th><td class="wb-mono">LapVar ${fq.frame_laplacian_var} · Lum ${fq.frame_brightness_mean} · Contrast ${fq.frame_contrast_std}</td></tr>
          <tr><th>Detected Faces</th><td class="wb-mono"><strong>${detCount !== null ? detCount : "Not run"}</strong>${
      meta.ground_truth_available ? ` (GT: ${meta.ground_truth_boxes.length})` : " (No GT)"
    }</td></tr>
        </tbody>
      </table>
      <div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;">
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-to-det" type="button">→ Detection Lab</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-to-align" type="button">→ Alignment Lab</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-set-a" type="button">Set as 1:1 Image A</button>
        <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ie-set-b" type="button">Set as 1:1 Image B</button>
      </div>
    `;

    container.querySelector("#ie-to-det")?.addEventListener("click", () => navigateTo("detection_lab"));
    container.querySelector("#ie-to-align")?.addEventListener("click", () => navigateTo("alignment_lab"));
    container.querySelector("#ie-set-a")?.addEventListener("click", () => {
      store.exp.imageARef = meta.image_ref;
      store.exp.faceIndexA = store.exp.selectedFaceIndex;
      showToast(`Assigned ${meta.filename} (face_${store.exp.selectedFaceIndex}) as Image A.`);
    });
    container.querySelector("#ie-set-b")?.addEventListener("click", () => {
      store.exp.imageBRef = meta.image_ref;
      store.exp.faceIndexB = store.exp.selectedFaceIndex;
      showToast(`Assigned ${meta.filename} (face_${store.exp.selectedFaceIndex}) as Image B.`);
    });
  }

  async function runDetectionAndCropInspection() {
    const detSel = container.querySelector("#ie-detector-select");
    if (!detSel) return;
    try {
      const detId = detSel.value;
      store.exp.detectorId = detId;
      const detRes = await api.runDetection({
        image_ref: store.exp.selectedImageRef,
        detector_id: detId,
        params: store.exp.detectorParamsByModel[detId] || {},
      });
      store.exp.lastDetectionResult = detRes;

      const qualRes =
        detRes.detection_count > 0
          ? await api.runQuality({
              image_refs: [store.exp.selectedImageRef],
              detector_id: detId,
              recognizer_id: store.exp.recognizerId,
            })
          : { items: [] };
      currentFacePipeline = { detRes, qualRes };

      primaryViewer.setScene({
        detections: detRes.detections,
        selectedFaceIndex: Math.min(store.exp.selectedFaceIndex, Math.max(0, detRes.detection_count - 1)),
        gtBoxes: detRes.image_meta.ground_truth_boxes || [],
        unmatchedGtBoxes: detRes.gt_evaluation.unmatched_gt_boxes || [],
      });

      renderMetaBox(detRes.image_meta, detRes.detection_count);
      renderFaceDetail(Math.min(store.exp.selectedFaceIndex, Math.max(0, detRes.detection_count - 1)));
    } catch (err) {
      showToast(`Face inspection error: ${err.message}`, true);
    }
  }

  function renderFaceDetail(faceIdx) {
    const box = container.querySelector("#ie-face-detail-box");
    if (!box) return;
    if (!currentFacePipeline || currentFacePipeline.detRes.detection_count === 0) {
      box.innerHTML = `<div class="wb-help">Zero faces detected in this image under current detector settings.</div>`;
      return;
    }
    const det = currentFacePipeline.detRes.detections[faceIdx];
    const qItem = currentFacePipeline.qualRes.items[faceIdx];
    if (!det) return;

    box.innerHTML = `
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px;">
        ${
          qItem
            ? `<img src="${qItem.crop_data_url}" alt="Aligned 112x112 crop" style="width:88px;height:88px;border-radius:4px;border:2px solid var(--sem-selected);" />`
            : ""
        }
        <div>
          <div class="wb-badge" style="margin-bottom:4px;">ANONYMOUS FACE INSTANCE (NOT IDENTITY)</div>
          <div class="wb-mono" style="font-weight:700;">${esc(det.face_instance_id)}</div>
          <div class="wb-muted-xs wb-mono">
            Detected Box: (${det.bounding_box.x.toFixed(1)}, ${det.bounding_box.y.toFixed(1)}, ${det.bounding_box.w.toFixed(1)}×${det.bounding_box.h.toFixed(1)}px)<br/>
            Confidence: ${(det.confidence * 100).toFixed(2)}% · ${esc(det.size_category)}
          </div>
        </div>
      </div>
      ${
        qItem
          ? `<table class="wb-table">
              <tbody>
                <tr><th>Stage 1: Original Image</th><td class="wb-mono">${currentInspection.width}×${currentInspection.height} BGR Frame</td></tr>
                <tr><th>Stage 2: Detected Region</th><td class="wb-mono">${det.dimensions_wh[0]}×${det.dimensions_wh[1]} px · 5-pt Landmarks</td></tr>
                <tr><th>Stage 3: Aligned Crop</th><td class="wb-mono">112×112 Canonical Umeyama Crop · Quality ${esc(qItem.quality_category)}</td></tr>
                <tr><th>Stage 4: Embedding Proxy</th><td class="wb-mono">||z_raw||_2 = ${qItem.extended_p3.learned_embedding_l2_norm.toFixed(3)} · LapVar = ${qItem.classical_p1.sharpness_laplacian_var.toFixed(1)}</td></tr>
                <tr><th>Quality Warnings</th><td class="wb-mono">${
                  qItem.classical_p1.warnings.length ? esc(qItem.classical_p1.warnings.join(", ")) : "None"
                }</td></tr>
              </tbody>
            </table>`
          : ""
      }
    `;
  }

  // Event bindings
  container.querySelector("#ie-run-process-btn").addEventListener("click", () => runDetectionAndCropInspection());
  container.querySelector("#ie-detector-select").addEventListener("change", () => runDetectionAndCropInspection());

  container.querySelector("#ie-dataset-select").addEventListener("change", async (e) => {
    const dsId = e.target.value;
    if (!dsId) return;
    const res = await api.getDatasetItems(dsId, { kind: "images", limit: 36 });
    store.exp.explorerImages = res.items.map((it) => it.image_ref);
    if (store.exp.explorerImages.length > 0) {
      store.exp.selectedImageRef = store.exp.explorerImages[0];
      store.exp.selectedFaceIndex = 0;
      renderThumbStrip();
      await inspectAndRenderPrimary(true);
    }
  });

  container.querySelector("#ie-scan-btn").addEventListener("click", async () => {
    try {
      const dirVal = container.querySelector("#ie-dir-input").value.trim();
      const res = await api.scanDirectory(dirVal, 60);
      if (res.images.length === 0) {
        showToast("No supported images found in that directory.", true);
        return;
      }
      store.exp.explorerImages = res.images.map((i) => i.image_ref);
      store.exp.selectedImageRef = store.exp.explorerImages[0];
      renderThumbStrip();
      await inspectAndRenderPrimary(true);
      showToast(`Loaded ${res.images.length} images from ${res.directory_path}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  container.querySelector("#ie-file-input").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    let loadedCount = 0;
    try {
      for (const f of files) {
        const b64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(f);
        });
        const up = await api.uploadLocalImage(f.name, b64);
        if (!store.exp.explorerImages.includes(up.image_ref)) {
          store.exp.explorerImages.unshift(up.image_ref);
        }
        store.exp.selectedImageRef = up.image_ref;
        loadedCount += 1;
      }
      if (loadedCount > 0) {
        renderThumbStrip();
        await inspectAndRenderPrimary(true);
        showToast(`Loaded ${loadedCount} local image(s) into session scratch.`);
      }
    } catch (err) {
      showToast(`Upload rejected: ${err.message}`, true);
    } finally {
      e.target.value = "";
    }
  });

  container.querySelector("#ie-sbs-toggle").addEventListener("click", async (e) => {
    sideBySide = !sideBySide;
    e.target.textContent = sideBySide ? "Mode: Side-by-Side (Shift+Click Thumb for B)" : "Mode: Single View";
    const grid = container.querySelector("#ie-viewers-grid");
    const secEl = container.querySelector("#ie-viewer-secondary");
    if (sideBySide) {
      grid.className = "wb-grid-2";
      secEl.classList.remove("hidden");
      await loadSecondaryImage();
    } else {
      grid.className = "";
      secEl.classList.add("hidden");
    }
    primaryViewer.fitToScreen();
  });

  const stepImage = async (delta) => {
    const list = store.exp.explorerImages;
    if (!list.length) return;
    const cur = Math.max(0, list.indexOf(store.exp.selectedImageRef));
    const next = (cur + delta + list.length) % list.length;
    store.exp.selectedImageRef = list[next];
    store.exp.selectedFaceIndex = 0;
    renderThumbStrip();
    await inspectAndRenderPrimary(true);
  };
  container.querySelector("#ie-prev-btn").addEventListener("click", () => stepImage(-1));
  container.querySelector("#ie-next-btn").addEventListener("click", () => stepImage(1));

  renderThumbStrip();
  await inspectAndRenderPrimary(true);
}
