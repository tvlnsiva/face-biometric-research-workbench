/**
 * Workspace 10 — Dataset & Evaluation Explorer (`frontend/js/workspaces/dataset_explorer.js`).
 *
 * Displays dataset provenance, SHA-256 manifests, identity-disjoint split metadata, face-size
 * distributions, image browser, and evaluation pair inspector keeping Ground Truth strictly
 * separated from live algorithm predictions.
 */

import { api } from "../api.js";
import { renderBarChart } from "../components/charts.js";
import { showToast, store } from "../state.js";
import { helpBadgeHtml, withBusyButton } from "../utils/tooltips.js";

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function renderDatasetExplorer(container, navigateTo) {
  let activeDsId = "nasa_p11_multiscale";
  let activeKind = "images";
  let activeSplit = "all";
  let pageOffset = 0;
  const pageSize = 24;

  container.innerHTML = `
    <div class="wb-workspace-header">
      <div>
        <h2>10 · Dataset &amp; Evaluation Explorer</h2>
        <p class="wb-workspace-desc">
          Inspect dataset manifests, SHA-256 digests, identity-disjoint splits, face-size distributions, and
          individual verification pairs. Ground-truth labels are always displayed separately from model predictions.
        </p>
      </div>
    </div>

    <div class="wb-panel">
      <div class="wb-panel-header"><span>Registered Local Datasets &amp; Provenance Governance</span></div>
      <div class="wb-panel-body">
        <div class="wb-table-wrap">
          <table class="wb-table" id="ds-catalog-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Images / IDs / Faces</th>
                <th>Evaluation Split</th>
                <th>License &amp; Restrictions</th>
                <th>Manifest SHA-256</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${(store.datasets || [])
                .map(
                  (d) => `<tr data-ds="${esc(d.dataset_id)}" class="${d.dataset_id === activeDsId ? "selected-row" : ""}">
                    <td><strong>${esc(d.name)}</strong><div class="wb-muted-xs">${esc(d.source)}</div></td>
                    <td class="wb-mono">${d.num_images.toLocaleString()} / ${d.num_identities.toLocaleString()} / ${d.num_faces.toLocaleString()}</td>
                    <td class="wb-mono" style="font-size:11px;">${esc(d.evaluation_split)}</td>
                    <td style="font-size:11px;"><div><strong>${esc(d.license)}</strong></div><div class="wb-muted-xs">${esc(
                    d.restrictions
                  )}</div></td>
                    <td class="wb-mono" style="font-size:10px;">${esc((d.manifest_sha256 || "—").slice(0, 16))}…</td>
                    <td><button class="wb-btn wb-btn-secondary wb-btn-sm" data-browse="${esc(d.dataset_id)}" type="button">Browse</button></td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="wb-split-layout">
      <div>
        <div class="wb-panel">
          <div class="wb-panel-header"><span>Browser Controls &amp; Split Summary</span></div>
          <div class="wb-panel-body">
            <div class="wb-field">
              <label class="wb-label" for="ds-kind-select">View Mode ${helpBadgeHtml("calibration_threshold")}</label>
              <select id="ds-kind-select" class="wb-select">
                <option value="images">Dataset Images</option>
                <option value="pairs">Verification Pairs (Ground Truth)</option>
              </select>
              <div class="wb-help" id="ds-kind-note"></div>
            </div>
            <div class="wb-field" id="ds-split-field">
              <label class="wb-label" for="ds-split-select">LFW Fold Partition Filter ${helpBadgeHtml("identity_disjoint_split")}</label>
              <select id="ds-split-select" class="wb-select"
                data-tip-key="identity_disjoint_split">
                <option value="all">All 10 Folds (6,000 Pairs)</option>
                <option value="calibration_folds_1_5">Calibration Folds 1–5 (2,086 Subjects · 3,000 Pairs)</option>
                <option value="test_folds_6_10">Held-Out Test Folds 6–10 (2,195 Subjects · 3,000 Pairs)</option>
              </select>
            </div>
            <div id="ds-dist-chart" style="margin-top:10px;"></div>
          </div>
        </div>

        <div class="wb-panel">
          <div class="wb-panel-header"><span>Live Pair Inspector (GT vs. Model Prediction)</span></div>
          <div class="wb-panel-body" id="ds-pair-live-box">
            <div class="wb-help">Select a verification pair on the right to compare Ground Truth against live recognizer output.</div>
          </div>
        </div>
      </div>

      <div class="wb-panel">
        <div class="wb-panel-header">
          <span id="ds-items-title">Dataset Items</span>
          <div class="wb-toolbar-actions">
            <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ds-prev-page" type="button">◀ Prev</button>
            <span class="wb-mono wb-muted-xs" id="ds-page-info">0–0</span>
            <button class="wb-btn wb-btn-secondary wb-btn-sm" id="ds-next-page" type="button">Next ▶</button>
          </div>
        </div>
        <div class="wb-panel-body" id="ds-items-body"></div>
      </div>
    </div>
  `;

  async function refreshBrowser() {
    if (!container.isConnected) return;
    const dsObj = (store.datasets || []).find((d) => d.dataset_id === activeDsId);
    const kindSel = container.querySelector("#ds-kind-select");
    const kindNote = container.querySelector("#ds-kind-note");
    if (!dsObj?.has_verification_pairs && activeKind === "pairs") {
      activeKind = "images";
      kindSel.value = "images";
    }
    kindSel.querySelector('option[value="pairs"]').disabled = !dsObj?.has_verification_pairs;
    if (!dsObj?.has_verification_pairs) {
      kindNote.textContent = "Note: Verification Pairs mode is disabled because this split provides detection ground truth rather than 1:1 pair manifests.";
      kindSel.setAttribute("data-tip-title", "Dataset Images Only");
      kindSel.setAttribute("data-tip-body", "Select 'LFW 10-Fold Verification Split' or 'NASA Smoke Verification Pairs' above to inspect Ground-Truth 1:1 pairs.");
    } else {
      kindNote.textContent = "Switch between individual dataset images and ground-truth 1:1 verification pairs.";
      kindSel.setAttribute("data-tip-title", "Dataset View Mode");
      kindSel.setAttribute("data-tip-body", "Inspect raw images or evaluate individual genuine/impostor pairs against the active recognizer.");
    }
    container.querySelector("#ds-split-field").style.display = activeDsId === "lfw_research_split" ? "block" : "none";

    container.querySelectorAll("#ds-catalog-table tr[data-ds]").forEach((tr) => {
      tr.classList.toggle("selected-row", tr.getAttribute("data-ds") === activeDsId);
    });

    const data = await api.getDatasetItems(activeDsId, {
      kind: activeKind,
      split: activeSplit,
      offset: pageOffset,
      limit: pageSize,
    });
    if (!container.isConnected) return;

    const prevBtn = container.querySelector("#ds-prev-page");
    const nextBtn = container.querySelector("#ds-next-page");
    prevBtn.disabled = pageOffset <= 0;
    prevBtn.title = pageOffset <= 0 ? "Already on the first page." : "Previous page";
    nextBtn.disabled = data.offset + data.limit >= data.total;
    nextBtn.title = data.offset + data.limit >= data.total ? "Already on the last page." : "Next page";

    container.querySelector("#ds-items-title").textContent = `${dsObj?.name || activeDsId} — ${data.total.toLocaleString()} ${data.kind}`;
    container.querySelector("#ds-page-info").textContent = `${data.offset + 1}–${Math.min(
      data.offset + data.limit,
      data.total
    )} of ${data.total}`;

    // Render distribution chart
    const distEl = container.querySelector("#ds-dist-chart");
    if (data.distributions?.face_size_buckets) {
      const b = data.distributions.face_size_buckets;
      renderBarChart(distEl, {
        title: "Ground-Truth Face Size Distribution",
        subtitle: "Original image min(w, h) buckets",
        categories: ["Small (<32px)", "Medium (32–64px)", "Large (>64px)"],
        values: [b.small_lt_32px, b.medium_32_64px, b.large_gt_64px],
        unit: " faces",
      });
    } else if (data.distributions?.pair_balance) {
      const pb = data.distributions.pair_balance;
      renderBarChart(distEl, {
        title: "Verification Pair Class Balance",
        subtitle: data.distributions.identity_disjoint_splits
          ? "Calib 2,086 IDs · Held-Out Test 2,195 IDs · Overlap = 0"
          : "Ground-truth pair balance",
        categories: ["Genuine (Same Subject)", "Impostor (Diff Subject)"],
        values: [pb.positive_genuine, pb.negative_impostor],
        unit: " pairs",
      });
    } else {
      distEl.innerHTML = `<div class="wb-help">Select a detection GT or verification split to view distributions.</div>`;
    }

    const body = container.querySelector("#ds-items-body");
    if (activeKind === "images") {
      body.innerHTML = `
        <div class="wb-thumb-grid">
          ${data.items
            .map(
              (it) => `
              <div class="wb-thumb-card" data-open-img="${esc(it.image_ref)}">
                <img src="${api.rawImageUrl(it.image_ref)}" alt="${esc(it.image_id)}" loading="lazy" />
                <div class="wb-mono" style="font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(
                  it.image_id
                )}</div>
                <div class="wb-muted-xs">${
                  it.gt_face_count !== undefined
                    ? `GT Faces: ${it.gt_face_count}`
                    : esc(it.subject_tag_eval_only || it.category || "")
                }</div>
              </div>`
            )
            .join("")}
        </div>
      `;
      body.querySelectorAll("[data-open-img]").forEach((el) => {
        el.addEventListener("click", () => {
          store.exp.selectedImageRef = el.getAttribute("data-open-img");
          navigateTo("image_explorer");
        });
      });
    } else {
      body.innerHTML = `
        <div class="wb-table-wrap">
          <table class="wb-table">
            <thead><tr><th>Pair ID</th><th>Image A</th><th>Image B</th><th>Ground-Truth Label</th><th>Action</th></tr></thead>
            <tbody>
              ${data.items
                .map(
                  (p, idx) => `<tr>
                    <td class="wb-mono">${esc(p.pair_id)}</td>
                    <td class="wb-mono" style="font-size:11px;">${esc(p.image_a.split("/").pop())}</td>
                    <td class="wb-mono" style="font-size:11px;">${esc(p.image_b.split("/").pop())}</td>
                    <td><span class="wb-badge ${p.is_same_subject ? "wb-badge-perf" : ""}">${
                    p.is_same_subject ? "GENUINE (Same Subject)" : "IMPOSTOR (Different)"
                  }</span></td>
                    <td>
                      <button class="wb-btn wb-btn-secondary wb-btn-sm" data-eval-pair="${idx}" type="button">Evaluate</button>
                      <button class="wb-btn wb-btn-secondary wb-btn-sm" data-send-1to1="${idx}" type="button">→ 1:1 Lab</button>
                    </td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        </div>
      `;

      body.querySelectorAll("[data-eval-pair]").forEach((btn) => {
        btn.addEventListener("click", () =>
          withBusyButton(
            btn,
            async () => {
              const p = data.items[Number(btn.getAttribute("data-eval-pair"))];
              const liveBox = container.querySelector("#ds-pair-live-box");
              if (!liveBox) return;
              liveBox.innerHTML = `<div class="wb-help">Running live verification with ${esc(store.exp.recognizerId)}…</div>`;
              try {
                const ver = await api.runVerification({
                  image_a_ref: p.image_a,
                  image_b_ref: p.image_b,
                  detector_id: store.exp.detectorId,
                  recognizer_id: store.exp.recognizerId,
                  metric: store.exp.matchingMetric,
                  threshold: store.exp.threshold,
                });
                if (!container.isConnected) return;
                const correct = ver.decision.is_match === Boolean(p.is_same_subject);
                liveBox.innerHTML = `
                  <div style="display:flex;gap:8px;margin-bottom:8px;">
                    <img src="${ver.image_a.face.alignment.crop_data_url}" alt="Aligned Face A" style="width:72px;height:72px;border-radius:4px;" />
                    <img src="${ver.image_b.face.alignment.crop_data_url}" alt="Aligned Face B" style="width:72px;height:72px;border-radius:4px;" />
                  </div>
                  <table class="wb-table">
                    <tbody>
                      <tr><th>Ground-Truth Label (Manifest)</th><td><strong>${
                        p.is_same_subject ? "GENUINE (Same Subject)" : "IMPOSTOR (Different Subject)"
                      }</strong></td></tr>
                      <tr><th>Algorithm Prediction (${esc(store.exp.recognizerId)})</th><td><strong>${esc(
                  ver.decision.verdict
                )}</strong></td></tr>
                      <tr><th>Cosine / Threshold</th><td class="wb-mono">${ver.metrics.cosine} vs τ=${ver.decision.threshold}</td></tr>
                      <tr><th>Evaluation Outcome</th><td><span class="wb-badge">${
                        correct ? "CORRECT PREDICTION" : "ERROR (FP / FN)"
                      }</span></td></tr>
                    </tbody>
                  </table>
                `;
              } catch (err) {
                if (container.isConnected) {
                  liveBox.innerHTML = `<div class="wb-banner wb-banner-nonmatch">${esc(err.message)}</div>`;
                }
              }
            },
            "Evaluating…"
          )
        );
      });

      body.querySelectorAll("[data-send-1to1]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const p = data.items[Number(btn.getAttribute("data-send-1to1"))];
          store.exp.imageARef = p.image_a;
          store.exp.imageBRef = p.image_b;
          store.exp.faceIndexA = 0;
          store.exp.faceIndexB = 0;
          navigateTo("recognition_lab");
        });
      });
    }
  }

  container.querySelectorAll("[data-browse]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeDsId = btn.getAttribute("data-browse");
      pageOffset = 0;
      refreshBrowser();
    });
  });
  container.querySelector("#ds-kind-select").addEventListener("change", (e) => {
    activeKind = e.target.value;
    pageOffset = 0;
    refreshBrowser();
  });
  container.querySelector("#ds-split-select").addEventListener("change", (e) => {
    activeSplit = e.target.value;
    pageOffset = 0;
    refreshBrowser();
  });
  container.querySelector("#ds-prev-page").addEventListener("click", () => {
    pageOffset = Math.max(0, pageOffset - pageSize);
    refreshBrowser();
  });
  container.querySelector("#ds-next-page").addEventListener("click", () => {
    pageOffset += pageSize;
    refreshBrowser();
  });

  await refreshBrowser();
}
