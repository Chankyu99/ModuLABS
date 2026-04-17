// Node-oriented editable pro deck builder.
// Run this after editing SLIDES, SOURCES, and layout functions.
// The init script installs a sibling node_modules/@oai/artifact-tool package link
// and package.json with type=module for shell-run eval builders. Run with the
// Node executable from Codex workspace dependencies or the platform-appropriate
// command emitted by the init script.
// Do not use pnpm exec from the repo root or any Node binary whose module
// lookup cannot resolve the builder's sibling node_modules/@oai/artifact-tool.

const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "eth-monitoring-draft";
const OUT_DIR = "/Users/chankyulee/Desktop/ModuLABS/05_TimeSeries/Projects/FTS_Projects/presentation/final";
const REF_DIR = "/Users/chankyulee/Desktop/ModuLABS/05_TimeSeries/Projects/FTS_Projects/presentation/reference-images";
const SCRATCH_DIR = path.resolve(process.env.PPTX_SCRATCH_DIR || path.join("tmp", "slides", DECK_ID));
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");
const MAX_RENDER_VERIFY_LOOPS = 3;

const INK = "#101214";
const GRAPHITE = "#30363A";
const MUTED = "#687076";
const PAPER = "#F7F4ED";
const PAPER_96 = "#F7F4EDF5";
const WHITE = "#FFFFFF";
const ACCENT = "#27C47D";
const ACCENT_DARK = "#116B49";
const GOLD = "#D7A83D";
const CORAL = "#E86F5B";
const TRANSPARENT = "#00000000";

const TITLE_FACE = "Apple SD Gothic Neo";
const BODY_FACE = "Apple SD Gothic Neo";
const MONO_FACE = "Menlo";

const FALLBACK_PLATE_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";

const SOURCES = {
  repo: "Local project notebooks and outputs under /Users/chankyulee/Desktop/ModuLABS/05_TimeSeries/Projects/FTS_Projects",
  audit: "notebooks/01_data_audit.ipynb and outputs/01_data_audit/*.csv",
  eda: "notebooks/02_eda_feature_review.ipynb and outputs/02_eda_feature_review/*.csv",
  baseline: "notebooks/03_baseline_modeling.ipynb and outputs/03_baseline_modeling/*.csv",
  tuning: "notebooks/04_threshold_tuning_and_rules.ipynb and outputs/04_threshold_tuning_and_rules/*.csv",
  story: "notebooks/05_monitoring_story_and_dashboard.ipynb and outputs/05_monitoring_story_and_dashboard/*.csv",
};

const SLIDES = [
  {
    kicker: "ANOMALY MONITORING PORTFOLIO",
    title: "이더리움 1분봉 기반\n이상 상태 조기경보 시스템",
    subtitle: "고빈도 시계열 데이터를 반도체 공정 모니터링 문제로 재해석해 이상 탐지와 오경보 제어 규칙을 설계했습니다.",
    expectedVisual: "Minimal editorial title slide with strong left accent, clean cover hierarchy, and one compact idea panel.",
    moment: "탐지 성능과 오경보 부담의 균형 설계",
    notes: "트레이딩 프로젝트가 아니라 운영 가능한 이상 탐지 시스템을 만들었다는 점을 첫 슬라이드에서 강조한다.",
    sources: ["repo", "story"],
  },
  {
    kicker: "PROJECT FRAMING",
    title: "왜 트레이딩이 아니라 모니터링 문제로 풀었나",
    subtitle: "수익률 예측보다 이상 상태 탐지와 운영 가능성을 보여주는 것이 목표였습니다.",
    expectedVisual: "Three structured cards on a calm editorial canvas.",
    cards: [
      [
        "직무 적합성",
        "반도체 데이터 직무는 이상 감지, 변동성 모니터링, 오경보 감소 역량을 더 직접적으로 요구합니다."
      ],
      [
        "문제 재정의",
        "가격 급등락은 공정 excursion, 거래량 burst는 센서 신호 급증으로 해석했습니다."
      ],
      [
        "핵심 질문",
        "이상 상태를 더 빨리 잡으면서도 현업이 감당 가능한 경보량으로 줄일 수 있는가를 검증했습니다."
      ]
    ],
    notes: "취업용 프로젝트이기 때문에 수익률보다 모니터링과 운영 규칙 설계를 선택한 배경을 설명한다.",
    sources: ["repo", "story"],
  },
  {
    kicker: "DATA AUDIT",
    title: "원본 데이터 품질과 리스크",
    subtitle: "연속 1분봉으로 가정할 수 없는 구간을 먼저 분리해 모델링 전에 데이터 품질 문제를 해결했습니다.",
    expectedVisual: "Three metric tiles with restrained dashboard language.",
    metrics: [
      ["1,000,000", "원본 행 수", "2017-09-25 ~ 2019-11-03"],
      ["9.73%", "누락 분 비율", "전체 1분 시간축 기준"],
      ["4,924분", "최대 gap", "긴 공백 전후 구간 별도 주의"]
    ],
    notes: "이 프로젝트가 성능 이전에 데이터 감사부터 시작되었다는 점을 보여준다. gap과 저유동성 이슈는 운영형 모델링의 전제 조건이다.",
    sources: ["audit"],
  },
  {
    kicker: "FEATURE REVIEW",
    title: "EDA에서 확인한 특징",
    subtitle: "feature drift, redundancy, separability를 확인해 실제로 쓸 만한 feature와 주의할 feature를 구분했습니다.",
    expectedVisual: "Three analysis cards with short evidence-based statements.",
    cards: [
      [
        "drift",
        "가격 레벨 feature는 월별 drift가 컸고, 변동성 feature도 레짐 변화에 민감했습니다."
      ],
      [
        "redundancy",
        "중복 정보가 큰 feature 쌍이 있어 대표 변수만 남기는 전략이 필요했습니다."
      ],
      [
        "separability",
        "t_value 분리력은 momentum, return 계열이 높아 방향성 정보가 강하다는 점을 확인했습니다."
      ]
    ],
    notes: "단순히 feature를 많이 쓰는 것이 아니라, drift와 중복을 먼저 파악해 이후 rule tuning을 위한 기반을 만들었다고 설명한다.",
    sources: ["eda"],
  },
  {
    kicker: "BASELINES",
    title: "Baseline 비교 전략",
    subtitle: "복잡한 모델보다 먼저 단순 baseline의 trade-off를 확인하고, 운영 친화적인 기준을 찾는 데 집중했습니다.",
    expectedVisual: "Three baseline cards with clear role separation.",
    cards: [
      [
        "rolling z-score",
        "직전 60분 대비 현재 수익률이 얼마나 이례적인지 계산해 민감한 탐지형 baseline을 만들었습니다."
      ],
      [
        "EWMA",
        "최근 데이터에 더 빠르게 반응하는 평균과 표준편차를 사용해 상태 변화 대응성을 비교했습니다."
      ],
      [
        "Isolation Forest",
        "다변량 feature를 활용한 보수형 비지도 이상탐지 모델로 false alarm 절감 가능성을 봤습니다."
      ]
    ],
    notes: "모델 복잡도보다 baseline 비교를 먼저 한 이유는 운영 trade-off를 빠르게 이해하기 위해서라고 설명한다.",
    sources: ["baseline"],
  },
  {
    kicker: "RULE TUNING",
    title: "Validation에서 rule을 어떻게 골랐나",
    subtitle: "threshold 와 cooldown 조합을 validation에서 비교해 운영 목적에 따라 서로 다른 rule을 분리했습니다.",
    expectedVisual: "Three tuning metrics emphasizing alert budget and suppression logic.",
    metrics: [
      ["0.0075", "Balanced alert rate", "event_f1 과 false alert/day 균형"],
      ["0.0050", "Conservative alert rate", "더 낮은 경보량 목표"],
      ["20분", "Cooldown", "같은 이벤트 반복 알람 억제"]
    ],
    notes: "good model 하나를 고른 것이 아니라, validation에서 운영 목적별 규칙을 분리한 과정이 핵심이라고 설명한다.",
    sources: ["tuning"],
  },
  {
    kicker: "FINAL RESULT",
    title: "Test에서 확인한 핵심 개선",
    subtitle: "balanced rule은 baseline보다 경보 부담을 크게 줄이면서 point 수준 F1도 개선했습니다.",
    expectedVisual: "Three high-impact metric cards for final comparison.",
    metrics: [
      ["+0.0187", "Balanced point F1 개선", "0.2323 -> 0.2510"],
      ["-47.3%", "Balanced false alerts/day", "8.56 -> 4.51"],
      ["-72.9%", "Conservative false alerts/day", "8.56 -> 2.32"]
    ],
    notes: "balanced는 기본 운영안, conservative는 저경보 운영안으로 제안 가능하다는 메시지를 함께 전달한다.",
    sources: ["tuning", "story"],
  },
  {
    kicker: "DAILY OPERATIONS",
    title: "운영 관점에서 본 하루 단위 KPI",
    subtitle: "일별 집계로 보면 balanced rule이 실제 이벤트 규모와 가장 가까운 경보량을 유지했습니다.",
    expectedVisual: "Three daily-operation metrics focused on monitoring burden.",
    metrics: [
      ["204일", "Test 운영 기간", "2019-04-14 ~ 2019-11-03"],
      ["8.64건", "Balanced 일평균 경보", "baseline 14.18건 대비 감소"],
      ["0일", "Balanced missed day", "이벤트가 있던 날 경보 0건 없음"]
    ],
    notes: "실제 운영에서는 point score보다 하루 경보량과 missed day가 더 직관적이라는 점을 설명한다.",
    sources: ["story", "tuning"],
  },
  {
    kicker: "DOMAIN TRANSLATION",
    title: "반도체 공정 모니터링 언어로 번역",
    subtitle: "같은 프로젝트도 어떤 언어로 설명하느냐에 따라 포트폴리오의 직무 적합도가 달라집니다.",
    expectedVisual: "Three translation cards that connect finance terms to manufacturing monitoring terms.",
    cards: [
      [
        "이상 이벤트",
        "급등락과 변동성 spike를 공정 excursion 또는 상태 이상으로 해석할 수 있습니다."
      ],
      [
        "오경보 제어",
        "cooldown 과 threshold tuning은 중복 알람 억제와 false alarm 제어 문제로 연결됩니다."
      ],
      [
        "운영 모드 분리",
        "balanced는 탐지형 기본 운영안, conservative는 야간 또는 보수 모드로 설명할 수 있습니다."
      ]
    ],
    notes: "면접에서는 모델 이름보다 anomaly monitoring, false alarm control, operations-ready rule design을 강조한다.",
    sources: ["story"],
  },
  {
    kicker: "CLOSING",
    title: "이 프로젝트가 보여주는 것",
    subtitle: "데이터 감사부터 운영 규칙 설계, 직무 언어 번역까지 한 흐름으로 연결한 포트폴리오입니다.",
    expectedVisual: "Three closing cards with concise portfolio claims.",
    cards: [
      [
        "분석 역량",
        "시계열 품질 점검, drift 분석, feature review, baseline 비교를 end-to-end로 수행했습니다."
      ],
      [
        "운영 역량",
        "validation 기반 rule tuning과 daily monitoring KPI 설계를 통해 실무 적용 가능성을 높였습니다."
      ],
      [
        "커뮤니케이션",
        "금융 데이터를 반도체 공정 모니터링 문제로 번역해 포트폴리오 메시지를 명확히 만들었습니다."
      ]
    ],
    notes: "마지막 슬라이드에서는 분석 능력뿐 아니라 운영 규칙 설계와 도메인 번역 능력을 함께 보여준다고 정리한다.",
    sources: ["repo", "story"],
  }
];

const inspectRecords = [];

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  if (!bytes.byteLength) {
    throw new Error(`Image file is empty: ${imagePath}`);
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function normalizeImageConfig(config) {
  if (!config.path) {
    return config;
  }
  const { path: imagePath, ...rest } = config;
  return {
    ...rest,
    blob: await readImageBlob(imagePath),
  };
}

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const obsoleteFinalArtifacts = [
    "preview",
    "verification",
    "inspect.ndjson",
    ["presentation", "proto.json"].join("_"),
    ["quality", "report.json"].join("_"),
  ];
  for (const obsolete of obsoleteFinalArtifacts) {
    await fs.rm(path.join(OUT_DIR, obsolete), { recursive: true, force: true });
  }
  await fs.mkdir(SCRATCH_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(VERIFICATION_DIR, { recursive: true });
}

function lineConfig(fill = TRANSPARENT, width = 0) {
  return { style: "solid", fill, width };
}

function recordShape(slideNo, shape, role, shapeType, x, y, w, h) {
  if (!slideNo) return;
  inspectRecords.push({
    kind: "shape",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    shapeType,
    bbox: [x, y, w, h],
  });
}

function addShape(slide, geometry, x, y, w, h, fill = TRANSPARENT, line = TRANSPARENT, lineWidth = 0, meta = {}) {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: lineConfig(line, lineWidth),
  });
  recordShape(meta.slideNo, shape, meta.role || geometry, geometry, x, y, w, h);
  return shape;
}

function normalizeText(text) {
  if (Array.isArray(text)) {
    return text.map((item) => String(item ?? "")).join("\n");
  }
  return String(text ?? "");
}

function textLineCount(text) {
  const value = normalizeText(text);
  if (!value.trim()) {
    return 0;
  }
  return Math.max(1, value.split(/\n/).length);
}

function requiredTextHeight(text, fontSize, lineHeight = 1.18, minHeight = 8) {
  const lines = textLineCount(text);
  if (lines === 0) {
    return minHeight;
  }
  return Math.max(minHeight, lines * fontSize * lineHeight);
}

function assertTextFits(text, boxHeight, fontSize, role = "text") {
  const required = requiredTextHeight(text, fontSize);
  const tolerance = Math.max(2, fontSize * 0.08);
  if (normalizeText(text).trim() && boxHeight + tolerance < required) {
    throw new Error(
      `${role} text box is too short: height=${boxHeight.toFixed(1)}, required>=${required.toFixed(1)}, ` +
        `lines=${textLineCount(text)}, fontSize=${fontSize}, text=${JSON.stringify(normalizeText(text).slice(0, 90))}`,
    );
  }
}

function wrapText(text, widthChars) {
  const words = normalizeText(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > widthChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }
  if (current) {
    lines.push(current);
  }
  return lines.join("\n");
}

function recordText(slideNo, shape, role, text, x, y, w, h) {
  const value = normalizeText(text);
  inspectRecords.push({
    kind: "textbox",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    text: value,
    textPreview: value.replace(/\n/g, " | ").slice(0, 180),
    textChars: value.length,
    textLines: textLineCount(value),
    bbox: [x, y, w, h],
  });
}

function recordImage(slideNo, image, role, imagePath, x, y, w, h) {
  inspectRecords.push({
    kind: "image",
    slide: slideNo,
    id: image?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    path: imagePath,
    bbox: [x, y, w, h],
  });
}

function applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit, listStyle) {
  box.text = text;
  box.text.fontSize = size;
  box.text.color = color;
  box.text.bold = Boolean(bold);
  box.text.alignment = align;
  box.text.verticalAlignment = valign;
  box.text.typeface = face;
  box.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  if (autoFit) {
    box.text.autoFit = autoFit;
  }
  if (listStyle) {
    box.text.style = "list";
  }
}

function addText(
  slide,
  slideNo,
  text,
  x,
  y,
  w,
  h,
  {
    size = 22,
    color = INK,
    bold = false,
    face = BODY_FACE,
    align = "left",
    valign = "top",
    fill = TRANSPARENT,
    line = TRANSPARENT,
    lineWidth = 0,
    autoFit = null,
    listStyle = false,
    checkFit = true,
    role = "text",
  } = {},
) {
  if (!checkFit && textLineCount(text) > 1) {
    throw new Error("checkFit=false is only allowed for single-line headers, footers, and captions.");
  }
  if (checkFit) {
    assertTextFits(text, h, size, role);
  }
  const box = addShape(slide, "rect", x, y, w, h, fill, line, lineWidth);
  applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit, listStyle);
  recordText(slideNo, box, role, text, x, y, w, h);
  return box;
}

async function addImage(slide, slideNo, config, position, role, sourcePath = null) {
  const image = slide.images.add(await normalizeImageConfig(config));
  image.position = position;
  recordImage(slideNo, image, role, sourcePath || config.path || config.uri || "inline-data-url", position.left, position.top, position.width, position.height);
  return image;
}

async function addPlate(slide, slideNo, opacityPanel = false) {
  slide.background.fill = PAPER;
  const platePath = path.join(REF_DIR, `slide-${String(slideNo).padStart(2, "0")}.png`);
  if (await pathExists(platePath)) {
    await addImage(
      slide,
      slideNo,
      { path: platePath, fit: "cover", alt: `Text-free art-direction plate for slide ${slideNo}` },
      { left: 0, top: 0, width: W, height: H },
      "art plate",
      platePath,
    );
  } else {
    await addImage(
      slide,
      slideNo,
      { dataUrl: FALLBACK_PLATE_DATA_URL, fit: "cover", alt: `Fallback blank art plate for slide ${slideNo}` },
      { left: 0, top: 0, width: W, height: H },
      "fallback art plate",
      "fallback-data-url",
    );
  }
  if (opacityPanel) {
    addShape(slide, "rect", 0, 0, W, H, "#FFFFFFB8", TRANSPARENT, 0, { slideNo, role: "plate readability overlay" });
  }
}

function addHeader(slide, slideNo, kicker, idx, total) {
  addText(slide, slideNo, String(kicker || "").toUpperCase(), 64, 34, 430, 24, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "header",
  });
  addText(slide, slideNo, `${String(idx).padStart(2, "0")} / ${String(total).padStart(2, "0")}`, 1114, 34, 104, 24, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    align: "right",
    checkFit: false,
    role: "header",
  });
  addShape(slide, "rect", 64, 64, 1152, 2, INK, TRANSPARENT, 0, { slideNo, role: "header rule" });
  addShape(slide, "ellipse", 57, 57, 16, 16, ACCENT, INK, 2, { slideNo, role: "header marker" });
}

function addTitleBlock(slide, slideNo, title, subtitle = null, x = 64, y = 86, w = 780, dark = false) {
  const titleColor = dark ? PAPER : INK;
  const bodyColor = dark ? PAPER : GRAPHITE;
  addText(slide, slideNo, title, x, y, w, 142, {
    size: 40,
    color: titleColor,
    bold: true,
    face: TITLE_FACE,
    role: "title",
  });
  if (subtitle) {
    addText(slide, slideNo, subtitle, x + 2, y + 148, Math.min(w, 720), 70, {
      size: 19,
      color: bodyColor,
      face: BODY_FACE,
      role: "subtitle",
    });
  }
}

function addIconBadge(slide, slideNo, x, y, accent = ACCENT, kind = "signal") {
  addShape(slide, "ellipse", x, y, 54, 54, PAPER_96, INK, 1.2, { slideNo, role: "icon badge" });
  if (kind === "flow") {
    addShape(slide, "ellipse", x + 13, y + 18, 10, 10, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "ellipse", x + 31, y + 27, 10, 10, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 22, y + 25, 19, 3, INK, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
  } else if (kind === "layers") {
    addShape(slide, "roundRect", x + 13, y + 15, 26, 13, accent, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "roundRect", x + 18, y + 24, 26, 13, GOLD, INK, 1, { slideNo, role: "icon glyph" });
    addShape(slide, "roundRect", x + 23, y + 33, 20, 10, CORAL, INK, 1, { slideNo, role: "icon glyph" });
  } else {
    addShape(slide, "rect", x + 16, y + 29, 6, 12, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 25, y + 21, 6, 20, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
    addShape(slide, "rect", x + 34, y + 14, 6, 27, accent, TRANSPARENT, 0, { slideNo, role: "icon glyph" });
  }
}

function addCard(slide, slideNo, x, y, w, h, label, body, { accent = ACCENT, fill = PAPER_96, line = INK, iconKind = "signal" } = {}) {
  if (h < 156) {
    throw new Error(`Card is too short for editable pro-deck copy: height=${h.toFixed(1)}, minimum=156.`);
  }
  addShape(slide, "roundRect", x, y, w, h, fill, line, 1.2, { slideNo, role: `card panel: ${label}` });
  addShape(slide, "rect", x, y, 8, h, accent, TRANSPARENT, 0, { slideNo, role: `card accent: ${label}` });
  addIconBadge(slide, slideNo, x + 22, y + 24, accent, iconKind);
  addText(slide, slideNo, label, x + 88, y + 22, w - 108, 28, {
    size: 15,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    role: "card label",
  });
  const wrapped = wrapText(body, Math.max(28, Math.floor(w / 13)));
  const bodyY = y + 86;
  const bodyH = h - (bodyY - y) - 22;
  if (bodyH < 54) {
    throw new Error(`Card body area is too short: height=${bodyH.toFixed(1)}, cardHeight=${h.toFixed(1)}, label=${JSON.stringify(label)}.`);
  }
  addText(slide, slideNo, wrapped, x + 24, bodyY, w - 48, bodyH, {
    size: 16,
    color: INK,
    face: BODY_FACE,
    role: `card body: ${label}`,
  });
}

function addMetricCard(slide, slideNo, x, y, w, h, metric, label, note = null, accent = ACCENT) {
  if (h < 132) {
    throw new Error(`Metric card is too short for editable pro-deck copy: height=${h.toFixed(1)}, minimum=132.`);
  }
  addShape(slide, "roundRect", x, y, w, h, PAPER_96, INK, 1.2, { slideNo, role: `metric panel: ${label}` });
  addShape(slide, "rect", x, y, w, 7, accent, TRANSPARENT, 0, { slideNo, role: `metric accent: ${label}` });
  addText(slide, slideNo, metric, x + 22, y + 24, w - 44, 54, {
    size: 34,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "metric value",
  });
  addText(slide, slideNo, label, x + 24, y + 82, w - 48, 38, {
    size: 16,
    color: GRAPHITE,
    face: BODY_FACE,
    role: "metric label",
  });
  if (note) {
    addText(slide, slideNo, note, x + 24, y + h - 42, w - 48, 22, {
      size: 10,
      color: MUTED,
      face: BODY_FACE,
      role: "metric note",
    });
  }
}

function addNotes(slide, body, sourceKeys) {
  const sourceLines = (sourceKeys || []).map((key) => `- ${SOURCES[key] || key}`).join("\n");
  slide.speakerNotes.setText(`${body || ""}\n\n[Sources]\n${sourceLines}`);
}

function addReferenceCaption(slide, slideNo) {
  addText(
    slide,
    slideNo,
    "Generated art plate is used as visual direction; all meaningful copy and structure are editable PowerPoint objects.",
    64,
    674,
    980,
    22,
    {
      size: 10,
      color: MUTED,
      face: BODY_FACE,
      checkFit: false,
      role: "caption",
    },
  );
}

async function slideCover(presentation) {
  const slideNo = 1;
  const data = SLIDES[0];
  const slide = presentation.slides.add();
  await addPlate(slide, slideNo);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFCC", TRANSPARENT, 0, { slideNo, role: "cover contrast overlay" });
  addShape(slide, "rect", 64, 86, 7, 455, ACCENT, TRANSPARENT, 0, { slideNo, role: "cover accent rule" });
  addText(slide, slideNo, data.kicker, 86, 88, 520, 26, {
    size: 13,
    color: ACCENT_DARK,
    bold: true,
    face: MONO_FACE,
    role: "kicker",
  });
  addText(slide, slideNo, data.title, 82, 130, 785, 184, {
    size: 48,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "cover title",
  });
  addText(slide, slideNo, data.subtitle, 86, 326, 610, 86, {
    size: 20,
    color: GRAPHITE,
    face: BODY_FACE,
    role: "cover subtitle",
  });
  addShape(slide, "roundRect", 86, 456, 390, 92, PAPER_96, INK, 1.2, { slideNo, role: "cover moment panel" });
  addText(slide, slideNo, data.moment || "Replace with core idea", 112, 478, 336, 40, {
    size: 23,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "cover moment",
  });
  addReferenceCaption(slide, slideNo);
  addNotes(slide, data.notes, data.sources);
}

async function slideCards(presentation, idx) {
  const data = SLIDES[idx - 1];
  const slide = presentation.slides.add();
  await addPlate(slide, idx);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFB8", TRANSPARENT, 0, { slideNo: idx, role: "content contrast overlay" });
  addHeader(slide, idx, data.kicker, idx, SLIDES.length);
  addTitleBlock(slide, idx, data.title, data.subtitle, 64, 86, 760);
  const cards = data.cards?.length
    ? data.cards
    : [
        ["Replace", "Add a specific, sourced point for this slide."],
        ["Author", "Use native PowerPoint chart objects for charts; use deterministic geometry for cards and callouts."],
        ["Verify", "Render previews, inspect them at readable size, and fix actionable layout issues within 3 total render loops."],
      ];
  const cols = Math.min(3, cards.length);
  const cardW = (1114 - (cols - 1) * 24) / cols;
  const iconKinds = ["signal", "flow", "layers"];
  for (let cardIdx = 0; cardIdx < cols; cardIdx += 1) {
    const [label, body] = cards[cardIdx];
    const x = 84 + cardIdx * (cardW + 24);
    addCard(slide, idx, x, 426, cardW, 176, label, body, { iconKind: iconKinds[cardIdx % iconKinds.length] });
  }
  addReferenceCaption(slide, idx);
  addNotes(slide, data.notes, data.sources);
}

async function slideMetrics(presentation, idx) {
  const data = SLIDES[idx - 1];
  const slide = presentation.slides.add();
  await addPlate(slide, idx);
  addShape(slide, "rect", 0, 0, W, H, "#FFFFFFBD", TRANSPARENT, 0, { slideNo: idx, role: "metrics contrast overlay" });
  addHeader(slide, idx, data.kicker, idx, SLIDES.length);
  addTitleBlock(slide, idx, data.title, data.subtitle, 64, 86, 700);
  const metrics = data.metrics || [
    ["00", "Replace metric", "Source"],
    ["00", "Replace metric", "Source"],
    ["00", "Replace metric", "Source"],
  ];
  const accents = [ACCENT, GOLD, CORAL];
  for (let metricIdx = 0; metricIdx < Math.min(3, metrics.length); metricIdx += 1) {
    const [metric, label, note] = metrics[metricIdx];
    addMetricCard(slide, idx, 92 + metricIdx * 370, 404, 330, 174, metric, label, note, accents[metricIdx % accents.length]);
  }
  addReferenceCaption(slide, idx);
  addNotes(slide, data.notes, data.sources);
}

async function createDeck() {
  await ensureDirs();
  if (!SLIDES.length) {
    throw new Error("SLIDES must contain at least one slide.");
  }
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  await slideCover(presentation);
  for (let idx = 2; idx <= SLIDES.length; idx += 1) {
    const data = SLIDES[idx - 1];
    if (data.metrics) {
      await slideMetrics(presentation, idx);
    } else {
      await slideCards(presentation, idx);
    }
  }
  return presentation;
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function writeInspectArtifact(presentation) {
  inspectRecords.unshift({
    kind: "deck",
    id: DECK_ID,
    slideCount: presentation.slides.count,
    slideSize: { width: W, height: H },
  });
  presentation.slides.items.forEach((slide, index) => {
    inspectRecords.splice(index + 1, 0, {
      kind: "slide",
      slide: index + 1,
      id: slide?.id || `slide-${index + 1}`,
    });
  });
  const lines = inspectRecords.map((record) => JSON.stringify(record)).join("\n") + "\n";
  await fs.writeFile(INSPECT_PATH, lines, "utf8");
}

async function currentRenderLoopCount() {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  if (!(await pathExists(logPath))) return 0;
  const previous = await fs.readFile(logPath, "utf8");
  return previous.split(/\r?\n/).filter((line) => line.trim()).length;
}

async function nextRenderLoopNumber() {
  return (await currentRenderLoopCount()) + 1;
}

async function appendRenderVerifyLoop(presentation, previewPaths, pptxPath) {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  const priorCount = await currentRenderLoopCount();
  const record = {
    kind: "render_verify_loop",
    deckId: DECK_ID,
    loop: priorCount + 1,
    maxLoops: MAX_RENDER_VERIFY_LOOPS,
    capReached: priorCount + 1 >= MAX_RENDER_VERIFY_LOOPS,
    timestamp: new Date().toISOString(),
    slideCount: presentation.slides.count,
    previewCount: previewPaths.length,
    previewDir: PREVIEW_DIR,
    inspectPath: INSPECT_PATH,
    pptxPath,
  };
  await fs.appendFile(logPath, JSON.stringify(record) + "\n", "utf8");
  return record;
}

async function verifyAndExport(presentation) {
  await ensureDirs();
  const nextLoop = await nextRenderLoopNumber();
  if (nextLoop > MAX_RENDER_VERIFY_LOOPS) {
    throw new Error(
      `Render/verify/fix loop cap reached: ${MAX_RENDER_VERIFY_LOOPS} total renders are allowed. ` +
        "Do not rerender; note any remaining visual issues in the final response.",
    );
  }
  await writeInspectArtifact(presentation);
  const previewPaths = [];
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const slide = presentation.slides.items[idx];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  await pptxBlob.save(pptxPath);
  const loopRecord = await appendRenderVerifyLoop(presentation, previewPaths, pptxPath);
  return { pptxPath, loopRecord };
}

const presentation = await createDeck();
const result = await verifyAndExport(presentation);
console.log(result.pptxPath);
