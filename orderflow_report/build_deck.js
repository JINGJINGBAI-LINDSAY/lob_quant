const pptxgen = require("pptxgenjs");

let pres = new pptxgen();
pres.author = "Order Flow Research";
pres.title = "订单簿驱动的短期价格方向预测";

// ==================== 画布尺寸 ====================
pres.layout = "LAYOUT_16x9";
const SLIDE_W = 10;
const SLIDE_H = 5.625;
const MARGIN = 0.5;
const CONTENT_X = MARGIN;
const CONTENT_W = SLIDE_W - 2 * MARGIN; // 9

// ==================== 设计系统 ====================
const C = {
  navy: "0E1A3B",
  primary: "1E3A8A",
  cyan: "22D3EE",
  cyanDark: "0E7490",
  ink: "172554",
  body: "334155",
  muted: "64748B",
  light: "F4F7FB",
  cardLine: "E2E8F0",
  white: "FFFFFF",
  softCyan: "ECFEFF",
  amber: "F59E0B",
};
const SERIF = "Songti SC";
const SANS = "PingFang SC";
const IMG = "/Users/bytedance/Downloads/order book/orderflow_report/images/";

const makeShadow = () => ({ type: "outer", blur: 10, offset: 3, angle: 135, color: "0B2A4A", opacity: 0.18 });

// ==================== 容器系统 ====================
function calculateScaledImageOpts(opts) {
  const { path, w: targetW, h: targetH, x = 0, y = 0, mode = "cover", ...rest } = opts;
  if (!path || !targetW || !targetH) return opts;
  return { path, x, y, w: targetW, h: targetH, sizing: { type: mode, w: targetW, h: targetH }, ...rest };
}
function createVirtualNode(type, data, parentX = 0, parentY = 0) {
  const opts = data.opts || {};
  const node = {
    type, data,
    absX: parentX + (opts.x || 0),
    absY: parentY + (opts.y || 0),
    w: opts.w || 0, h: opts.h || 0,
    children: [],
  };
  node.addShape = function (shapeType, o = {}) {
    const child = createVirtualNode("shape", { shapeType, opts: o }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  node.addText = function (text, o = {}) {
    const safeOpts = { fit: "shrink", ...o };
    const bulletRe = /^(?:[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25CF\u25CB\u2013\u2014]\s*|\-\s+)/;
    if (Array.isArray(text)) {
      text = text.map((it) => {
        if (it && it.options && it.options.bullet && typeof it.text === "string") {
          return { ...it, text: it.text.replace(bulletRe, "") };
        }
        return it;
      });
    }
    const child = createVirtualNode("text", { text, opts: safeOpts }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  node.addImage = function (o = {}) {
    const child = createVirtualNode("image", { opts: calculateScaledImageOpts(o) }, node.absX, node.absY);
    node.children.push(child);
    return child;
  };
  return node;
}
function flattenNode(node, realSlide) {
  const absOpts = { ...node.data.opts, x: node.absX, y: node.absY };
  if (node.type === "shape") realSlide.addShape(node.data.shapeType, absOpts);
  else if (node.type === "text") realSlide.addText(node.data.text, absOpts);
  else if (node.type === "image") realSlide.addImage(absOpts);
  node.children.forEach((ch) => flattenNode(ch, realSlide));
}
const originalAddSlide = pres.addSlide.bind(pres);
pres.addSlide = function (options) {
  const realSlide = originalAddSlide(options);
  const virtualSlide = {
    children: [],
    _realSlide: realSlide,
    set background(val) { realSlide.background = val; },
    get background() { return realSlide.background; },
    addShape: function (st, o = {}) { const n = createVirtualNode("shape", { shapeType: st, opts: o }, 0, 0); this.children.push(n); return n; },
    addText: function (t, o = {}) { const n = createVirtualNode("text", { text: t, opts: { fit: "shrink", ...o } }, 0, 0); this.children.push(n); return n; },
    addImage: function (o = {}) { const n = createVirtualNode("image", { opts: calculateScaledImageOpts(o) }, 0, 0); this.children.push(n); return n; },
    addChart: function (ct, d, o = {}) { realSlide.addChart(ct, d, o); },
    render: function () { this.children.forEach((ch) => flattenNode(ch, realSlide)); },
  };
  return virtualSlide;
};

// ==================== 通用组件 ====================
function contentHeader(slide, title, sub) {
  slide.addText(title, {
    x: CONTENT_X, y: 0.34, w: CONTENT_W, h: 0.62, margin: 0,
    fontFace: SERIF, fontSize: 26, bold: true, color: C.ink, charSpacing: 1.5,
  });
  if (sub) {
    slide.addText(sub, { x: CONTENT_X, y: 0.98, w: CONTENT_W, h: 0.3, margin: 0, fontFace: SANS, fontSize: 12.5, color: C.muted });
  }
}
function pageNo(slide, n) {
  slide.addText(String(n).padStart(2, "0"), { x: 9.3, y: 5.18, w: 0.4, h: 0.25, margin: 0, align: "right", fontFace: SANS, fontSize: 10, color: "9AA7BD" });
}
function card(slide, x, y, w, h, fill = C.white) {
  return slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, line: { color: C.cardLine, width: 0.75 }, rectRadius: 0.09, shadow: makeShadow(),
  });
}

// ============================================================
// Slide 1 — 封面
// ============================================================
{
  let s = pres.addSlide();
  s.background = { path: IMG + "bg-cover_16x9.jpg" };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 5.4, h: SLIDE_H, fill: { color: "0A1330", transparency: 18 }, line: { type: "none" } });
  s.addText("订单簿驱动的\n短期价格方向预测", {
    x: 0.7, y: 1.5, w: 4.7, h: 1.7, margin: 0, fontFace: SERIF, fontSize: 37, bold: true,
    color: C.white, charSpacing: 2.5, lineSpacingMultiple: 1.12,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.72, y: 3.32, w: 0.62, h: 0.05, fill: { color: C.cyan }, line: { type: "none" } });
  s.addText("基于深度学习的加密货币微观结构研究汇报", {
    x: 0.7, y: 3.5, w: 4.7, h: 0.4, margin: 0, fontFace: SANS, fontSize: 15, color: "C7D6F2",
  });
  s.addText("Binance BTC / ETH　|　30 秒事件级数据　|　2026.10", {
    x: 0.7, y: 4.75, w: 4.7, h: 0.3, margin: 0, fontFace: SANS, fontSize: 11.5, color: "8AA0CC",
  });
  s.render();
}

// ============================================================
// Slide 2 — 目录
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "目录", "CONTENTS");
  const items = [
    ["01", "研究任务与标签口径"],
    ["02", "数据与 49 维特征"],
    ["03", "实验演进脉络"],
    ["04", "核心模型架构"],
    ["05", "关键结果与可交易性"],
    ["06", "结论与下一步"],
  ];
  const colW = 3.7, rowH = 0.92, gapX = 0.4, gapY = 0.28;
  const startX = (SLIDE_W - (colW * 2 + gapX)) / 2;
  const startY = 1.55;
  items.forEach((it, i) => {
    const c = i % 2, r = Math.floor(i / 2);
    const x = startX + c * (colW + gapX);
    const y = startY + r * (rowH + gapY);
    let box = card(s, x, y, colW, rowH);
    box.addText(it[0], { x: 0.28, y: 0, w: 0.9, h: rowH, margin: 0, valign: "middle", fontFace: SERIF, fontSize: 26, bold: true, color: C.cyan, charSpacing: 1 });
    box.addShape(pres.shapes.RECTANGLE, { x: 1.12, y: 0.22, w: 0.02, h: rowH - 0.44, fill: { color: C.cardLine }, line: { type: "none" } });
    box.addText(it[1], { x: 1.32, y: 0, w: colW - 1.55, h: rowH, margin: 0, valign: "middle", fontFace: SANS, fontSize: 15, bold: true, color: C.ink });
  });
  pageNo(s, 2);
  s.render();
}

// ============================================================
// Slide 3 — 研究任务与标签口径
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "研究任务与标签口径", "预测未来 5 分钟价格方向，并严格防止信息泄漏");
  let left = card(s, 0.5, 1.42, 5.25, 3.5);
  const rows = [
    ["数据频率", "每 30 秒一帧，按交易对各自按时间切分"],
    ["预测目标", "未来 5 分钟（10 行）均价相对当前价的收益"],
    ["三类标签", "按训练段 70 分位阈值切为 跌 / 中性 / 涨"],
    ["切分规则", "训练 70%，外推 30% 再等分 dev / test"],
    ["防泄漏", "训练尾部按 horizon 剔除；标准化只用训练段"],
  ];
  rows.forEach((r, i) => {
    const y = 0.24 + i * 0.63;
    left.addShape(pres.shapes.RECTANGLE, { x: 0.26, y: y + 0.06, w: 0.09, h: 0.42, fill: { color: C.cyan }, line: { type: "none" } });
    left.addText(r[0], { x: 0.5, y, w: 1.35, h: 0.55, margin: 0, valign: "middle", fontFace: SANS, fontSize: 12.5, bold: true, color: C.primary });
    left.addText(r[1], { x: 1.72, y, w: 3.3, h: 0.55, margin: 0, valign: "middle", fontFace: SANS, fontSize: 11.5, color: C.body });
  });
  s.addImage(calculateScaledImageOpts({ path: IMG + "image-task_4x3.jpg", x: 6.0, y: 1.78, w: 3.5, h: 2.63, mode: "cover" }));
  s.addText("方向分类（Macro-F1）为主，辅以 AUC、IC", { x: 6.0, y: 4.5, w: 3.5, h: 0.3, margin: 0, align: "center", fontFace: SANS, fontSize: 11, color: C.muted });
  pageNo(s, 3);
  s.render();
}

// ============================================================
// Slide 4 — 数据与 49 维特征
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "数据与 49 维特征", "全部为事件级特征，严格因果，可逐维解释");
  const groups = [
    { n: "24", t: "深度对数", d: "6 档（±0.2/1/2/3/4/5%）的买/卖深度与名义价值，取 log1p，共 4×6 维" },
    { n: "12", t: "不平衡与变化", d: "depth_imbalance 衡量买卖力量对比，depth_change 衡量逐档深度的环比变化" },
    { n: "13", t: "交易与流量尾部", d: "成交不平衡、VWAP 收益、OFI、成交量 z-score、return_1 等短期压力信号" },
  ];
  const cw = 2.9, gap = 0.15, y = 1.5, ch = 2.55;
  groups.forEach((g, i) => {
    const x = 0.5 + i * (cw + gap);
    let box = card(s, x, y, cw, ch);
    box.addShape(pres.shapes.OVAL, { x: 0.26, y: 0.26, w: 0.72, h: 0.72, fill: { color: C.softCyan }, line: { color: C.cyan, width: 1 } });
    box.addText(g.n + " 维", { x: 0.2, y: 0.26, w: 0.84, h: 0.72, margin: 0, align: "center", valign: "middle", fontFace: SERIF, fontSize: 17, bold: true, color: C.cyanDark });
    box.addText(g.t, { x: 0.26, y: 1.12, w: cw - 0.5, h: 0.35, margin: 0, fontFace: SANS, fontSize: 15, bold: true, color: C.ink });
    box.addText(g.d, { x: 0.26, y: 1.52, w: cw - 0.52, h: 0.9, margin: 0, fontFace: SANS, fontSize: 11, color: C.body, lineSpacingMultiple: 1.12 });
  });
  let note = card(s, 0.5, 4.28, 9.0, 0.66, C.softCyan);
  note.addText([
    { text: "合计 24 + 12 + 13 = 49 维。", options: { bold: true, color: C.cyanDark } },
    { text: " 另有 15 个 regime 状态变量不纳入方向输入；此处 mid_price 取每桶最后一笔成交价。", options: { color: C.body } },
  ], { x: 0.28, y: 0, w: 8.5, h: 0.66, margin: 0, valign: "middle", fontFace: SANS, fontSize: 11.5 });
  pageNo(s, 4);
  s.render();
}

// ============================================================
// Slide 5 — 实验演进脉络
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "实验演进脉络", "从数据踩坑到首次超越，再到逐档 L2 严格检验");
  const stages = [
    { tag: "E001–E004", t: "数据踩坑", d: "实时短数据 + 错误切分，出现满分假象；确立按币种时间切分" },
    { tag: "E005–E006", t: "选对方向", d: "Coinbase 整周数据欠拟合，但受控对比确认短期事件级路线" },
    { tag: "E007–E010", t: "公平较量", d: "官方 9 月数据，深度模型屡败；识破混合模型的 ensemble 假象" },
    { tag: "E011/13/14", t: "首次超越", d: "样本翻倍，空洞卷积单模型胜出，E014 达到最强 0.446" },
    { tag: "E016–E018", t: "L2 检验", d: "逐笔账本重建质检通过；领先不显著，桶→L2 为负迁移" },
  ];
  const n = 5, cw = 1.72, gap = 0.1, y = 1.55, ch = 3.0;
  const startX = (SLIDE_W - (n * cw + (n - 1) * gap)) / 2;
  stages.forEach((g, i) => {
    const x = startX + i * (cw + gap);
    let box = card(s, x, y, cw, ch);
    box.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: cw, h: 0.12, fill: { color: i === 3 ? C.cyan : C.primary }, line: { type: "none" } });
    box.addText(String(i + 1), { x: 0.18, y: 0.24, w: 0.6, h: 0.5, margin: 0, fontFace: SERIF, fontSize: 22, bold: true, color: i === 3 ? C.cyanDark : C.primary });
    box.addText(g.tag, { x: 0.18, y: 0.82, w: cw - 0.3, h: 0.3, margin: 0, fontFace: SANS, fontSize: 9.5, color: C.muted });
    box.addText(g.t, { x: 0.18, y: 1.12, w: cw - 0.3, h: 0.35, margin: 0, fontFace: SANS, fontSize: 13.5, bold: true, color: C.ink });
    box.addText(g.d, { x: 0.18, y: 1.52, w: cw - 0.34, h: 1.4, margin: 0, fontFace: SANS, fontSize: 9.8, color: C.body, lineSpacingMultiple: 1.14 });
  });
  pageNo(s, 5);
  s.render();
}

// ============================================================
// Slide 6 — 核心模型架构
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "核心模型架构", "空洞因果卷积为最有效的归纳偏置，叠加时间与特征轴注意力");
  const nodes = ["49 维输入", "空洞因果卷积", "时间注意力", "注意力池化", "特征轴注意力", "三分类输出"];
  const n = nodes.length, nw = 1.36, gap = 0.13, ny = 1.7, nh = 1.0;
  const startX = (SLIDE_W - (n * nw + (n - 1) * gap)) / 2;
  const palette = [C.primary, C.primary, C.cyanDark, C.cyanDark, C.cyanDark, C.primary];
  nodes.forEach((label, i) => {
    const x = startX + i * (nw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: ny, w: nw, h: nh, fill: { color: palette[i] }, line: { type: "none" }, rectRadius: 0.1, shadow: makeShadow() });
    s.addText(label, { x, y: ny, w: nw, h: nh, margin: 0.06, align: "center", valign: "middle", fontFace: SANS, fontSize: 10.5, bold: true, color: C.white, lineSpacingMultiple: 1.05 });
    if (i < n - 1) {
      s.addShape(pres.shapes.LINE, { x: x + nw, y: ny + nh / 2, w: gap, h: 0, line: { color: C.muted, width: 1.5, endArrowType: "triangle" } });
    }
  });
  let call = card(s, 0.5, 3.25, 9.0, 1.5, C.softCyan);
  call.addText("慢变量的角色：择时，不定向", { x: 0.3, y: 0.2, w: 8.4, h: 0.35, margin: 0, fontFace: SANS, fontSize: 13.5, bold: true, color: C.cyanDark });
  call.addText([
    { text: "方向路仅使用 49 个短期特征，慢变量与 FiLM 均不进入分类头（E013 硬拼反而变差）。", options: { breakLine: true, color: C.body } },
    { text: "中长慢信号只在独立的置信过滤器中决定“要不要出手”，无把握时空仓。", options: { color: C.body } },
  ], { x: 0.3, y: 0.6, w: 8.4, h: 0.8, margin: 0, fontFace: SANS, fontSize: 11.5, lineSpacingMultiple: 1.2 });
  pageNo(s, 6);
  s.render();
}

// ============================================================
// Slide 7 — 关键实验结果
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "关键实验结果", "深度模型出现胜出苗头，但优势尚未达到统计显著");
  const chartData = [
    { name: "深度模型", labels: ["E007", "E011", "E013", "E014", "E017"], values: [0.431, 0.420, 0.439, 0.446, 0.357] },
    { name: "Logistic", labels: ["E007", "E011", "E013", "E014", "E017"], values: [0.449, 0.408, 0.413, 0.413, 0.303] },
  ];
  s.addChart(pres.charts.BAR, chartData, {
    x: 0.4, y: 1.4, w: 5.55, h: 3.55,
    barDir: "col", barGrouping: "clustered",
    chartColors: [C.cyan, C.primary],
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: C.muted, catAxisLabelFontFace: SANS, catAxisLabelFontSize: 10,
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 9, valAxisMin: 0.25, valAxisMax: 0.5, valAxisMajorUnit: 0.05,
    valGridLine: { color: "E7EDF5", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendColor: C.body, legendFontFace: SANS, legendFontSize: 10,
    showValue: false, showTitle: true, title: "Macro-F1：深度模型 vs Logistic",
    titleColor: C.ink, titleFontFace: SANS, titleFontSize: 12,
  });
  const insights = [
    ["最强成绩", "E014 达 0.446，领先 Logistic 0.033"],
    ["尚不显著", "L2 上配对 bootstrap 的 95% CI 跨过 0"],
    ["成本后无利", "单边约 0.07% 成本，潜向量直交易为负"],
  ];
  insights.forEach((it, i) => {
    const y = 1.4 + i * 1.22;
    let box = card(s, 6.15, y, 3.35, 1.05);
    box.addText(it[0], { x: 0.24, y: 0.14, w: 2.9, h: 0.3, margin: 0, fontFace: SANS, fontSize: 12, bold: true, color: C.primary });
    box.addText(it[1], { x: 0.24, y: 0.46, w: 2.95, h: 0.5, margin: 0, fontFace: SANS, fontSize: 10.5, color: C.body, lineSpacingMultiple: 1.1 });
  });
  pageNo(s, 7);
  s.render();
}

// ============================================================
// Slide 8 — 潜向量与可交易性
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "潜向量 zₜ 与可交易性", "潜向量含方向信号，但不足以覆盖交易成本");
  let l = card(s, 0.5, 1.5, 4.4, 3.4);
  l.addText("潜向量特征", { x: 0.28, y: 0.22, w: 3.8, h: 0.35, margin: 0, fontFace: SANS, fontSize: 14, bold: true, color: C.ink });
  const lstats = [
    ["128 维", "encoder 输出潜向量，已随结果保存"],
    ["AUC 0.673", "单划分线性探针可提取方向信号"],
    ["ARI 0.075", "聚类不按方向自然分簇"],
  ];
  lstats.forEach((it, i) => {
    const y = 0.7 + i * 0.88;
    l.addText(it[0], { x: 0.28, y, w: 1.7, h: 0.4, margin: 0, fontFace: SERIF, fontSize: 18, bold: true, color: C.cyanDark });
    l.addText(it[1], { x: 0.28, y: y + 0.42, w: 3.85, h: 0.35, margin: 0, fontFace: SANS, fontSize: 10.5, color: C.muted });
  });
  let r = card(s, 5.1, 1.5, 4.4, 3.4);
  r.addText("模拟交易结果", { x: 0.28, y: 0.22, w: 3.8, h: 0.35, margin: 0, fontFace: SANS, fontSize: 14, bold: true, color: C.ink });
  const rstats = [
    ["净收益为负", "IC ≈ −0.02，命中率约 50%"],
    ["单边 0.07%", "手续费 + 滑点侵蚀微弱优势"],
    ["过滤后改善", "仅 28% 时段出手，AUC 升至 0.588"],
  ];
  rstats.forEach((it, i) => {
    const y = 0.7 + i * 0.88;
    const isGood = i === 2;
    r.addText(it[0], { x: 0.28, y, w: 3.6, h: 0.4, margin: 0, fontFace: SERIF, fontSize: 18, bold: true, color: isGood ? C.cyanDark : "B91C1C" });
    r.addText(it[1], { x: 0.28, y: y + 0.42, w: 3.85, h: 0.35, margin: 0, fontFace: SANS, fontSize: 10.5, color: C.muted });
  });
  pageNo(s, 8);
  s.render();
}

// ============================================================
// Slide 9 — 桶 vs 逐笔 L2
// ============================================================
{
  let s = pres.addSlide();
  s.background = { color: C.light };
  contentHeader(s, "百分比桶 vs 逐笔 L2", "现在用桶交付，长期 alpha 押注 L2");
  let l = card(s, 0.5, 1.5, 4.4, 3.25);
  l.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 4.4, h: 0.14, fill: { color: C.primary }, line: { type: "none" } });
  l.addText("百分比桶 — 当前主力", { x: 0.28, y: 0.3, w: 3.9, h: 0.35, margin: 0, fontFace: SANS, fontSize: 14, bold: true, color: C.primary });
  l.addText([
    { text: "现成免费，52 天 BTC+ETH 轻松获取", options: { bullet: true, breakLine: true } },
    { text: "样本约 30 万，多实验反复验证", options: { bullet: true, breakLine: true } },
    { text: "最强 0.446，当前分数最好", options: { bullet: true, breakLine: true } },
    { text: "风险：alpha 天花板可能有限", options: { bullet: true } },
  ], { x: 0.28, y: 0.78, w: 3.85, h: 2.2, margin: 0, fontFace: SANS, fontSize: 11.5, color: C.body, paraSpaceAfter: 8, lineSpacingMultiple: 1.1 });
  let r = card(s, 5.1, 1.5, 4.4, 3.25);
  r.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 4.4, h: 0.14, fill: { color: C.cyan }, line: { type: "none" } });
  r.addText("逐笔 L2 — 未来押注", { x: 0.28, y: 0.3, w: 3.9, h: 0.35, margin: 0, fontFace: SANS, fontSize: 14, bold: true, color: C.cyanDark });
  r.addText([
    { text: "逐档反转信号约 −0.57，强十几倍", options: { bullet: true, breakLine: true } },
    { text: "保留真实价量，信息未被抹平", options: { bullet: true, breakLine: true } },
    { text: "短板：3.4GB/周，当前样本太少", options: { bullet: true, breakLine: true } },
    { text: "桶→L2 已验证为负迁移", options: { bullet: true } },
  ], { x: 0.28, y: 0.78, w: 3.85, h: 2.2, margin: 0, fontFace: SANS, fontSize: 11.5, color: C.body, paraSpaceAfter: 8, lineSpacingMultiple: 1.1 });
  let bar = card(s, 0.5, 4.95, 9.0, 0.5, C.softCyan);
  bar.addText("裁决标准：4 个分散周、约 2,300 OOS 上的移动块配对 bootstrap，95% CI 下界 > 0 即切换主力", {
    x: 0.28, y: 0, w: 8.5, h: 0.5, margin: 0, valign: "middle", fontFace: SANS, fontSize: 10.8, bold: true, color: C.cyanDark,
  });
  pageNo(s, 9);
  s.render();
}

// ============================================================
// Slide 10 — 结论与下一步
// ============================================================
{
  let s = pres.addSlide();
  s.background = { path: IMG + "bg-closing_16x9.jpg" };
  s.addShape(pres.shapes.RECTANGLE, { x: 1.1, y: 0.75, w: 7.8, h: 4.1, fill: { color: "0A1330", transparency: 30 }, line: { color: "1D3566", width: 0.75 } });
  s.addText("方向对、管道通、值得继续", {
    x: 1.3, y: 1.05, w: 7.4, h: 0.7, margin: 0, align: "center", fontFace: SERIF, fontSize: 30, bold: true, color: C.white, charSpacing: 2,
  });
  s.addText("但尚不能宣称已稳健胜出，更不能直接用于盈利交易", {
    x: 1.3, y: 1.78, w: 7.4, h: 0.35, margin: 0, align: "center", fontFace: SANS, fontSize: 13, color: "A9C2EC",
  });
  const steps = [
    ["1", "扩充数据", "抽取 4 个分散周的逐笔 L2，覆盖更多市场状态"],
    ["2", "严格验证", "多种子重复 + 移动块配对 bootstrap，看 95% CI"],
    ["3", "成本回测", "把手续费滑点计入，检验成本后是否净盈利"],
  ];
  steps.forEach((it, i) => {
    const x = 1.4 + i * 2.52;
    s.addShape(pres.shapes.OVAL, { x, y: 2.45, w: 0.5, h: 0.5, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText(it[0], { x, y: 2.45, w: 0.5, h: 0.5, margin: 0, align: "center", valign: "middle", fontFace: SERIF, fontSize: 18, bold: true, color: C.navy });
    s.addText(it[1], { x: x + 0.6, y: 2.45, w: 1.85, h: 0.5, margin: 0, valign: "middle", fontFace: SANS, fontSize: 13, bold: true, color: C.white });
    s.addText(it[2], { x, y: 3.1, w: 2.35, h: 0.9, margin: 0, fontFace: SANS, fontSize: 10, color: "A9C2EC", lineSpacingMultiple: 1.18 });
  });
  s.addText("谢谢聆听 · 欢迎指正", { x: 1.3, y: 4.25, w: 7.4, h: 0.35, margin: 0, align: "center", fontFace: SERIF, fontSize: 15, color: C.cyan, charSpacing: 1 });
  s.render();
}

// ==================== 输出 ====================
pres.writeFile({ fileName: "/Users/bytedance/Downloads/order book/orderflow_report/订单簿方向预测_汇报.pptx" })
  .then((f) => console.log("SAVED:", f));
