(function () {
  var css = getComputedStyle(document.documentElement);
  function tok(name, fallback) {
    var v = css.getPropertyValue(name).trim();
    return v || fallback;
  }
  var navy = tok("--chart-series-1", "#1E3A8A");
  var cyan = tok("--chart-series-2", "#06B6D4");
  var other = tok("--chart-other", "#DDE6F1");
  var grid = tok("--chart-grid", "rgba(14,26,59,0.10)");
  var axis = tok("--chart-axis", "#5B6B8C");

  var baseTooltip = {
    appendToBody: true,
    backgroundColor: tok("--chart-tooltip-bg", "#FFFFFF"),
    borderColor: other,
    borderWidth: 1,
    textStyle: { color: axis, fontSize: 12 },
  };

  // ---------- 图 1：Macro-F1 对比 ----------
  var el1 = document.getElementById("chart-f1");
  if (el1) {
    var c1 = echarts.init(el1, null, { renderer: "svg" });
    c1.setOption({
      animation: false,
      color: [cyan, navy],
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" } }, baseTooltip),
      legend: {
        data: ["深度模型", "Logistic"],
        bottom: 0,
        textStyle: { color: axis, fontSize: 12 },
        itemWidth: 12, itemHeight: 12,
      },
      grid: { left: 46, right: 16, top: 28, bottom: 44 },
      xAxis: {
        type: "category",
        data: ["E007", "E011", "E013", "E014", "E017"],
        axisLine: { lineStyle: { color: other } },
        axisTick: { show: false },
        axisLabel: { color: axis, fontSize: 11 },
      },
      yAxis: {
        type: "value", min: 0.25, max: 0.5, interval: 0.05,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: grid } },
        axisLabel: { color: axis, fontSize: 10, formatter: function (v) { return v.toFixed(2); } },
      },
      series: [
        {
          name: "深度模型", type: "bar", barWidth: 14, itemStyle: { borderRadius: [3, 3, 0, 0] },
          data: [0.431, 0.42, 0.439, 0.446, 0.357],
          markLine: {
            symbol: "none",
            lineStyle: { color: "#9AA7BD", type: "dashed" },
            label: { formatter: "随机基线 0.333", color: "#9AA7BD", fontSize: 10, position: "insideEndTop" },
            data: [{ yAxis: 0.333 }],
          },
        },
        {
          name: "Logistic", type: "bar", barWidth: 14, itemStyle: { borderRadius: [3, 3, 0, 0] },
          data: [0.449, 0.408, 0.413, 0.413, 0.303],
        },
      ],
    });
    window.addEventListener("resize", function () { c1.resize(); });
  }

  // ---------- 图 2：微观反转信号衰减 ----------
  var el2 = document.getElementById("chart-signal");
  if (el2) {
    var c2 = echarts.init(el2, null, { renderer: "svg" });
    c2.setOption({
      animation: false,
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" },
        formatter: function (p) { return p[0].name + "<br/>|RankIC|：" + p[0].value; } }, baseTooltip),
      grid: { left: 48, right: 16, top: 24, bottom: 36 },
      xAxis: {
        type: "category",
        data: ["逐笔 L2（逐档）", "百分比桶（聚合）"],
        axisLine: { lineStyle: { color: other } }, axisTick: { show: false },
        axisLabel: { color: axis, fontSize: 11 },
      },
      yAxis: {
        type: "value", min: 0, max: 0.6, interval: 0.15,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: { lineStyle: { color: grid } },
        axisLabel: { color: axis, fontSize: 10 },
      },
      series: [
        {
          type: "bar", barWidth: 46,
          data: [
            { value: 0.57, itemStyle: { color: cyan, borderRadius: [4, 4, 0, 0] } },
            { value: 0.04, itemStyle: { color: "#B7C2D6", borderRadius: [4, 4, 0, 0] } },
          ],
          label: { show: true, position: "top", color: axis, fontSize: 12, formatter: "{c}" },
        },
      ],
    });
    window.addEventListener("resize", function () { c2.resize(); });
  }
})();
