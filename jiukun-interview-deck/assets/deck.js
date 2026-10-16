var echarts = window["echarts"];
document.addEventListener("DOMContentLoaded", function () {
  var directionEl = document.getElementById("direction-chart");
  var returnEl = document.getElementById("return-chart");
  var charts = [];

  if (directionEl) {
    var directionChart = echarts.init(directionEl, null, { renderer: "svg" });
    charts.push(directionChart);
    window.__deckECharts.register(directionChart, function (c) {
      return {
        animationDuration: 500,
        backgroundColor: "transparent",
        grid: { left: 92, right: 42, top: 82, bottom: 92 },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "shadow" },
          valueFormatter: function (v) { return Number(v).toFixed(5); }
        },
        xAxis: {
          type: "category",
          data: ["默认线性基线", "公平类权重基线", "两轴订单簿模型"],
          axisTick: { show: false },
          axisLine: { lineStyle: { color: c["--border"] } },
          axisLabel: { color: c["--text-2"], fontSize: 17, interval: 0, margin: 20 }
        },
        yAxis: {
          type: "value",
          min: 0.32,
          max: 0.46,
          interval: 0.02,
          name: "宏平均调和分数",
          nameTextStyle: { color: c["--text-3"], fontSize: 16, padding: [0, 0, 12, 0] },
          axisLabel: {
            color: c["--text-2"],
            fontSize: 16,
            formatter: function (v) { return Number(v).toFixed(2); }
          },
          splitLine: { lineStyle: { color: c["--border"], type: "dashed" } }
        },
        series: [{
          type: "bar",
          barWidth: 82,
          data: [
            { value: 0.35847, itemStyle: { color: c["--text-3"] } },
            { value: 0.39337, itemStyle: { color: c["--accent-2"] } },
            { value: 0.43501, itemStyle: { color: c["--good"] } }
          ],
          label: {
            show: true,
            position: "top",
            color: c["--text-1"],
            fontFamily: "IBM Plex Mono, monospace",
            fontWeight: 700,
            fontSize: 21,
            formatter: function (p) { return Number(p.value).toFixed(4); }
          },
          itemStyle: { borderRadius: [4, 4, 0, 0] }
        }]
      };
    });
  }

  if (returnEl) {
    var returnChart = echarts.init(returnEl, null, { renderer: "svg" });
    charts.push(returnChart);
    window.__deckECharts.register(returnChart, function (c) {
      return {
        animationDuration: 500,
        backgroundColor: "transparent",
        grid: { left: 96, right: 42, top: 72, bottom: 92 },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "shadow" },
          valueFormatter: function (v) { return Number(v).toFixed(3) + " 基点"; }
        },
        xAxis: {
          type: "category",
          data: ["毛收益", "扣 2 基点成本", "扣 7 基点成本"],
          axisTick: { show: false },
          axisLine: { lineStyle: { color: c["--border"] } },
          axisLabel: { color: c["--text-2"], fontSize: 17, interval: 0, margin: 20 }
        },
        yAxis: {
          type: "value",
          min: -7.5,
          max: 2,
          interval: 1.5,
          name: "单笔均值（基点）",
          nameTextStyle: { color: c["--text-3"], fontSize: 16, padding: [0, 0, 12, 0] },
          axisLabel: { color: c["--text-2"], fontSize: 16 },
          splitLine: { lineStyle: { color: c["--border"], type: "dashed" } }
        },
        series: [{
          type: "bar",
          barWidth: 84,
          data: [
            { value: 0.484, itemStyle: { color: c["--warn"] }, label: { position: "top" } },
            { value: -1.516, itemStyle: { color: c["--bad"] }, label: { position: "bottom" } },
            { value: -6.516, itemStyle: { color: c["--bad"], opacity: 0.72 }, label: { position: "bottom" } }
          ],
          label: {
            show: true,
            color: c["--text-1"],
            fontFamily: "IBM Plex Mono, monospace",
            fontWeight: 700,
            fontSize: 21,
            formatter: function (p) {
              var value = Number(p.value);
              return (value > 0 ? "+" : "") + value.toFixed(3) + " 基点";
            }
          },
          itemStyle: { borderRadius: [4, 4, 0, 0] },
          markLine: {
            silent: true,
            symbol: "none",
            label: { show: false },
            lineStyle: { color: c["--text-2"], width: 2 },
            data: [{ yAxis: 0 }]
          }
        }]
      };
    });
  }

  window.addEventListener("resize", function () {
    charts.forEach(function (chart) { chart.resize(); });
  });
});
