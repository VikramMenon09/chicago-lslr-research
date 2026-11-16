/* Chicago LSLR explorer. All values come from data/data.json (written by web/export.py). */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const seq = () => css("--seq").split(",").map((s) => s.trim());

  const f0 = d3.format(",.0f"), f1 = d3.format(",.1f"), f2 = d3.format(",.2f"), pct = d3.format(".0%"), pct1 = d3.format(".1%");
  const fy = d3.format(".1f");
  const bn = (v) => `$${f2(v)}B`;
  const usd = (v) => `$${f0(v)}`;
  const usdk = (v) => `$${d3.format(",.1f")(v / 1000)}k`;
  const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + d3.format(`.${d}f`)(Math.abs(v));

  const SHORT = { SQ: "Status quo", SQ_block: "SQ, block-priced", PROP: "Proportional", PRIO: "Harm priority", CM: "Cost-min.",
    HW: "Health-welfare", EW: "Equity (linear)", EWc: "Equity (convex)", DW: "Distributional",
    HW_r2: "HW, r=2%", HW_r7: "HW, r=7%", HW_child: "HW, child only", HW_high: "HW, high CVD", HW_huynh: "HW, tap-lead" };
  const MAIN = ["SQ", "PROP", "PRIO", "CM", "HW", "EW", "EWc", "DW"];
  const GROUPS = { baseline: "Heuristic baselines", optimized: "Optimized (MILP)", robustness: "Robustness checks (health-welfare)" };

  const LAYERS = [
    { id: "year", label: "Expected replacement year (scenario)", scen: true, fmt: fy,
      bins: [2030, 2033, 2036, 2039, 2042, 2045], domain: [2027, 2048] },
    { id: "cum", label: "Share of lines replaced by year (scenario)", scen: true, fmt: pct,
      bins: [1, 2, 3, 4, 5, 6].map((i) => i / 7), domain: [0, 1] },
    { id: "n_action", label: "Lines needing replacement (observed)", fmt: f0 },
    { id: "action_share", label: "Share of all service lines needing replacement (observed)", fmt: pct },
    { id: "svi", label: "Social Vulnerability Index (observed, 0–1)", fmt: f2 },
    { id: "kids_per_line", label: "Children under 6 per line (observed)", fmt: f2 },
    { id: "holc_D", label: "Share in 1930s HOLC “D” redlined areas (observed)", fmt: pct },
    { id: "median_income", label: "Median household income (observed)", fmt: usd },
    { id: "bll_2009_13", label: "Children with elevated blood lead, 2009–13 (observed, %)", fmt: (v) => f2(v) + "%" },
    { id: "harm_per_line", label: "Health damage per unreplaced line, $/yr (modeled)", fmt: usd },
  ];

  const state = { scen: "HW", layer: "year", year: 2036, rate: 0.03, ca: null, hover: null, frontierZoom: innerWidth < 600 };
  let D, GEO, CA;

  // ---------- tooltip
  const tip = $("#tip");
  function showTip(html, ev) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect(), pad = 14;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  const row = (k, v) => `<div class="r"><span>${k}</span><span>${v}</span></div>`;

  function legendHTML(items) {
    return `<div class="chart-legend">${items.map((i) =>
      `<span><i class="${i.kind || ""}" style="${i.kind === "dash" ? `border-color:${i.color}` : `background:${i.color}`}"></i>${i.label}</span>`).join("")}</div>`;
  }

  // ---------- load
  Promise.all([d3.json("data/data.json"), d3.json("data/community_areas.geojson")]).then(([d, g]) => {
    D = d; GEO = g; CA = new Map(D.ca.map((c) => [c.id, c]));
    initControls(); renderAll();
    let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderAll, 150); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  }).catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p role="alert">Could not load the study data. Reload the page to try again. (${e})</p>`); const h = $("#headline"); if (h) { h.removeAttribute("aria-busy"); h.innerHTML = ""; } });

  function initControls() {
    const sel = $("#scenario");
    for (const [grp, name] of Object.entries(GROUPS)) {
      const og = document.createElement("optgroup"); og.label = name;
      for (const [k, s] of Object.entries(D.scenarios)) if (s.group === grp) og.append(new Option(s.label, k));
      sel.append(og);
    }
    sel.value = state.scen;
    sel.addEventListener("change", () => { state.scen = sel.value; renderScenario(); });

    const ls = $("#layer");
    LAYERS.forEach((l) => ls.append(new Option(l.label, l.id)));
    ls.value = state.layer;
    ls.addEventListener("change", () => { state.layer = ls.value; renderMap(); });

    const yr = $("#year");
    yr.min = D.meta.years[0]; yr.max = D.meta.years.at(-1); yr.value = state.year;
    yr.addEventListener("input", () => { state.year = +yr.value; $("#year-out").textContent = yr.value; renderMap(); });

    const rc = $("#rate-ctl");
    rc.innerHTML = `<div class="seg"><span>Discount rate</span>${D.meta.rates.map((r) =>
      `<button type="button" role="radio" data-r="${r}" aria-checked="${r === state.rate}">${pct(r)}</button>`).join("")}</div>`;
    rc.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      state.rate = +b.dataset.r; rc.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", x === b));
      renderBCR();
    }));
  }

  function renderAll() { renderHeadline(); renderScenario(); renderBCR(); renderTornado(); renderFinance(); renderPSA(); }
  function renderScenario() { renderScenTiles(); renderMap(); renderPace(); renderSocial(); renderFrontier(); }

  // ---------- headline
  function renderHeadline() {
    const S = D.scenarios, dec = D.decomposition, m = D.meta;
    const b31 = D.npv.filter((r) => r.cost === 31000).map((r) => r.bcr);
    const tiles = [
      [f0(m.n_action), `service lines need replacement, out of ${f0(m.n_lines)} in the 2025 inventory`],
      [bn(dec.total), `saved (present value) by the health-welfare schedule vs. the status quo, ${pct1(dec.total / S.SQ.metrics.pv_social_bn)} of its social cost`],
      [`${S.HW.metrics.completion_year} vs ${S.SQ.metrics.completion_year}`, "finish year: optimized schedule vs. continuing the 2021–25 pattern"],
      [`${f1(-S.HW.metrics.q5_gap)} yrs`, "earlier than average that the most vulnerable fifth of lines are reached under the optimum"],
      [`${f2(d3.min(b31))}–${f2(d3.max(b31))}`, "benefit–cost ratio per $31k line across every benefit tier and discount rate tested"],
    ];
    $("#headline").innerHTML = tiles.map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
    $("#headline").removeAttribute("aria-busy");
  }

  function renderScenTiles() {
    const s = D.scenarios[state.scen], m = s.metrics, sq = D.scenarios.SQ.metrics;
    $("#scen-desc").textContent = s.desc; $("#scen-desc-m").textContent = s.desc;
    const dv = m.pv_social_bn - sq.pv_social_bn;
    const tiles = [
      [bn(m.pv_social_bn), "PV social cost (harm + cost)", state.scen === "SQ" ? "baseline" : `${dv < 0 ? "−" : "+"}$${f2(Math.abs(dv))}B vs. status quo`],
      [bn(m.pv_harm_bn), "PV health harm", ""],
      [bn(m.pv_cost_bn), "PV construction cost", `$${f1(m.nominal_cost_bn)}B undiscounted`],
      [String(m.completion_year), "last line replaced", m.completion_year > 2046 ? "misses the 2046 window" : "within the 20-year window"],
      [fy(D.meta.base_year - 1 + m.mean_wait), "average replacement year", `EJ areas ${fy(D.meta.base_year - 1 + m.wait_ej)} · others ${fy(D.meta.base_year - 1 + m.wait_nonej)}`],
      [`${signed(m.q5_gap)} yrs`, "Q5 wait gap (most vulnerable fifth vs. average)", m.q5_gap < 0 ? "served earlier" : "served later"],
      [pct(m.block_share), "of lines done in block programs", `${f0(m.mobilizations)} block mobilizations`],
    ];
    $("#scen-tiles").innerHTML = tiles.map(([v, k, d]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div>${d ? `<div class="d">${d}</div>` : ""}</div>`).join("");
  }

  // ---------- map
  function layerValue(L, id) {
    if (L.id === "year") return D.ca_sched[state.scen].year[id];
    if (L.id === "cum") { const c = D.ca_sched[state.scen].cum[id]; return c ? c[state.year - D.meta.years[0]] : null; }
    const c = CA.get(id); return c ? c[L.id] : null;
  }

  function renderMap() {
    const L = LAYERS.find((l) => l.id === state.layer);
    $("#year-ctl").style.display = L.id === "cum" ? "" : "none";
    const colors = seq();
    let scale, breaks;
    if (L.bins) { scale = d3.scaleThreshold().domain(L.bins).range(colors); breaks = [L.domain[0], ...L.bins]; }
    else {
      const vals = D.ca.map((c) => c[L.id]).filter((v) => v != null);
      scale = d3.scaleQuantile().domain(vals).range(colors); breaks = [d3.min(vals), ...scale.quantiles()];
    }
    const svg = d3.select("#map"), W = 520, H = 640;
    svg.attr("viewBox", `0 0 ${W} ${H}`);
    const proj = d3.geoMercator().fitSize([W, H], GEO), path = d3.geoPath(proj);
    const nodata = css("--nodata");
    svg.selectAll("path.ca").data(GEO.features, (f) => f.properties.id).join("path")
      .attr("class", (f) => "ca" + (f.properties.id === state.ca ? " sel" : ""))
      .attr("d", path)
      .attr("fill", (f) => { const v = layerValue(L, f.properties.id); return v == null ? nodata : scale(v); })
      .on("pointermove", (ev, f) => {
        const v = layerValue(L, f.properties.id);
        showTip(`<b>${f.properties.name}</b>${row(L.label.replace(/ \(.*\)$/, ""), v == null ? "no data" : L.fmt(v))}`, ev);
        if (!state.ca) renderPanel(f.properties.id);
      })
      .on("pointerleave", () => { hideTip(); if (!state.ca) renderPanel(null); })
      .on("click", (ev, f) => {
        state.ca = state.ca === f.properties.id ? null : f.properties.id;
        svg.selectAll("path.ca").classed("sel", (g) => g.properties.id === state.ca);
        renderPanel(state.ca || f.properties.id);
      });
    svg.selectAll("path.ca.sel").raise();

    const lab = (v) => (L.id === "cum" ? pct(v) : L.id === "year" ? String(Math.round(v)) : L.fmt(v));
    $("#map-legend").innerHTML = `<div style="width:100%">${L.label}${L.id === "cum" ? ` — by end of ${state.year}` : ""}</div>
      <div><div class="ramp">${colors.map((c) => `<i style="background:${c}"></i>`).join("")}</div>
      <div class="ramp">${breaks.map((b) => `<span style="width:34px;font-size:.72rem">${lab(b)}</span>`).join("")}</div></div>
      ${D.ca.some((c) => layerValue(L, c.id) == null) ? `<span class="swatch"><i style="background:${nodata}"></i>No data</span>` : ""}`;
    let note = "";
    if (L.scen) {
      const s = D.scenarios[state.scen];
      if (L.id === "cum") {
        const done = d3.sum(s.lines_year.slice(0, state.year - D.meta.years[0] + 1)) / D.meta.n_action;
        note = `Citywide under “${s.label}”: ${pct(Math.min(done, 1))} of lines replaced by the end of ${state.year}.`;
      } else note = `Line-weighted mean of the expected replacement year across the corridors in each area, under “${s.label}”.`;
      note += " Values roll up the 160-corridor schedule; they are model outputs, not a city plan.";
    } else if (L.id === "bll_2009_13") note = "CDPH screening data; used only to validate the model's risk ranking.";
    else note = "Shading uses seven equal-count classes across the 77 community areas.";
    $("#map-note").textContent = note;
    renderPanel(state.ca || state.hover);
  }

  function renderPanel(id) {
    state.hover = id;
    const el = $("#ca-panel");
    if (!id || !CA.get(id)) {
      el.innerHTML = `<h3>Chicago</h3><p class="hint">Hover or tap a community area to see its data and when each schedule reaches it. Click to pin.</p>
        <dl><dt>Lines needing replacement</dt><dd>${f0(D.meta.n_action)}</dd><dt>Community areas</dt><dd>77</dd><dt>Corridors in the model</dt><dd>${D.meta.n_zones}</dd>
        <dt>20-year capital budget</dt><dd>$${f1(D.meta.total_budget_bn)}B</dd></dl>`;
      return;
    }
    const c = CA.get(id);
    const keys = MAIN.includes(state.scen) ? MAIN : [...MAIN, state.scen];
    const y0 = D.meta.years[0], y1 = D.meta.years.at(-1);
    const bars = keys.map((k) => {
      const y = D.ca_sched[k].year[id];
      return `<div class="${k === state.scen ? "cur" : ""}" style="display:contents"><span>${SHORT[k]}</span>
        <span class="b"><i style="width:${(100 * (y - y0 + 1)) / (y1 - y0 + 1)}%"></i></span><span class="y">${fy(y)}</span></div>`;
    }).join("");
    el.innerHTML = `<h3>${c.name}${state.ca === id ? ' <span class="hint" style="font-size:.8rem;font-weight:400">(pinned)</span>' : ""}</h3>
      <dl>
        <dt>Lines needing replacement</dt><dd>${f0(c.n_action)} <span class="hint">(${pct(c.action_share)})</span></dd>
        <dt>Social Vulnerability Index</dt><dd>${f2(c.svi)}</dd>
        <dt>Children under 6 per line</dt><dd>${f2(c.kids_per_line)}</dd>
        <dt>In 1930s redlined (D) areas</dt><dd>${pct(c.holc_D)}</dd>
        <dt>Median household income</dt><dd>${usd(c.median_income)}</dd>
        <dt>Elevated child blood lead, 2009–13</dt><dd>${c.bll_2009_13 == null ? "n/a" : f2(c.bll_2009_13) + "%"}</dd>
        <dt>Modeled damage per line-year</dt><dd>${usd(c.harm_per_line)}</dd>
      </dl>
      <div style="font-weight:600;margin-bottom:6px">Expected replacement year by schedule</div>
      <div class="bars">${bars}</div>`;
  }

  // ---------- generic chart frame
  function frame(sel, h, m = { t: 10, r: 16, b: 34, l: 52 }) {
    const el = $(sel), W = Math.max(300, el.clientWidth);
    const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${h}`).attr("width", W).attr("height", h);
    return { el, svg, W, H: h, m, iw: W - m.l - m.r, ih: h - m.t - m.b, g: svg.append("g").attr("transform", `translate(${m.l},${m.t})`) };
  }
  function yGrid(g, y, iw, fmt, ticks = 5) {
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(ticks).tickSize(-iw).tickFormat("")).call((s) => s.select(".domain").remove());
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(ticks).tickFormat(fmt).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
  }
  function xAxis(g, x, ih, fmt, ticks) {
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`).call(d3.axisBottom(x).ticks(ticks).tickFormat(fmt).tickSizeOuter(0));
  }

  // ---------- pace
  function renderPace() {
    const k = state.scen, s = D.scenarios[k], sq = D.scenarios.SQ, years = D.meta.years;
    const series = [{ key: k, label: s.label, color: css("--s1"), v: s.lines_year }];
    if (k !== "SQ") series.push({ key: "SQ", label: "Status quo", color: css("--s2"), v: sq.lines_year });
    const cap = D.meta.capacity;
    const F = frame("#pace", 280);
    const x = d3.scaleLinear().domain([years[0], years.at(-1)]).range([0, F.iw]);
    const y = d3.scaleLinear().domain([0, Math.max(d3.max(cap), ...series.map((q) => d3.max(q.v))) * 1.08]).nice().range([F.ih, 0]);
    yGrid(F.g, y, F.iw, (v) => d3.format("~s")(v));
    xAxis(F.g, x, F.ih, d3.format("d"), Math.min(11, F.iw / 60));
    F.g.append("path").datum(cap).attr("fill", "none").attr("stroke", css("--neutral")).attr("stroke-width", 1.5).attr("stroke-dasharray", "4 4")
      .attr("d", d3.line().x((d, i) => x(years[i])).y((d) => y(d)));
    for (const q of series.slice().reverse()) {
      F.g.append("path").datum(q.v).attr("fill", "none").attr("stroke", q.color).attr("stroke-width", 2)
        .attr("d", d3.line().x((d, i) => x(years[i])).y((d) => y(d)));
      F.g.selectAll(null).data(q.v).join("circle").attr("cx", (d, i) => x(years[i])).attr("cy", (d) => y(d)).attr("r", 3)
        .attr("fill", q.color).attr("stroke", css("--bg")).attr("stroke-width", 1.5);
    }
    const cross = F.g.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text-3")).attr("stroke-width", 1).style("display", "none");
    F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent")
      .on("pointermove", (ev) => {
        const yr = Math.round(x.invert(d3.pointer(ev)[0])), i = yr - years[0];
        if (i < 0 || i >= years.length) return;
        cross.style("display", null).attr("x1", x(yr)).attr("x2", x(yr));
        showTip(`<b>${yr}</b>${series.map((q) => row(q.label, q.v[i] ? f0(q.v[i]) + " lines" : "done")).join("")}${i < cap.length ? row("Crew capacity", f0(cap[i])) : ""}`, ev);
      })
      .on("pointerleave", () => { cross.style("display", "none"); hideTip(); });
    F.el.innerHTML = legendHTML([...series.map((q) => ({ label: q.label, color: q.color, kind: "line" })),
      { label: "Crew capacity (2027–46)", color: css("--neutral"), kind: "dash" }]);
    F.el.append(F.svg.node());
  }

  // ---------- social cost
  function renderSocial() {
    const keys = Object.keys(D.scenarios);
    const rowsH = 26, F = frame("#social", keys.length * rowsH + 40, { t: 6, r: 64, b: 30, l: Math.min(170, $("#social").clientWidth * 0.36) });
    const y = d3.scaleBand().domain(keys).range([0, F.ih]).padding(0.28);
    const x = d3.scaleLinear().domain([0, d3.max(keys, (k) => D.scenarios[k].metrics.pv_social_bn)]).nice().range([0, F.iw]);
    F.g.append("g").attr("class", "grid").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(6).tickSize(-F.ih).tickFormat(""));
    xAxis(F.g, x, F.ih, (v) => `$${v}B`, 6);
    const cH = css("--s1"), cC = css("--s2"), bg = css("--bg");
    for (const k of keys) {
      const m = D.scenarios[k].metrics, yy = y(k), bh = y.bandwidth();
      const g = F.g.append("g");
      g.append("rect").attr("x", 0).attr("y", yy).attr("width", Math.max(0, x(m.pv_harm_bn) - 1)).attr("height", bh).attr("fill", cH).attr("rx", 2);
      g.append("rect").attr("x", x(m.pv_harm_bn) + 1).attr("y", yy).attr("width", Math.max(0, x(m.pv_social_bn) - x(m.pv_harm_bn) - 1)).attr("height", bh).attr("fill", cC).attr("rx", 2);
      g.append("text").attr("x", x(m.pv_social_bn) + 6).attr("y", yy + bh / 2).attr("dy", "0.35em").attr("class", k === state.scen ? "lbl" : "")
        .style("font-weight", k === state.scen ? 650 : null).text(f2(m.pv_social_bn));
      F.g.append("text").attr("x", -8).attr("y", yy + bh / 2).attr("dy", "0.35em").attr("text-anchor", "end")
        .attr("class", k === state.scen ? "lbl" : "").style("font-weight", k === state.scen ? 650 : null)
        .text((k === state.scen ? "▸ " : "") + SHORT[k]);
      g.append("rect").attr("x", -F.m.l).attr("y", yy - 4).attr("width", F.iw + F.m.l + F.m.r).attr("height", bh + 8).attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("pointermove", (ev) => showTip(`<b>${D.scenarios[k].label}</b>${row("Health harm", bn(m.pv_harm_bn))}${row("Construction cost", bn(m.pv_cost_bn))}${row("Total", bn(m.pv_social_bn))}${row("vs. status quo", (m.pv_social_bn - D.scenarios.SQ.metrics.pv_social_bn < 0 ? "−" : "+") + bn(Math.abs(m.pv_social_bn - D.scenarios.SQ.metrics.pv_social_bn)))}`, ev))
        .on("pointerleave", hideTip)
        .on("click", () => { state.scen = k; $("#scenario").value = k; renderScenario(); });
    }
    void bg;
    F.el.innerHTML = legendHTML([{ label: "PV health harm", color: cH }, { label: "PV construction cost", color: cC }]);
    F.el.append(F.svg.node());
    F.el.insertAdjacentHTML("beforeend", `<p class="note">Robustness schedules are scored on the same central yardstick (central benefits, 3%) so all bars are comparable. Click a bar to select that scenario.</p>`);
  }

  // ---------- frontier
  function renderFrontier() {
    const el = $("#frontier");
    const eff = D.frontier.filter((d) => d.branch === "efficient").sort((a, b) => a.q5_gap - b.q5_gap);
    const dom = D.frontier.filter((d) => d.branch === "dominated").sort((a, b) => a.q5_gap - b.q5_gap);
    let keys = ["SQ", "PROP", "PRIO", "CM", "HW", "EW", "EWc", "DW"];
    if (!keys.includes(state.scen)) keys.push(state.scen);
    const pts = keys.map((k) => ({ k, x: D.scenarios[k].metrics.q5_gap, y: D.scenarios[k].metrics.pv_social_bn }));
    const zoom = state.frontierZoom;
    const shown = zoom ? pts.filter((p) => p.y < 29.5) : pts;
    const all = [...eff, ...dom].map((d) => ({ x: d.q5_gap, y: d.pv_social_bn })).concat(shown);
    const F = frame("#frontier", 400, { t: 14, r: 24, b: 44, l: 56 });
    const x = d3.scaleLinear().domain(d3.extent(all, (d) => d.x)).nice().range([0, F.iw]);
    const yd = d3.extent(all, (d) => d.y);
    const y = d3.scaleLinear().domain([yd[0] - (yd[1] - yd[0]) * 0.06, yd[1] + (yd[1] - yd[0]) * 0.06]).nice().range([F.ih, 0]);
    yGrid(F.g, y, F.iw, (v) => `$${d3.format(zoom ? ".1f" : ".0f")(v)}B`);
    xAxis(F.g, x, F.ih, (v) => signed(v, 0), 8);
    F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih + 36).attr("text-anchor", "middle").text(F.iw < 480 ? "Q5 wait gap, years" : "Q5 wait gap, years (negative = most vulnerable served earlier)");
    if (x.domain()[0] < 0 && x.domain()[1] > 0)
      F.g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text-3")).attr("stroke-dasharray", "2 3");
    const line = d3.line().x((d) => x(d.q5_gap)).y((d) => y(d.pv_social_bn));
    F.g.append("path").datum(dom).attr("fill", "none").attr("stroke", css("--neutral")).attr("stroke-width", 2).attr("stroke-dasharray", "5 4").attr("d", line);
    F.g.append("path").datum(eff).attr("fill", "none").attr("stroke", css("--s1")).attr("stroke-width", 2).attr("d", line);
    const tipF = (d) => `<b>${d.key === "HW" ? "Health-welfare optimum" : d.branch === "efficient" ? "Efficient branch" : "Dominated branch"}</b>${row("Q5 gap", signed(d.q5_gap, 2) + " yrs")}${row("PV social cost", bn(d.pv_social_bn))}${d.key !== "HW" ? row("vs. optimum", "+" + bn(d.pv_social_bn - D.scenarios.HW.metrics.pv_social_bn)) : ""}`;
    F.g.selectAll(null).data(eff.filter((d) => d.key !== "HW")).join("circle").attr("cx", (d) => x(d.q5_gap)).attr("cy", (d) => y(d.pv_social_bn)).attr("r", 4.5)
      .attr("fill", css("--s1")).attr("stroke", css("--bg")).attr("stroke-width", 2).on("pointermove", (ev, d) => showTip(tipF(d), ev)).on("pointerleave", hideTip);
    F.g.selectAll(null).data(dom.filter((d) => d.key !== "HW")).join("rect").attr("x", (d) => x(d.q5_gap) - 4).attr("y", (d) => y(d.pv_social_bn) - 4).attr("width", 8).attr("height", 8)
      .attr("fill", css("--bg")).attr("stroke", css("--neutral")).attr("stroke-width", 2).on("pointermove", (ev, d) => showTip(tipF(d), ev)).on("pointerleave", hideTip);
    const tc = css("--text");
    const sp = F.g.selectAll(null).data(shown).join("g").attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`).style("cursor", "pointer");
    sp.append("circle").attr("r", (d) => (d.k === state.scen ? 7 : 5)).attr("fill", (d) => (d.k === state.scen ? css("--s2") : tc))
      .attr("stroke", css("--bg")).attr("stroke-width", 2);
    const PLACE = { SQ: [-9, "0.35em", "end"], EWc: [-9, "0.35em", "end"], EW: [9, "-0.45em", "start"], HW: [9, "1.05em", "start"], DW: [0, "-0.9em", "middle"] };
    const pl = (d) => (d.k !== state.scen || !d.k.startsWith("HW_") ? PLACE[d.k] : null) || [9, d.k.startsWith("HW_") ? "1.9em" : "0.35em", "start"];
    sp.append("text").attr("x", (d) => pl(d)[0]).attr("dy", (d) => pl(d)[1]).attr("text-anchor", (d) => pl(d)[2])
      .attr("class", "lbl").style("font-weight", (d) => (d.k === state.scen ? 650 : 400)).style("font-size", "11.5px").text((d) => SHORT[d.k]);
    sp.on("pointermove", (ev, d) => showTip(`<b>${D.scenarios[d.k].label}</b>${row("Q5 gap", signed(d.x, 2) + " yrs")}${row("PV social cost", bn(d.y))}`, ev))
      .on("pointerleave", hideTip).on("click", (ev, d) => { state.scen = d.k; $("#scenario").value = d.k; renderScenario(); });
    el.innerHTML = `<div class="controls"><div class="seg"><span>View</span>
      <button type="button" role="radio" aria-checked="${!zoom}" data-z="0">All schedules</button>
      <button type="button" role="radio" aria-checked="${zoom}" data-z="1">Zoom to optimized</button></div></div>` +
      legendHTML([{ label: "Efficient branch (vulnerable served even earlier)", color: css("--s1"), kind: "line" },
        { label: "Dominated branch (vulnerable served later)", color: css("--neutral"), kind: "dash" },
        { label: "Scenario", color: tc }, { label: "Selected scenario", color: css("--s2") }]);
    el.append(F.svg.node());
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.frontierZoom = b.dataset.z === "1"; renderFrontier(); }));
  }

  // ---------- BCR by cost
  function renderBCR() {
    const tiers = [["child", "Child IQ only"], ["central", "Central (+ adult CVD, Aoki)"], ["high", "High (+ adult CVD, Lanphear)"]];
    const colors = [css("--s1"), css("--s2"), css("--s3")];
    const costs = [4700, 10000, 31000, 39000], clab = { 4700: "$4.7k EPA national", 10000: "$10k", 31000: "$31k Chicago avg.", 39000: "$39k one-off" };
    const rows = D.npv.filter((r) => Math.abs(r.rate - state.rate) < 1e-9);
    const F = frame("#bcr", 300, { t: 10, r: 10, b: 34, l: 40 });
    const x0 = d3.scaleBand().domain(costs).range([0, F.iw]).padding(0.22);
    const x1 = d3.scaleBand().domain(tiers.map((t) => t[0])).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, d3.max(D.npv, (r) => r.bcr)]).nice().range([F.ih, 0]);
    yGrid(F.g, y, F.iw, (v) => v);
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x0).tickFormat((c) => (F.iw < 380 ? `$${c / 1000}k` : clab[c])).tickSizeOuter(0));
    for (const r of rows) {
      const i = tiers.findIndex((t) => t[0] === r.tier);
      const bx = x0(r.cost) + x1(r.tier), bw = x1.bandwidth();
      F.g.append("path").attr("d", topRounded(bx, y(r.bcr), bw, F.ih - y(r.bcr), 3)).attr("fill", colors[i])
        .on("pointermove", (ev) => showTip(`<b>${tiers[i][1]}</b>${row("Cost per line", usd(r.cost))}${row("Discount rate", pct(r.rate))}${row("Benefit–cost ratio", f2(r.bcr))}${row("Net benefit per line", usd(r.npv))}`, ev))
        .on("pointerleave", hideTip);
    }
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(1)).attr("y2", y(1)).attr("stroke", css("--text")).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3");
    const el = F.el; el.innerHTML = legendHTML([...tiers.map((t, i) => ({ label: t[1], color: colors[i] })), { label: "Break-even (ratio = 1)", color: css("--text"), kind: "dash" }]); el.append(F.svg.node());
  }
  function topRounded(x, y, w, h, r) {
    r = Math.min(r, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  // ---------- tornado
  function renderTornado() {
    const T = D.tornado.slice().sort((a, b) => b.range - a.range);
    const narrow = $("#tornado").clientWidth < 420;
    const F = frame("#tornado", T.length * 34 + 50, { t: 6, r: 16, b: 34, l: narrow ? 120 : 170 });
    const base = T[0].base;
    const x = d3.scaleLinear().domain([Math.min(0.8, d3.min(T, (d) => Math.min(d.low, d.high))), d3.max(T, (d) => Math.max(d.low, d.high))]).nice().range([0, F.iw]);
    const y = d3.scaleBand().domain(T.map((d) => d.param)).range([0, F.ih]).padding(0.3);
    F.g.append("g").attr("class", "grid").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(6).tickSize(-F.ih).tickFormat(""));
    xAxis(F.g, x, F.ih, (v) => v, 6);
    const cLo = css("--s2"), cHi = css("--s1");
    for (const d of T) {
      const yy = y(d.param), bh = y.bandwidth();
      for (const [v, c, nm] of [[d.low, cLo, "Low end"], [d.high, cHi, "High end"]]) {
        const a = Math.min(base, v), b = Math.max(base, v);
        F.g.append("rect").attr("x", x(a)).attr("y", yy).attr("width", Math.max(1, x(b) - x(a))).attr("height", bh).attr("fill", c).attr("rx", 2)
          .on("pointermove", (ev) => showTip(`<b>${d.param}</b>${row(nm + " of range", "BCR " + f2(v))}${row("Base case", "BCR " + f2(base))}`, ev)).on("pointerleave", hideTip);
      }
      F.g.append("text").attr("x", -8).attr("y", yy + bh / 2).attr("dy", "0.35em").attr("text-anchor", "end").style("font-size", "11.5px").text(d.param.split(" (")[0]);
    }
    F.g.append("line").attr("x1", x(base)).attr("x2", x(base)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text")).attr("stroke-width", 1.2);
    F.g.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text")).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3");
    F.el.innerHTML = legendHTML([{ label: "Low end of parameter range", color: cLo }, { label: "High end", color: cHi }, { label: `Base case ${f2(base)}`, color: css("--text"), kind: "line" }, { label: "Break-even", color: css("--text"), kind: "dash" }]);
    F.el.append(F.svg.node());
  }

  // ---------- financing
  function renderFinance() {
    const modes = [["cash", "Pay cash"], ["dwsrf", `SRF loan (${pct1(D.meta.dwsrf_rate)})`], ["bond", `Revenue bond (${pct(D.meta.bond_rate)})`]];
    const colors = [css("--s1"), css("--s2"), css("--s3")];
    const F = frame("#finance", 300, { t: 10, r: 10, b: 40, l: 52 });
    const x0 = d3.scaleBand().domain(D.financing.map((d) => d.rate)).range([0, F.iw]).padding(0.25);
    const x1 = d3.scaleBand().domain(modes.map((m) => m[0])).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, d3.max(D.financing, (d) => Math.max(d.cash, d.dwsrf, d.bond))]).nice().range([F.ih, 0]);
    yGrid(F.g, y, F.iw, (v) => `$${v / 1000}k`);
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x0).tickFormat((r) => `${pct(r)} social discount rate`).tickSizeOuter(0));
    for (const d of D.financing) modes.forEach(([k, nm], i) => {
      const bx = x0(d.rate) + x1(k), bw = x1.bandwidth();
      F.g.append("path").attr("d", topRounded(bx, y(d[k]), bw, F.ih - y(d[k]), 3)).attr("fill", colors[i])
        .on("pointermove", (ev) => showTip(`<b>${nm}</b>${row("Social discount rate", pct(d.rate))}${row("PV cost of a $31k line", usd(d[k]))}${row("vs. cash", (d[k] >= d.cash ? "+" : "−") + usd(Math.abs(d[k] - d.cash)))}`, ev))
        .on("pointerleave", hideTip);
      if (bw > 34) F.g.append("text").attr("x", bx + bw / 2).attr("y", y(d[k]) - 5).attr("text-anchor", "middle").style("font-size", "11px").text(usdk(d[k]));
    });
    F.el.innerHTML = legendHTML(modes.map((m, i) => ({ label: m[1], color: colors[i] })));
    F.el.append(F.svg.node());
  }

  // ---------- PSA
  function renderPSA() {
    const P = D.psa;
    $("#psa-sub").innerHTML = `The probabilistic sensitivity analysis redraws the uncertain inputs together ${f0(P.n)} times: blood-lead increments, IQ slope, value per IQ point, the cardiovascular function, value of a statistical life, mobilization and per-line costs, and the discount rate. The health-welfare schedule beats the status quo in <strong>${pct1(P.hw_beats_sq)}</strong> of draws and is the cheapest of the six schedules tested in ${pct1(P.hw_best)}. Median saving <strong>${bn(P.save_q["0.5"])}</strong> (90% of draws between ${bn(P.save_q["0.05"])} and ${bn(P.save_q["0.95"])}); the per-line benefit–cost ratio at $31k has median <strong>${f2(P.bcr_q["0.5"])}</strong> and falls below 1 in ${pct1(P.bcr_below_one)} of draws.`;
    // histogram
    {
      const F = frame("#psa-hist", 260, { t: 10, r: 10, b: 40, l: 46 });
      const ext = d3.extent([...P.save_vs_sq, ...P.save_vs_cm]);
      const x = d3.scaleLinear().domain([Math.min(0, ext[0]), ext[1]]).nice().range([0, F.iw]);
      const bin = d3.bin().domain(x.domain()).thresholds(x.ticks(36));
      const A = bin(P.save_vs_sq), B = bin(P.save_vs_cm);
      const y = d3.scaleLinear().domain([0, d3.max([...A, ...B], (b) => b.length)]).nice().range([F.ih, 0]);
      yGrid(F.g, y, F.iw, (v) => v, 4);
      xAxis(F.g, x, F.ih, (v) => `$${v}B`, 6);
      F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih + 34).attr("text-anchor", "middle").text("PV saving, $ billion (draws per bin on the vertical axis)");
      const cA = css("--s1"), cB = css("--s2");
      F.g.selectAll(null).data(A).join("rect").attr("x", (b) => x(b.x0) + 1).attr("width", (b) => Math.max(0, x(b.x1) - x(b.x0) - 2))
        .attr("y", (b) => y(b.length)).attr("height", (b) => F.ih - y(b.length)).attr("fill", cA).attr("rx", 1.5);
      const step = (bins) => d3.line().curve(d3.curveStepAfter).x((d) => x(d.x)).y((d) => y(d.n))(
        bins.flatMap((b) => [{ x: b.x0, n: b.length }]).concat([{ x: bins.at(-1).x1, n: 0 }]));
      F.g.append("path").attr("d", step(B)).attr("fill", "none").attr("stroke", cB).attr("stroke-width", 2);
      F.g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text")).attr("stroke-width", 1.2);
      F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent").on("pointermove", (ev) => {
        const v = x.invert(d3.pointer(ev)[0]); const i = A.findIndex((b) => v >= b.x0 && v < b.x1); if (i < 0) return;
        showTip(`<b>$${f1(A[i].x0)}B to $${f1(A[i].x1)}B</b>${row("vs. status quo", A[i].length + " draws")}${row("vs. cost-minimizing", B[i].length + " draws")}`, ev);
      }).on("pointerleave", hideTip);
      F.el.innerHTML = legendHTML([{ label: "Saving vs. status quo", color: cA }, { label: "Saving vs. cost-minimizing", color: cB, kind: "line" }]);
      F.el.append(F.svg.node());
    }
    // CDF of BCR
    {
      const F = frame("#psa-cdf", 260, { t: 10, r: 14, b: 40, l: 46 });
      const rates = Object.keys(P.bcr_31k), colors = [css("--s1"), css("--s2"), css("--s3")];
      const allv = rates.flatMap((r) => P.bcr_31k[r]);
      const x = d3.scaleLog().domain([Math.min(0.8, d3.min(allv)), d3.max(allv)]).range([0, F.iw]);
      const y = d3.scaleLinear().domain([0, 1]).range([F.ih, 0]);
      yGrid(F.g, y, F.iw, pct, 4);
      F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickValues([1, 2, 5, 10, 20].filter((v) => v <= x.domain()[1])).tickFormat((v) => v).tickSizeOuter(0));
      F.g.append("text").attr("x", F.iw / 2).attr("y", F.ih + 34).attr("text-anchor", "middle").text("Benefit–cost ratio (log scale); share of draws at or below");
      rates.forEach((r, i) => {
        const v = P.bcr_31k[r];
        F.g.append("path").datum(v).attr("fill", "none").attr("stroke", colors[i]).attr("stroke-width", 2)
          .attr("d", d3.line().x((d) => x(d)).y((d, j) => y((j + 1) / v.length)));
      });
      F.g.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text")).attr("stroke-width", 1.5).attr("stroke-dasharray", "5 3");
      const cross = F.g.append("line").attr("y1", 0).attr("y2", F.ih).attr("stroke", css("--text-3")).style("display", "none");
      F.g.append("rect").attr("width", F.iw).attr("height", F.ih).attr("fill", "transparent").on("pointermove", (ev) => {
        const v = x.invert(d3.pointer(ev)[0]); cross.style("display", null).attr("x1", x(v)).attr("x2", x(v));
        showTip(`<b>BCR ≤ ${f2(v)}</b>${rates.map((r) => row(`r = ${pct(+r)}`, pct1(d3.bisectRight(P.bcr_31k[r], v) / P.bcr_31k[r].length) + " of draws")).join("")}`, ev);
      }).on("pointerleave", () => { cross.style("display", "none"); hideTip(); });
      F.el.innerHTML = legendHTML(rates.map((r, i) => ({ label: `Discount rate ${pct(+r)}`, color: colors[i], kind: "line" })));
      F.el.append(F.svg.node());
    }
  }
})();
