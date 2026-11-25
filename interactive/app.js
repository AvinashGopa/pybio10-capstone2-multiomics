/* Capstone-II interactive explorer.
   All data are real TCGA-BRCA results exported by scripts/export_interactive.py.
   Counting, thresholds, FDR, correlations, ROC and the Mann-Whitney test are recomputed here in the browser;
   model fits (PCA, logistic regression, leakage runs) were pre-computed in Python for every setting offered. */
'use strict';

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtP = p => (p === null || p === undefined) ? 'n/a' : (p < 1e-300 ? '< 1e-300' : p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const COL = { LumA: '#1f77b4', LumB: '#7fb8e6', Her2: '#d62728', Basal: '#2ca02c', none: '#c8cdd0',
              Tumour: '#c0392b', Normal: '#2e86c1', up: '#c0392b', down: '#2e86c1', ns: '#cfd4d6' };
const SUB = ['LumA', 'LumB', 'Her2', 'Basal'];
const SUBNAME = { LumA: 'Luminal A', LumB: 'Luminal B', Her2: 'HER2-enriched', Basal: 'Basal-like', none: 'no PAM50 label', Normal: 'Normal-like' };

const cache = {};
const J = name => cache[name] ??= fetch('data/' + name).then(r => { if (!r.ok) throw new Error(name); return r.json(); });
const B = (name, T) => cache[name] ??= fetch('data/' + name).then(r => { if (!r.ok) throw new Error(name); return r.arrayBuffer(); }).then(b => new T(b));

function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const sum = a => { let s = 0; for (const v of a) s += v; return s; };
const mean = a => sum(a) / a.length;
function quantile(a, q) { const s = Float64Array.from(a).sort(); const pos = (s.length - 1) * q, lo = Math.floor(pos);
  return lo + 1 < s.length ? s[lo] + (pos - lo) * (s[lo + 1] - s[lo]) : s[lo]; }
const median = a => quantile(a, 0.5);
function pearson(x, y) { const n = x.length; let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxy / Math.sqrt(sxx * syy); }
function erfc(x) { const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r; }
const normSf = z => 0.5 * erfc(z / Math.SQRT2);
function bh(p) { const n = p.length, idx = [...p.keys()].sort((a, b) => p[a] - p[b]), q = new Float64Array(n); let m = 1;
  for (let i = n - 1; i >= 0; i--) { const j = idx[i]; m = Math.min(m, p[j] * n / (i + 1)); q[j] = m; } return q; }
// Mann-Whitney U, two-sided, normal approximation with tie + continuity correction (as scipy.stats.mannwhitneyu)
function mannWhitney(inSet, values) { const n = values.length, idx = [...values.keys()].sort((a, b) => values[a] - values[b]);
  const rank = new Float64Array(n); let tie = 0;
  for (let i = 0; i < n;) { let j = i; while (j + 1 < n && values[idx[j + 1]] === values[idx[i]]) j++;
    const rk = (i + j) / 2 + 1, t = j - i + 1; tie += t * t * t - t; for (let k = i; k <= j; k++) rank[idx[k]] = rk; i = j + 1; }
  let n1 = 0, R1 = 0; for (let i = 0; i < n; i++) if (inSet[i]) { n1++; R1 += rank[i]; }
  const n2 = n - n1, U1 = R1 - n1 * (n1 + 1) / 2, U = Math.max(U1, n1 * n2 - U1), mu = n1 * n2 / 2;
  const s = Math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))); return Math.min(1, 2 * normSf((U - mu - 0.5) / s)); }
function rocCurve(truth, prob) { const idx = [...prob.keys()].sort((a, b) => prob[b] - prob[a]); const P = sum(truth), N = truth.length - P;
  const fpr = [0], tpr = [0], thr = [Infinity]; let tp = 0, fp = 0;
  for (let i = 0; i < idx.length; i++) { truth[idx[i]] ? tp++ : fp++;
    if (i === idx.length - 1 || prob[idx[i + 1]] !== prob[idx[i]]) { fpr.push(fp / N); tpr.push(tp / P); thr.push(prob[idx[i]]); } }
  let auc = 0; for (let i = 1; i < fpr.length; i++) auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
  return { fpr, tpr, thr, auc }; }

// ------------------------------------------------------------------ plotting
const AX = { gridcolor: '#e9edef', zeroline: false, linecolor: '#9aa5ab', ticks: 'outside', tickcolor: '#c4cbcf', automargin: true };
function plot(el, data, layout = {}) {
  const L = { font: { family: 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif', size: 12, color: '#1d2a31' },
    margin: { l: 55, r: 15, t: 30, b: 45, ...(layout.margin || {}) }, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
    legend: { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0, ...(layout.legend || {}) }, hovermode: 'closest' };
  for (const [k, v] of Object.entries(layout)) if (!['margin', 'legend'].includes(k)) L[k] = /^[xy]axis/.test(k) ? { ...AX, ...v } : v;
  if (!L.xaxis) L.xaxis = { ...AX }; if (!L.yaxis) L.yaxis = { ...AX };
  return Plotly.react(el, data, L, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] });
}
const vline = (x, color = '#0F1D26', dash = 'dash', yref = 'paper') => ({ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref, line: { color, width: 1.3, dash } });
const hline = (y, color = '#0F1D26', dash = 'dash', xref = 'paper', x0 = 0, x1 = 1) => ({ type: 'line', y0: y, y1: y, x0, x1, xref, line: { color, width: 1.3, dash } });
const chip = (label, value, cls = '') => `<div class="chip ${cls}"><span>${label}</span><b>${value}</b></div>`;

// ------------------------------------------------------------------ controls
function controls(root, specs, onChange) {
  const box = $('.controls', root), state = {};
  for (const s of specs) {
    const w = document.createElement('div'); w.className = 'ctl ctl-' + s.type; box.appendChild(w);
    const lab = `<span class="lab">${s.label}</span>`;
    if (s.type === 'range' || s.type === 'steps') {
      const vals = s.values, i0 = vals ? vals.indexOf(s.value) : s.value;
      w.innerHTML = `<label>${lab}<output></output><input type="range" min="${vals ? 0 : s.min}" max="${vals ? vals.length - 1 : s.max}" step="${vals ? 1 : s.step}" value="${i0}"></label>`;
      const inp = $('input', w), out = $('output', w);
      const read = () => { const v = vals ? vals[+inp.value] : +inp.value; state[s.key] = v; out.textContent = s.fmt ? s.fmt(v) : v; };
      read(); inp.addEventListener('input', () => { read(); onChange(s.key); });
      state['_set_' + s.key] = v => { inp.value = vals ? vals.indexOf(v) : v; read(); };
    } else if (s.type === 'select') {
      w.innerHTML = `<label>${lab}<select>${s.options.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select></label>`;
      const sel = $('select', w); sel.value = s.value; state[s.key] = s.value;
      sel.addEventListener('change', () => { state[s.key] = sel.value; onChange(s.key); });
      state['_set_' + s.key] = v => { sel.value = v; state[s.key] = v; };
      state['_el_' + s.key] = sel;
    } else if (s.type === 'checks') {
      w.innerHTML = `<fieldset>${lab}${s.options.map(([v, l, c]) => `<label class="chk"><input type="checkbox" value="${v}" checked><i style="background:${c || '#888'}"></i>${l}</label>`).join('')}</fieldset>`;
      const read = () => { state[s.key] = new Set([...w.querySelectorAll('input:checked')].map(i => i.value)); };
      read(); w.addEventListener('change', () => { read(); onChange(s.key); });
    } else if (s.type === 'search') {
      w.innerHTML = `<label>${lab}<input type="text" placeholder="${s.placeholder || ''}" autocomplete="off" spellcheck="false"></label><div class="sugg"></div>`;
      const inp = $('input', w), sugg = $('.sugg', w); state[s.key] = s.value || '';
      inp.value = s.value || '';
      const pick = v => { inp.value = v; state[s.key] = v; sugg.innerHTML = ''; onChange(s.key); };
      inp.addEventListener('input', () => {
        const q = inp.value.trim().toUpperCase(); sugg.innerHTML = '';
        if (!q) { if (s.allowEmpty) pick(''); return; }
        const list = s.list(), hits = [];
        for (const v of list) { if (v.toUpperCase().startsWith(q)) hits.push(v); if (hits.length >= 8) break; }
        sugg.innerHTML = hits.map(h => `<button type="button">${h}</button>`).join('');
        const exact = list.find(v => v.toUpperCase() === q); if (exact) { state[s.key] = exact; onChange(s.key); }
      });
      sugg.addEventListener('click', e => { if (e.target.tagName === 'BUTTON') pick(e.target.textContent); });
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { const b = $('button', sugg); if (b) pick(b.textContent); } });
      state['_set_' + s.key] = v => { inp.value = v; state[s.key] = v; sugg.innerHTML = ''; };
    } else if (s.type === 'buttons') {
      w.innerHTML = `<div class="btns">${lab}${s.options.map(([v, l]) => `<button type="button" data-v="${v}">${l}</button>`).join('')}</div>`;
      w.addEventListener('click', e => { if (e.target.dataset.v !== undefined) s.onClick(e.target.dataset.v); });
    } else if (s.type === 'textarea') {
      w.innerHTML = `<label>${lab}<textarea rows="3" spellcheck="false">${s.value}</textarea></label>`;
      const ta = $('textarea', w); state[s.key] = s.value;
      ta.addEventListener('input', () => { state[s.key] = ta.value; onChange(s.key); });
      state['_set_' + s.key] = v => { ta.value = v; state[s.key] = v; };
    }
  }
  return state;
}
const plotsIn = (root, n) => { const p = $('.plots', root); p.innerHTML = ''; p.style.setProperty('--n', n);
  return Array.from({ length: n }, () => { const d = document.createElement('div'); d.className = 'plot'; p.appendChild(d); return d; }); };
const readout = (root, html) => { $('.readout', root).innerHTML = html; };

// ================================================================== WIDGETS
const W = {};

// ---------- NB1: sample QC thresholds -----------------------------------------------------------
W.qc = async root => {
  const d = await J('nb1_qc.json'); const n = d.id.length;
  const [p1, p2] = plotsIn(root, 2);
  const st = controls(root, [
    { key: 'metric', type: 'select', label: 'Histogram of', value: 'lib', options: [['lib', 'Library size (million reads)'], ['det', 'Genes detected'], ['mito', '% mitochondrial reads']] },
    { key: 'lib', type: 'range', label: 'Minimum library size (M reads)', min: 0, max: 40, step: 1, value: 10 },
    { key: 'det', type: 'range', label: 'Minimum genes detected', min: 10000, max: 19000, step: 250, value: 14000, fmt: fmt },
    { key: 'mito', type: 'range', label: 'Flag if mitochondrial reads above (%)', min: 5, max: 50, step: 1, value: 20 },
    { key: 'k', type: 'range', label: 'Outlier rule: median correlation below Q1 − k × IQR, k', min: 1, max: 5, step: 0.5, value: 3 },
  ], update);
  function update() {
    const cut = {};
    for (const t of ['Tumour', 'Normal']) { const v = d.medcorr.filter((_, i) => d.type[i] === t);
      const q1 = quantile(v, 0.25), q3 = quantile(v, 0.75); cut[t] = q1 - st.k * (q3 - q1); }
    const low = [], few = [], mito = [], out = [];
    for (let i = 0; i < n; i++) { if (d.lib[i] < st.lib) low.push(i); if (d.det[i] < st.det) few.push(i);
      if (d.mito[i] > st.mito) mito.push(i); if (d.medcorr[i] < cut[d.type[i]]) out.push(i); }
    const removed = new Set([...low, ...few, ...out]);
    const m = st.metric, lineAt = { lib: st.lib, det: st.det, mito: st.mito }[m];
    const label = { lib: 'Library size (million reads)', det: 'Genes detected (of 19,944)', mito: '% reads from mitochondrial genes' }[m];
    plot(p1, ['Tumour', 'Normal'].map(t => ({ type: 'histogram', name: t, x: d[m].filter((_, i) => d.type[i] === t), nbinsx: 45,
      marker: { color: COL[t] }, opacity: 0.65 })), { barmode: 'overlay', shapes: [vline(lineAt)],
      xaxis: { title: { text: label } }, yaxis: { title: { text: 'Samples' } } });
    const R = rng(7), traces = [];
    ['Tumour', 'Normal'].forEach((t, k) => { const idx = d.id.map((_, i) => i).filter(i => d.type[i] === t);
      for (const flag of [false, true]) { const sel = idx.filter(i => (d.medcorr[i] < cut[t]) === flag);
        traces.push({ type: 'scatter', mode: 'markers', name: flag ? `${t}: outlier` : t, x: sel.map(() => k + (R() - 0.5) * 0.6), y: sel.map(i => d.medcorr[i]),
          text: sel.map(i => d.id[i]), hovertemplate: '%{text}<br>median r = %{y:.3f}<extra></extra>',
          marker: { color: flag ? '#000' : COL[t], size: flag ? 11 : 5, symbol: flag ? 'x' : 'circle', opacity: flag ? 1 : 0.55 } }); } });
    plot(p2, traces, { showlegend: false, xaxis: { tickvals: [0, 1], ticktext: ['Tumour', 'Normal'], range: [-0.6, 1.6] },
      yaxis: { title: { text: 'Median correlation with its own group' } },
      shapes: [hline(cut.Tumour, '#0F1D26', 'dash', 'x', -0.45, 0.45), hline(cut.Normal, '#0F1D26', 'dash', 'x', 0.55, 1.45)] });
    const names = a => a.length ? a.slice(0, 6).map(i => `<code>${d.id[i]}</code>`).join(' ') + (a.length > 6 ? ' …' : '') : '';
    readout(root, `<div class="chips">${chip('Low library size', low.length, low.length ? 'bad' : '')}${chip('Too few genes', few.length, few.length ? 'bad' : '')}
      ${chip('High mitochondrial (flag only)', mito.length, mito.length ? 'warn' : '')}${chip('Correlation outliers', out.length, out.length ? 'bad' : '')}
      ${chip('Samples removed', removed.size, removed.size ? 'bad' : '')}${chip('Samples kept', fmt(n - removed.size))}</div>
      <p>${out.length ? 'Outliers: ' + names(out) + '. ' : ''}${mito.length ? 'High-mito flags: ' + names(mito) + '.' : ''}
      Outlier cut-offs: tumour r &lt; ${cut.Tumour.toFixed(3)}, normal r &lt; ${cut.Normal.toFixed(3)}.</p>`);
  }
  update();
};

// ---------- NB1: gene filter -------------------------------------------------------------------
W.genefilter = async root => {
  const d = await J('nb1_genefilter.json');
  const [p1, p2] = plotsIn(root, 2);
  const st = controls(root, [
    { key: 'cut', type: 'select', label: 'A gene counts as “expressed” in a sample if CPM ≥', value: '1', options: d.cuts.map(c => [String(c), String(c)]) },
    { key: 'frac', type: 'range', label: 'Keep a gene if expressed in at least this % of samples', min: 0, max: 100, step: 5, value: 50, fmt: v => v + ' %' },
  ], update);
  function update() {
    const f = d.frac[st.cut], thr = st.frac / 100, kept = [], gone = [];
    f.forEach((v, i) => (v >= thr ? kept : gone).push(d.mean_logcpm[i]));
    plot(p1, [{ type: 'histogram', x: gone, name: `removed (${fmt(gone.length)})`, marker: { color: '#c8cdd0' }, xbins: { size: 0.25 } },
      { type: 'histogram', x: kept, name: `kept (${fmt(kept.length)})`, marker: { color: '#1F7A6E' }, xbins: { size: 0.25 } }],
      { barmode: 'stack', xaxis: { title: { text: 'Mean log2(CPM + 1) of the gene' } }, yaxis: { title: { text: 'Genes' } } });
    const xs = [...Array(101).keys()], ys = xs.map(x => f.filter(v => v >= x / 100).length);
    plot(p2, [{ type: 'scatter', mode: 'lines', x: xs, y: ys, line: { color: '#0F1D26' }, hovertemplate: '≥ %{x} % of samples: %{y:,} genes<extra></extra>' },
      { type: 'scatter', mode: 'markers', x: [st.frac], y: [kept.length], marker: { size: 12, color: '#C0632B' }, hoverinfo: 'skip' }],
      { showlegend: false, xaxis: { title: { text: '% of samples required' } }, yaxis: { title: { text: 'Genes kept' }, rangemode: 'tozero' } });
    readout(root, `<div class="chips">${chip('Genes kept', fmt(kept.length), 'good')}${chip('Removed', fmt(gone.length))}
      ${chip('Share kept', (kept.length / d.n * 100).toFixed(0) + ' %')}</div>`);
  }
  update();
};

// ---------- NB1: PCA of all samples ----------------------------------------------------------------
W.pca1 = async root => {
  const d = await J('nb1_qc.json'); const n = d.id.length;
  const [p1] = plotsIn(root, 1);
  const pcOpt = [1, 2, 3, 4, 5].map(i => [String(i - 1), 'PC' + i]);
  const st = controls(root, [
    { key: 'ntop', type: 'select', label: 'Use the most variable genes', value: '2000', options: ['500', '1000', '2000', '5000'].map(v => [v, fmt(+v) + ' genes']) },
    { key: 'x', type: 'select', label: 'x-axis', value: '0', options: pcOpt },
    { key: 'y', type: 'select', label: 'y-axis', value: '1', options: pcOpt },
    { key: 'color', type: 'select', label: 'Colour by', value: 'type', options: [['type', 'Tumour / normal'], ['pam50', 'PAM50 subtype'], ['mito', '> 20 % mitochondrial reads']] },
  ], update);
  function update() {
    const P = d.pca[st.ntop], x = P.pc[+st.x], y = P.pc[+st.y];
    const group = i => st.color === 'type' ? d.type[i] : st.color === 'pam50' ? (d.type[i] === 'Normal' ? 'normal tissue' : d.pam50[i]) : (d.mito[i] > 20 ? '> 20 % mito' : '≤ 20 % mito');
    const groups = [...new Set(d.id.map((_, i) => group(i)))];
    const colorOf = g => ({ ...COL, 'normal tissue': '#2e86c1', Normal: st.color === 'pam50' ? '#9b59b6' : COL.Normal, '> 20 % mito': '#000', '≤ 20 % mito': '#c8cdd0' })[g] || '#888';
    plot(p1, groups.map(g => { const idx = d.id.map((_, i) => i).filter(i => group(i) === g);
      return { type: 'scattergl', mode: 'markers', name: SUBNAME[g] && st.color === 'pam50' ? SUBNAME[g] : g, x: idx.map(i => x[i]), y: idx.map(i => y[i]),
        text: idx.map(i => `${d.id[i]}<br>${d.type[i]}${d.type[i] === 'Tumour' ? ' · ' + d.pam50[i] : ''}<br>mito ${d.mito[i]} %`),
        hovertemplate: '%{text}<extra></extra>', marker: { size: g === '> 20 % mito' ? 9 : 5, color: colorOf(g), opacity: 0.7 } }; }),
      { xaxis: { title: { text: `PC${+st.x + 1} (${P.var[+st.x]} % of variation)` } }, yaxis: { title: { text: `PC${+st.y + 1} (${P.var[+st.y]} %)` } } });
    readout(root, `<div class="chips">${P.var.map((v, i) => chip('PC' + (i + 1), v + ' %')).join('')}</div>`);
  }
  update();
};

// ---------- NB1: sex check ---------------------------------------------------------------------------
W.sex = async root => {
  const d = await J('nb1_qc.json'); const n = d.id.length; const G = ['RPS4Y1', 'DDX3Y', 'KDM5D', 'UTY'];
  const meanScore = d.id.map((_, i) => mean(G.map(g => d.y[g][i])));
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'marker', type: 'select', label: 'Score', value: 'mean', options: [['mean', 'Mean of the 4 Y genes'], ...G.map(g => [g, g + ' alone'])] },
    { key: 'thr', type: 'range', label: 'Call “male” if the score is above (log2 CPM)', min: 0, max: 6, step: 0.25, value: 2 },
  ], update);
  function update() {
    const s = st.marker === 'mean' ? meanScore : d.y[st.marker], R = rng(3), traces = [];
    const mism = []; for (let i = 0; i < n; i++) { const m = (s[i] > st.thr && d.gender[i] === 'Female') || (s[i] <= st.thr && d.gender[i] === 'Male'); if (m) mism.push(i); }
    const ms = new Set(mism);
    ['Female', 'Male'].forEach((g, k) => { for (const bad of [false, true]) { const idx = d.id.map((_, i) => i).filter(i => d.gender[i] === g && ms.has(i) === bad);
      traces.push({ type: 'scattergl', mode: 'markers', name: bad ? 'disagrees with record' : g, showlegend: k === 0 || bad,
        x: idx.map(() => k + (R() - 0.5) * 0.55), y: idx.map(i => s[i]), text: idx.map(i => `${d.id[i]} (${d.type[i]})`),
        hovertemplate: '%{text}<br>score %{y:.2f}<extra></extra>',
        marker: { color: bad ? '#000' : (g === 'Male' ? '#2471A3' : '#C0632B'), size: bad ? 12 : 5, symbol: bad ? 'x' : 'circle', opacity: bad ? 1 : 0.5 } }); } });
    plot(p1, traces, { xaxis: { tickvals: [0, 1], ticktext: ['Recorded female', 'Recorded male'], range: [-0.6, 1.6] },
      yaxis: { title: { text: st.marker === 'mean' ? 'Y score (mean log2 CPM of 4 genes)' : st.marker + ' (log2 CPM)' } }, shapes: [hline(st.thr)] });
    readout(root, `<div class="chips">${chip('Samples disagreeing with the record', mism.length, mism.length ? 'bad' : 'good')}${chip('Recorded male', d.gender.filter(g => g === 'Male').length)}</div>
      ${mism.length ? '<p>' + mism.slice(0, 8).map(i => `<code>${d.id[i]}</code> (${d.gender[i]}, score ${s[i].toFixed(2)})`).join(', ') + (mism.length > 8 ? ' …' : '') + '</p>' : ''}`);
  }
  update();
};

// ---------- NB1: protein missing values ---------------------------------------------------------------
W.protmiss = async root => {
  const d = await J('nb1_protein_missing.json'); const P = d.proteins.length, S = d.n_samples;
  const perP = d.missing.map(m => m.length / S * 100), cntS = new Array(S).fill(0);
  d.missing.forEach(m => m.forEach(j => cntS[j]++)); const perS = cntS.map(c => c / P * 100);
  const [p1, p2] = plotsIn(root, 2);
  const st = controls(root, [
    { key: 'pp', type: 'range', label: 'Drop a protein if more than this % of its values are missing', min: 0, max: 100, step: 1, value: 20, fmt: v => v + ' %' },
    { key: 'ss', type: 'range', label: 'Drop a sample if more than this % of its values are missing', min: 0, max: 100, step: 1, value: 50, fmt: v => v + ' %' },
  ], update);
  function update() {
    const goodP = perP.map(v => v <= st.pp), goodS = perS.map(v => v <= st.ss);
    let left = 0; d.missing.forEach((m, i) => { if (goodP[i]) for (const j of m) if (goodS[j]) left++; });
    const nP = goodP.filter(Boolean).length, nS = goodS.filter(Boolean).length;
    const h = (x, cut, t) => [{ type: 'histogram', x, xbins: { start: 0, end: 100, size: 2 }, marker: { color: '#7f8c8d' } }];
    plot(p1, h(perP), { showlegend: false, shapes: [vline(st.pp)], xaxis: { title: { text: '% missing per protein' } }, yaxis: { title: { text: 'Proteins' }, type: 'log' } });
    plot(p2, h(perS), { showlegend: false, shapes: [vline(st.ss)], xaxis: { title: { text: '% missing per sample' } }, yaxis: { title: { text: 'Samples' }, type: 'log' } });
    readout(root, `<div class="chips">${chip('Proteins kept', `${nP} of ${P}`)}${chip('Samples kept', `${nS} of ${S}`)}
      ${chip('Values still missing → median-imputed', fmt(left), left > 5000 ? 'bad' : '')}${chip('Share imputed', nP && nS ? (left / (nP * nS) * 100).toFixed(1) + ' %' : '–')}</div>`);
  }
  update();
};

// ---------- NB2: volcano / MA ---------------------------------------------------------------------------
W.volcano = async root => {
  const [de, mi] = await Promise.all([J('nb2_de.json'), J('nb2_mirna_de.json')]);
  const SETS = {
    tt: { name: de.symbol, x: de.lfc, q: de.fdr, a: de.symbol.map((_, i) => (de.mean_normal[i] + de.mean_tumour[i]) / 2), what: 'genes', label: 'paired t-test on log2(CPM + 1)' },
    ds: { name: de.symbol, x: de.ds_lfc, q: de.ds_padj, a: de.ds_base.map(v => Math.log2(v + 1)), what: 'genes', label: 'PyDESeq2, ~ patient + tissue' },
    un: { name: de.symbol, x: de.lfc, q: de.fdr_unpaired, a: de.symbol.map((_, i) => (de.mean_normal[i] + de.mean_tumour[i]) / 2), what: 'genes', label: 'unpaired Welch t-test (pairing ignored)' },
    mi: { name: mi.name, x: mi.lfc, q: mi.fdr, a: null, what: 'miRNAs', label: 'miRNA, paired t-test' },
  };
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'set', type: 'select', label: 'Data and test', value: 'tt', options: [['tt', 'mRNA · paired t-test (Notebook 2)'], ['ds', 'mRNA · PyDESeq2 count model'], ['un', 'mRNA · unpaired test (pairing ignored)'], ['mi', 'miRNA · paired t-test']] },
    { key: 'view', type: 'select', label: 'Plot', value: 'volcano', options: [['volcano', 'Volcano plot'], ['ma', 'MA plot (fold change vs expression)']] },
    { key: 'fdr', type: 'steps', label: 'FDR cut-off', values: [1e-10, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.2], value: 0.05, fmt: v => v < 0.001 ? v.toExponential(0) : v },
    { key: 'lfc', type: 'range', label: 'Minimum |log2 fold change|', min: 0, max: 4, step: 0.25, value: 1, fmt: v => `${v}  (${(2 ** v).toFixed(v % 1 ? 1 : 0)}-fold)` },
    { key: 'lab', type: 'range', label: 'Label the most changed (each side)', min: 0, max: 15, step: 1, value: 6 },
    { key: 'gene', type: 'search', label: 'Find a gene / miRNA', placeholder: 'e.g. ESR1, MKI67, mir-21', allowEmpty: true, list: () => SETS[st.set].name },
  ], update);
  function update() {
    const S = SETS[st.set], n = S.x.length, view = st.view === 'ma' && S.a ? 'ma' : 'volcano';
    const cls = new Array(n), groups = { up: [], down: [], ns: [] };
    for (let i = 0; i < n; i++) { const q = S.q[i]; let c = 'ns';
      if (q !== null && q < st.fdr && S.x[i] > st.lfc) c = 'up'; else if (q !== null && q < st.fdr && S.x[i] < -st.lfc) c = 'down';
      cls[i] = c; groups[c].push(i); }
    const Y = i => view === 'ma' ? S.x[i] : -Math.log10(Math.max(S.q[i] ?? 1, 1e-300)), X = i => view === 'ma' ? S.a[i] : S.x[i];
    const names = { up: 'Up in tumour', down: 'Down in tumour', ns: 'Not significant' };
    const tr = ['ns', 'down', 'up'].map(c => ({ type: 'scattergl', mode: 'markers', name: `${names[c]} (${fmt(groups[c].length)})`,
      x: groups[c].map(X), y: groups[c].map(Y), text: groups[c].map(i => `<b>${S.name[i]}</b><br>log2FC ${S.x[i].toFixed(2)} (${(2 ** Math.abs(S.x[i])).toFixed(1)}-fold ${S.x[i] > 0 ? 'up' : 'down'})<br>FDR ${fmtP(S.q[i])}`),
      hovertemplate: '%{text}<extra></extra>', marker: { color: COL[c], size: c === 'ns' ? 4 : 5, opacity: c === 'ns' ? 0.5 : 0.75 } }));
    const ann = [];
    const top = (arr, dir) => arr.slice().sort((a, b) => dir * (S.x[b] - S.x[a])).slice(0, st.lab);
    for (const [arr, dir] of [[groups.up, 1], [groups.down, -1]]) {
      const pts = top(arr, dir).sort((a, b) => Y(b) - Y(a));          // label from the top down, stacking the text
      let last = Infinity;
      pts.forEach(i => { const yy = Math.min(Y(i), last - (view === 'ma' ? 0.55 : 2.8)); last = yy;
        ann.push({ x: X(i), y: Y(i), ax: X(i) + dir * (view === 'ma' ? 0.9 : 1.1), ay: yy, axref: 'x', ayref: 'y', text: S.name[i], showarrow: true, arrowhead: 0,
          arrowwidth: 0.6, arrowcolor: '#888', xanchor: dir > 0 ? 'left' : 'right', font: { size: 10 } }); }); }
    let found = '';
    if (st.gene) { const gi = S.name.findIndex(v => v.toUpperCase() === st.gene.toUpperCase());
      if (gi >= 0) { tr.push({ type: 'scatter', mode: 'markers', name: S.name[gi], x: [X(gi)], y: [Y(gi)], hoverinfo: 'skip', marker: { size: 16, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: '#C0632B' } } });
        ann.push({ x: X(gi), y: Y(gi), text: `<b>${S.name[gi]}</b>`, showarrow: true, arrowcolor: '#C0632B', ax: 40, ay: -40, font: { size: 13, color: '#C0632B' }, bgcolor: '#fff' });
        found = `<p class="found"><b>${S.name[gi]}</b>: log2FC ${S.x[gi].toFixed(2)} (${(2 ** Math.abs(S.x[gi])).toFixed(1)}-fold ${S.x[gi] > 0 ? 'higher' : 'lower'} in tumour), FDR ${fmtP(S.q[gi])} → <b>${names[cls[gi]]}</b> at the current cut-offs.</p>`; }
      else found = `<p class="found">“${st.gene}” is not among the ${fmt(n)} ${S.what} tested here (too low expressed, or another name).</p>`; }
    const sh = view === 'ma' ? [hline(0, '#0F1D26', 'solid'), hline(st.lfc, '#888'), hline(-st.lfc, '#888')]
      : [vline(st.lfc, '#888', 'dash', 'paper'), vline(-st.lfc, '#888', 'dash', 'paper'), hline(-Math.log10(st.fdr), '#888')];
    plot(p1, tr, { annotations: ann, shapes: sh, height: 520,
      xaxis: view === 'ma' ? { title: { text: st.set === 'ds' ? 'log2(base mean + 1)' : 'Average expression, log2(CPM + 1)' } } : { title: { text: 'log2 fold change (tumour vs own normal)' }, range: st.set === 'mi' ? [-6.5, 6.5] : [-9.5, 9.5] },
      yaxis: { title: { text: view === 'ma' ? 'log2 fold change' : '−log10(FDR)' } } });
    const nSig = groups.up.length + groups.down.length;
    readout(root, `<div class="chips">${chip('Up in tumour', fmt(groups.up.length), 'up')}${chip('Down in tumour', fmt(groups.down.length), 'down')}
      ${chip(`Significant ${S.what}`, fmt(nSig))}${chip(`${S.what} tested`, fmt(n))}${chip('Expected false discoveries ≈ FDR × list', st.fdr * nSig < 1 ? '< 1' : fmt(Math.round(st.fdr * nSig)))}</div>
      <p class="muted">${S.label}.${st.view === 'ma' && !S.a ? ' MA plot needs average expression, which the miRNA table does not have: showing the volcano.' : ''}</p>${found}`);
  }
  update();
};

// ---------- NB2: multiple testing ------------------------------------------------------------------------
W.pvals = async root => {
  const de = await J('nb2_de.json'); const n = de.p.length;
  const R = rng(2026); const nullP = Array.from({ length: n }, () => R());
  const SRC = { real: de.p, un: de.p_unpaired, nul: nullP }, Q = {};
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'src', type: 'select', label: 'p-values from', value: 'real', options: [['real', 'Real data: paired t-test, tumour vs normal'], ['un', 'Real data: unpaired test'], ['nul', 'Simulation: no gene changes at all (random p)']] },
    { key: 'a', type: 'steps', label: 'Significance level α (for p, FDR and Bonferroni)', values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], value: 0.05 },
    { key: 'bins', type: 'range', label: 'Histogram bins', min: 10, max: 100, step: 10, value: 50 },
  ], update);
  function update() {
    const p = SRC[st.src]; const q = Q[st.src] ??= bh(p);
    let raw = 0, fdr = 0, bon = 0; for (let i = 0; i < n; i++) { if (p[i] < st.a) raw++; if (q[i] < st.a) fdr++; if (p[i] < st.a / n) bon++; }
    plot(p1, [{ type: 'histogram', x: p, xbins: { start: 0, end: 1, size: 1 / st.bins }, marker: { color: '#7f8c8d', line: { color: '#fff', width: 0.5 } }, name: 'genes' }],
      { showlegend: false, shapes: [hline(n / st.bins, '#C0632B', 'dash', 'paper'), vline(st.a, '#0F1D26', 'dot')],
        annotations: [{ xref: 'paper', x: 0.98, y: n / st.bins, text: 'height if no gene changed', showarrow: false, yshift: 10, xanchor: 'right', font: { color: '#C0632B' } }],
        xaxis: { title: { text: 'p-value' }, range: [0, 1] }, yaxis: { title: { text: 'Number of genes' } } });
    readout(root, `<div class="chips">${chip('Genes tested', fmt(n))}${chip(`p < ${st.a}`, fmt(raw))}${chip('…expected by luck alone (α × genes)', fmt(Math.round(st.a * n)), 'warn')}
      ${chip(`FDR (Benjamini–Hochberg) < ${st.a}`, fmt(fdr), 'good')}${chip(`Bonferroni (p < α / ${fmt(n)})`, fmt(bon))}</div>`);
  }
  update();
};

// ---------- NB2: one gene, 111 patients ------------------------------------------------------------------
W.geneexp = async root => {
  const [de, meta, M] = await Promise.all([J('nb2_de.json'), J('nb2_pairs_meta.json'), B('nb2_pairs_u8.bin', Uint8Array)]);
  const np = meta.patients.length, ns = 2 * np;
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'gene', type: 'search', label: 'Gene', value: 'COL10A1', placeholder: 'type a gene symbol', list: () => de.symbol },
    { key: 'pre', type: 'buttons', label: 'Examples', options: ['COL10A1', 'MMP11', 'TOP2A', 'MKI67', 'FABP4', 'ADIPOQ', 'ESR1', 'ERBB2', 'GAPDH'].map(g => [g, g]),
      onClick: v => { st._set_gene(v); update(); } },
  ], update);
  function update() {
    const gi = de.symbol.findIndex(v => v.toUpperCase() === (st.gene || '').toUpperCase());
    if (gi < 0) { readout(root, `<p class="found">“${st.gene}” is not among the ${fmt(de.symbol.length)} genes tested (filtered out as too low, or another symbol).</p>`); return; }
    const row = M.subarray(gi * ns, gi * ns + ns), T = [], N = [];
    for (let k = 0; k < np; k++) { T.push(row[k] / meta.scale); N.push(row[np + k] / meta.scale); }
    const lx = [], ly = [], up = [], dn = [];
    for (let k = 0; k < np; k++) (T[k] >= N[k] ? up : dn).push(k);
    const lines = (ks, color, name) => { const x = [], y = []; for (const k of ks) { x.push(0, 1, null); y.push(N[k], T[k], null); }
      return { type: 'scatter', mode: 'lines', x, y, name, line: { color, width: 1 }, opacity: 0.35, hoverinfo: 'skip' }; };
    plot(p1, [lines(up, '#c0392b', `higher in tumour (${up.length})`), lines(dn, '#2e86c1', `lower in tumour (${dn.length})`),
      { type: 'box', x: N.map(() => -0.12), y: N, name: 'Normal', marker: { color: COL.Normal }, width: 0.12, boxpoints: false, showlegend: false },
      { type: 'box', x: T.map(() => 1.12), y: T, name: 'Tumour', marker: { color: COL.Tumour }, width: 0.12, boxpoints: false, showlegend: false }],
      { xaxis: { tickvals: [0, 1], ticktext: ['Normal tissue', 'Tumour (same patient)'], range: [-0.35, 1.35] }, yaxis: { title: { text: `${de.symbol[gi]}  log2(CPM + 1)` } }, height: 460 });
    const fdr = de.fdr[gi], call = fdr < 0.05 && Math.abs(de.lfc[gi]) > 1 ? (de.lfc[gi] > 0 ? 'Up in tumour' : 'Down in tumour') : 'Not significant (FDR < 0.05 and ≥ 2-fold)';
    readout(root, `<div class="chips">${chip('log2 fold change', de.lfc[gi].toFixed(2), de.lfc[gi] > 0 ? 'up' : 'down')}${chip('Fold', (2 ** Math.abs(de.lfc[gi])).toFixed(1) + '× ' + (de.lfc[gi] > 0 ? 'higher' : 'lower'))}
      ${chip('FDR (paired t-test)', fmtP(fdr))}${chip('Patients changing in that direction', de.same_dir[gi] + ' %')}
      ${chip('PyDESeq2 log2FC', de.ds_lfc[gi] === null ? 'n/a' : de.ds_lfc[gi].toFixed(2))}${chip('PyDESeq2 padj', fmtP(de.ds_padj[gi]))}</div>
      <p>Mean log2(CPM + 1): normal ${de.mean_normal[gi]}, tumour ${de.mean_tumour[gi]} → <b>${call}</b> in Notebook 2.
      <span class="muted">Lines are drawn from values rounded to 0.1; the statistics are exact.</span></p>`);
  }
  update();
};

// ---------- NB2: gene-set ("theme") test ------------------------------------------------------------------
W.themes = async root => {
  const de = await J('nb2_de.json'); const n = de.symbol.length, upper = de.symbol.map(s => s.toUpperCase());
  const THEMES = {
    'Cell cycle / proliferation': 'MKI67 TOP2A CCNB1 CDK1 BIRC5 AURKA BUB1 CENPF KIF2C MCM2 PLK1 CCNA2 TPX2 UBE2C CDC20 FOXM1 MYBL2 PTTG1 RRM2 NEK2',
    'Matrix remodelling (fibroblasts)': 'COL1A1 COL1A2 COL3A1 COL5A1 COL10A1 COL11A1 FN1 POSTN MMP11 MMP13 MMP1 COMP INHBA LRRC15 SULF1 FAP CTHRC1',
    'Fat cells (adipocytes)': 'ADIPOQ LEP PLIN1 PLIN4 FABP4 CIDEA CIDEC LPL CD36 PPARG LIPE GPD1 PCK1 AQP7 RBP4 TRARG1',
    'Myoepithelial / basal layer': 'KRT5 KRT14 KRT17 TP63 MYH11 OXTR KIT ACTG2 CNN1 MYLK',
    'Immune cells (try it)': 'PTPRC CD3E CD8A CD4 MS4A1 CD19 CD68 CD163 GZMB PRF1 NKG7 LCK',
    'Housekeeping (try it)': 'GAPDH ACTB TUBB B2M PPIA RPLP0 HPRT1 TBP PGK1 YWHAZ',
  };
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'pre', type: 'buttons', label: 'Load a gene set', options: Object.keys(THEMES).map(k => [k, k]), onClick: v => { st._set_genes(THEMES[v]); update(); } },
    { key: 'genes', type: 'textarea', label: 'Genes in the set (any separator) – edit freely', value: THEMES['Fat cells (adipocytes)'] },
  ], update);
  function update() {
    const want = [...new Set(st.genes.toUpperCase().split(/[^A-Z0-9.\-]+/).filter(Boolean))];
    const W2 = new Set(want), inSet = upper.map(s => W2.has(s)), idx = inSet.flatMap((v, i) => v ? [i] : []);
    const missing = want.filter(g => !upper.includes(g));
    if (!idx.length) { plot(p1, [], {}); readout(root, '<p class="found">None of these genes were tested (check the symbols).</p>'); return; }
    const lf = idx.map(i => de.lfc[i]), p = mannWhitney(inSet, de.t);
    plot(p1, [{ type: 'box', x: de.lfc, name: 'All genes', orientation: 'h', marker: { color: '#95a5a6' }, boxpoints: false },
      { type: 'box', x: lf, name: 'Your set', orientation: 'h', marker: { color: '#C0632B' }, boxpoints: 'all', jitter: 0.5, pointpos: 0,
        text: idx.map(i => de.symbol[i]), hovertemplate: '%{text}: log2FC %{x:.2f}<extra></extra>' }],
      { showlegend: false, shapes: [vline(0, '#0F1D26', 'solid', 'paper')], xaxis: { title: { text: 'log2 fold change (tumour vs own normal)' } }, height: 300, margin: { l: 80 } });
    readout(root, `<div class="chips">${chip('Genes found', `${idx.length} of ${want.length}`)}${chip('Median log2FC of the set', median(lf).toFixed(2), median(lf) > 0 ? 'up' : 'down')}
      ${chip('Median log2FC, all genes', median(de.lfc).toFixed(2))}${chip('Mann–Whitney p (set vs rest)', fmtP(p))}</div>
      ${missing.length ? `<p class="muted">Not tested: ${missing.slice(0, 15).join(', ')}${missing.length > 15 ? ' …' : ''}</p>` : ''}`);
  }
  update();
};

// ---------- NB3: shared tumour data ---------------------------------------------------------------------
async function nb3() {
  const [meta, mr, mi, pr] = await Promise.all([J('nb3_meta.json'), B('nb3_mrna_i16.bin', Int16Array), B('nb3_mirna_i16.bin', Int16Array), B('nb3_protein_i16.bin', Int16Array)]);
  const T = meta.tumours.length, sc = meta.scale;
  const row = (arr, i) => { const o = new Float64Array(T); for (let k = 0; k < T; k++) o[k] = arr[i * T + k] / sc; return o; };
  const gIdx = new Map(meta.genes.map((g, i) => [g.toUpperCase(), i])), mIdx = new Map(meta.mirnas.map((g, i) => [g.toUpperCase(), i])),
        pIdx = new Map(meta.proteins.map((g, i) => [g.toUpperCase(), i]));
  return { meta, T, gene: g => gIdx.has(g.toUpperCase()) ? row(mr, gIdx.get(g.toUpperCase())) : null,
    mir: m => mIdx.has(m.toUpperCase()) ? row(mi, mIdx.get(m.toUpperCase())) : null,
    prot: p => pIdx.has(p.toUpperCase()) ? row(pr, pIdx.get(p.toUpperCase())) : null, mr, mi, sc };
}
const SUBCHECK = { key: 'subs', type: 'checks', label: 'Tumours included', options: [...SUB.map(s => [s, SUBNAME[s], COL[s]]), ['none', 'no label', COL.none], ['Normal', 'Normal-like', '#9b59b6']] };
function scatterBySubtype(meta, x, y, keep) {
  return [...SUB, 'Normal', 'none'].filter(s => keep.has(s)).map(s => { const idx = meta.subtype.flatMap((v, i) => v === s ? [i] : []);
    return { type: 'scattergl', mode: 'markers', name: SUBNAME[s], x: idx.map(i => x[i]), y: idx.map(i => y[i]), text: idx.map(i => meta.tumours[i]),
      hovertemplate: '%{text}<br>x %{x:.2f} · y %{y:.2f}<extra>' + SUBNAME[s] + '</extra>', marker: { size: 5, color: s === 'Normal' ? '#9b59b6' : COL[s], opacity: 0.65 } }; });
}
function fitLine(x, y) { const mx = mean(x), my = mean(y); let b = 0, s = 0; for (let i = 0; i < x.length; i++) { b += (x[i] - mx) * (y[i] - my); s += (x[i] - mx) ** 2; }
  b /= s; const lo = Math.min(...x), hi = Math.max(...x); return { type: 'scatter', mode: 'lines', x: [lo, hi], y: [my + b * (lo - mx), my + b * (hi - mx)], line: { color: '#0F1D26', width: 2 }, name: 'trend', hoverinfo: 'skip' }; }

// ---------- NB3: mRNA vs protein ----------------------------------------------------------------------------
W.mp = async root => {
  const D = await nb3(), meta = D.meta, mp = meta.mp;
  const [p1, p2] = plotsIn(root, 2);
  const st = controls(root, [
    { key: 'pair', type: 'select', label: 'Gene ↔ protein pair (sorted by r, all 204)', value: '0', options: mp.protein.map((p, i) => [String(i), `${mp.gene[i]} ↔ ${p}  (r ${mp.r[i].toFixed(2)})`]) },
    SUBCHECK,
  ], update);
  function update() {
    const i = +st.pair, x = D.gene(mp.gene[i]), y = D.prot(mp.protein[i]);
    const keep = meta.subtype.map(s => st.subs.has(s)), xs = x.filter((_, k) => keep[k]), ys = y.filter((_, k) => keep[k]);
    const rAll = pearson(x, y), rSub = xs.length > 2 ? pearson(xs, ys) : NaN;
    plot(p1, [...scatterBySubtype(meta, x, y, st.subs), ...(xs.length > 2 ? [fitLine(xs, ys)] : [])],
      { xaxis: { title: { text: `${mp.gene[i]} mRNA (log2 CPM)` } }, yaxis: { title: { text: `${mp.protein[i]} protein (RPPA)` } }, height: 430 });
    plot(p2, [{ type: 'histogram', x: mp.r, xbins: { start: -0.4, end: 1, size: 0.05 }, marker: { color: '#95a5a6' }, name: 'pairs' }],
      { showlegend: false, shapes: [vline(rAll, '#C0632B', 'solid'), vline(median(mp.r), '#0F1D26', 'dash')], height: 430,
        annotations: [{ x: rAll, yref: 'paper', y: 1, text: 'this pair', showarrow: false, yshift: 8, font: { color: '#C0632B' } }, { x: median(mp.r), yref: 'paper', y: 0.92, text: 'median', showarrow: false, xshift: -24 }],
        xaxis: { title: { text: 'r between mRNA and protein (204 pairs)' } }, yaxis: { title: { text: 'Gene–protein pairs' } } });
    readout(root, `<div class="chips">${chip('r, all 836 tumours', rAll.toFixed(2), rAll > 0.5 ? 'good' : '')}${chip('r, tumours shown', isNaN(rSub) ? '–' : rSub.toFixed(2))}${chip('Tumours shown', xs.length)}
      ${chip('Rank of this pair', `${i + 1} of ${mp.r.length}`)}${chip('Median r, all pairs', median(mp.r).toFixed(2))}</div>`);
  }
  update();
};

// ---------- NB3: miRNA vs target -------------------------------------------------------------------------------
W.mirtarget = async root => {
  const D = await nb3(), meta = D.meta, G = meta.genes.length, T = D.T;
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'mir', type: 'search', label: 'miRNA', value: 'mir-200c', list: () => meta.mirnas },
    { key: 'gene', type: 'search', label: `Gene (${fmt(G)} available)`, value: 'ZEB1', list: () => meta.genes },
    { key: 'pre', type: 'buttons', label: 'Examples from Notebook 3', options: [['mir-200c|ZEB1', 'miR-200c ↔ ZEB1'], ['mir-141|ZEB1', 'miR-141 ↔ ZEB1'], ['mir-18a|ESR1', 'miR-18a ↔ ESR1'], ['mir-155|SOCS1', 'miR-155 ↔ SOCS1'], ['mir-155|PTPRC', 'miR-155 ↔ PTPRC (immune)']],
      onClick: v => { const [a, b] = v.split('|'); st._set_mir(a); st._set_gene(b); update(); } },
    SUBCHECK,
  ], update);
  let lastMir = null, corrHtml = '';
  function update() {
    const x = D.mir(st.mir), y = D.gene(st.gene);
    if (!x || !y) { readout(root, `<p class="found">Pick a miRNA and a gene from the suggestions.</p>`); return; }
    const keep = meta.subtype.map(s => st.subs.has(s)), xs = x.filter((_, k) => keep[k]), ys = y.filter((_, k) => keep[k]);
    const r = pearson(x, y), rs = xs.length > 2 ? pearson(xs, ys) : NaN;
    plot(p1, [...scatterBySubtype(meta, x, y, st.subs), ...(xs.length > 2 ? [fitLine(xs, ys)] : [])],
      { xaxis: { title: { text: `${st.mir} (log2 RPM)` } }, yaxis: { title: { text: `${st.gene} mRNA (log2 CPM)` } }, height: 430 });
    if (st.mir !== lastMir) { lastMir = st.mir;
      const rs2 = []; for (let g = 0; g < G; g++) { const yy = new Float64Array(T); for (let k = 0; k < T; k++) yy[k] = D.mr[g * T + k] / D.sc; rs2.push([pearson(x, yy), meta.genes[g]]); }
      rs2.sort((a, b) => a[0] - b[0]);
      const li = a => a.map(([v, g]) => `<button type="button" class="gchip" data-g="${g}">${g} <small>${v.toFixed(2)}</small></button>`).join('');
      corrHtml = `<p><b>Most negative</b> (candidate targets, or simply other cell types): ${li(rs2.slice(0, 8))}</p><p><b>Most positive</b>: ${li(rs2.slice(-8).reverse())}</p>`; }
    readout(root, `<div class="chips">${chip('r, all tumours', r.toFixed(2), r < 0 ? 'down' : 'up')}${chip('r, tumours shown', isNaN(rs) ? '–' : rs.toFixed(2))}${chip('Tumours shown', xs.length)}</div>
      <div class="corrlist">Genes correlated with ${st.mir} across the 836 tumours (click to plot):${corrHtml}</div>`);
    root.querySelectorAll('.gchip').forEach(b => b.onclick = () => { st._set_gene(b.dataset.g); update(); });
  }
  update();
};

// ---------- NB3: PCA per layer -------------------------------------------------------------------------------
W.pca3 = async root => {
  const [d, meta] = await Promise.all([J('nb3_pca.json'), J('nb3_meta.json')]);
  const [p1] = plotsIn(root, 1);
  const pcOpt = [1, 2, 3, 4].map(i => [String(i - 1), 'PC' + i]);
  const st = controls(root, [
    { key: 'layer', type: 'select', label: 'Layer', value: 'Integrated', options: [['mRNA', 'mRNA only'], ['miRNA', 'miRNA only'], ['Protein', 'Protein only'], ['Integrated', 'All three integrated']] },
    { key: 'ntop', type: 'select', label: 'mRNA genes used', value: '2000', options: ['500', '2000', '5000'].map(v => [v, fmt(+v) + ' most variable']) },
    { key: 'w', type: 'select', label: 'Block weighting (integrated view)', value: '1', options: [['1', 'on: each block ÷ √features'], ['0', 'off: every feature counts once']] },
    { key: 'x', type: 'select', label: 'x-axis', value: '0', options: pcOpt }, { key: 'y', type: 'select', label: 'y-axis', value: '1', options: pcOpt },
  ], update);
  function update() {
    const L = st.layer, key = (L === 'miRNA' || L === 'Protein') ? `${L}|2000|1` : `${L}|${st.ntop}|${L === 'mRNA' ? 1 : st.w}`;
    const v = d.views[key], x = v.pc[+st.x], y = v.pc[+st.y];
    plot(p1, scatterBySubtype(meta, x, y, new Set([...SUB, 'none'])), { height: 470,
      xaxis: { title: { text: `PC${+st.x + 1} (${v.var[+st.x]} %)` } }, yaxis: { title: { text: `PC${+st.y + 1} (${v.var[+st.y]} %)` } } });
    const note = L === 'miRNA' || L === 'Protein' ? 'The gene number and weighting do not change this layer.' : L === 'mRNA' ? 'Weighting only matters when blocks are combined.' : '';
    readout(root, `<div class="chips">${chip('Neighbour agreement', v.knn.toFixed(2), 'good')}${chip('…expected by chance', d.chance.toFixed(2))}
      ${v.var.slice(0, 4).map((p, i) => chip('PC' + (i + 1), p + ' %')).join('')}</div>
      <p class="muted">Neighbour agreement = share of each tumour's 10 nearest tumours (in 10-PC space) with the same PAM50 subtype. ${note}</p>`);
  }
  update();
};

// ---------- NB3: marker heatmap ---------------------------------------------------------------------------------
W.heat = async root => {
  const d = await J('nb3_heat.json');
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'layer', type: 'select', label: 'Layer', value: 'mRNA', options: [['mRNA', 'mRNA'], ['miRNA', 'miRNA'], ['Protein', 'Protein']] },
    { key: 'k', type: 'range', label: 'Number of features (strongest subtype differences first)', min: 3, max: 30, step: 1, value: 12 },
  ], update);
  const bounds = []; let prev = null; d.subtype.forEach((s, i) => { if (s !== prev) { bounds.push([s, i]); prev = s; } }); bounds.push([null, d.subtype.length]);
  function update() {
    const L = d.layers[st.layer], k = st.k, yl = L.features.slice(0, k);
    plot(p1, [{ type: 'heatmap', z: L.z.slice(0, k), y: yl, x: d.subtype.map((_, i) => i), colorscale: [[0, '#2166ac'], [0.25, '#92c5de'], [0.5, '#f7f7f7'], [0.75, '#f4a582'], [1, '#b2182b']], zmin: -2.5, zmax: 2.5,
      colorbar: { title: { text: 'z' }, thickness: 12 }, hovertemplate: '%{y}<br>z = %{z:.2f}<extra></extra>' }],
      { height: Math.max(330, 60 + k * 19), margin: { l: 110, t: 40 }, xaxis: { showticklabels: false, showgrid: false }, yaxis: { autorange: 'reversed', showgrid: false, tickfont: { size: 11 } },
        shapes: bounds.slice(1, -1).map(([, i]) => ({ type: 'line', x0: i - 0.5, x1: i - 0.5, yref: 'paper', y0: 0, y1: 1, line: { color: '#000', width: 2 } })),
        annotations: bounds.slice(0, -1).map(([s, i], j) => ({ x: (i + bounds[j + 1][1]) / 2, yref: 'paper', y: 1, yanchor: 'bottom', text: `<b>${SUBNAME[s]}</b>`, showarrow: false, font: { color: COL[s] } })) });
    readout(root, `<p>${yl.map((f, i) => `<code>${f}</code> <small>F ${fmt(L.F[i])}</small>`).join(' · ')}</p>`);
  }
  update();
};

// ---------- NB4: model settings ---------------------------------------------------------------------------------
W.ml = async root => {
  const d = await J('nb4_ml_grid.json');
  const [p1, p2, p3] = plotsIn(root, 3);
  let st;
  const kOpts = L => d.grid[L].map(g => g.k).filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
  const kLab = (L, k) => k === d.n_features[L] ? `all ${k}` : String(k);
  st = controls(root, [
    { key: 'layer', type: 'select', label: 'Features from', value: 'mRNA', options: Object.keys(d.grid).map(k => [k, k === 'mRNA' ? 'mRNA (2,000 most variable genes)' : k]) },
    { key: 'k', type: 'steps', label: 'Features kept by SelectKBest (k)', values: kOpts('mRNA'), value: 200, fmt: v => kLab(st ? st.layer : 'mRNA', v) },
    { key: 'C', type: 'steps', label: 'Regularisation C (small = strong shrinkage)', values: d.cs, value: 0.05 },
  ], key => { if (key === 'layer') rebuildK(); update(); });
  function rebuildK() { const ks = kOpts(st.layer), w = root.querySelector('.ctl-steps input'); w.max = ks.length - 1;
    const cur = ks.includes(200) ? 200 : ks[ks.length - 1]; w.value = ks.indexOf(cur); w.dispatchEvent(new Event('input')); }
  function update() {
    const L = st.layer, ks = kOpts(L); const k = ks[Math.min(+root.querySelector('.ctl-steps input').value, ks.length - 1)]; st.k = k;
    root.querySelector('.ctl-steps output').textContent = kLab(L, k);
    const cell = d.grid[L].find(g => g.k === k && g.C === st.C);
    const z = ks.map(kk => d.cs.map(c => (d.grid[L].find(g => g.k === kk && g.C === c) || {}).test ?? null));
    const ci = d.cs.indexOf(st.C), ki = ks.indexOf(k);
    plot(p1, [{ type: 'heatmap', z, x: d.cs.map(String), y: ks.map(kk => kLab(L, kk)), colorscale: 'YlGnBu', reversescale: true, zmin: 0.25, zmax: 1, text: z.map(r => r.map(v => v === null ? '' : v.toFixed(2))),
      texttemplate: '%{text}', textfont: { size: 10 }, colorbar: { title: { text: 'test<br>bal. acc' }, thickness: 12 }, hovertemplate: 'k %{y}, C %{x}: %{z:.3f}<extra></extra>' }],
      { title: { text: 'Test balanced accuracy for every setting', font: { size: 13 } }, margin: { t: 40 }, xaxis: { title: { text: 'C' }, type: 'category' }, yaxis: { title: { text: 'k' }, type: 'category' },
        shapes: [{ type: 'rect', x0: ci - 0.5, x1: ci + 0.5, y0: ki - 0.5, y1: ki + 0.5, line: { color: '#C0632B', width: 3 } }] });
    plot(p2, [{ type: 'bar', x: ['Training set', '5-fold CV (train)', 'Test set'], y: [cell.train, cell.cv, cell.test], marker: { color: ['#95a5a6', '#1F7A6E', '#0F1D26'] },
      error_y: { type: 'data', array: [0, cell.cv_sd, 0], visible: true }, text: [cell.train, cell.cv, cell.test].map(v => v.toFixed(2)), textposition: 'outside' }],
      { title: { text: 'Balanced accuracy', font: { size: 13 } }, margin: { t: 40 }, showlegend: false, yaxis: { range: [0, 1.08] }, shapes: [hline(0.25, '#C0632B', 'dot')] });
    const cm = cell.cm;
    plot(p3, [{ type: 'heatmap', z: cm, x: d.classes, y: d.classes, colorscale: 'Blues', reversescale: true, showscale: false, text: cm.map(r => r.map(String)), texttemplate: '%{text}', hovertemplate: 'true %{y} → predicted %{x}: %{z}<extra></extra>' }],
      { title: { text: 'Confusion matrix (test set)', font: { size: 13 } }, margin: { t: 40 }, xaxis: { title: { text: 'Predicted' } }, yaxis: { title: { text: 'True' }, autorange: 'reversed' } });
    const gap = cell.train - cell.test;
    readout(root, `<div class="chips">${chip('Test balanced accuracy', cell.test.toFixed(3), 'good')}${chip('CV mean ± SD', `${cell.cv.toFixed(3)} ± ${cell.cv_sd.toFixed(3)}`)}
      ${chip('Training', cell.train.toFixed(3))}${chip('Train − test gap', gap.toFixed(2), gap > 0.15 ? 'bad' : '')}${chip('Errors on test set', sum(cm.flat()) - cm.reduce((s, r, i) => s + r[i], 0) + ' of ' + d.n_test)}</div>
      <p class="muted">${gap > 0.15 ? 'Large gap: the model fits the training tumours much better than new ones (over-fitting). ' : ''}Chance = 0.25 (four subtypes). Notebook 4 setting: k = 200, C = 0.05.</p>`);
  }
  update();
};

// ---------- NB4: ER status, ROC and threshold ---------------------------------------------------------------------
W.er = async root => {
  const d = await J('nb4_er.json'); const truth = d.truth, n = truth.length;
  const rocs = Object.fromEntries(Object.entries(d.prob).map(([k, p]) => [k, rocCurve(truth, p)]));
  const [p1, p2] = plotsIn(root, 2);
  const st = controls(root, [
    { key: 'layer', type: 'select', label: 'Model built on', value: 'All three', options: Object.keys(d.prob).map(k => [k, k]) },
    { key: 't', type: 'range', label: 'Call ER-positive if probability ≥', min: 0.01, max: 0.99, step: 0.01, value: 0.5 },
  ], update);
  function update() {
    const p = d.prob[st.layer]; let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < n; i++) { const c = p[i] >= st.t; if (c && truth[i]) tp++; else if (c) fp++; else if (truth[i]) fn++; else tn++; }
    const tpr = tp / (tp + fn), fpr = fp / (fp + tn);
    plot(p1, [...Object.entries(rocs).map(([k, r]) => ({ type: 'scatter', mode: 'lines', x: r.fpr, y: r.tpr, name: `${k} (AUC ${r.auc.toFixed(3)})`,
      line: { width: k === st.layer ? 3 : 1.2 }, opacity: k === st.layer ? 1 : 0.45 })),
      { type: 'scatter', mode: 'markers', x: [fpr], y: [tpr], name: 'current threshold', marker: { size: 13, color: '#C0632B', line: { color: '#000', width: 1 } } },
      { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dot', color: '#999' }, showlegend: false, hoverinfo: 'skip' }],
      { xaxis: { title: { text: 'False positive rate (ER-negative called positive)' }, range: [-0.02, 1.02] }, yaxis: { title: { text: 'True positive rate (ER-positive found)' }, range: [-0.02, 1.02] },
        legend: { x: 0.35, y: 0.05, orientation: 'v', bgcolor: 'rgba(255,255,255,.85)' }, height: 440 });
    plot(p2, [0, 1].map(c => ({ type: 'histogram', name: c ? 'ER-positive by IHC' : 'ER-negative by IHC', x: p.filter((_, i) => truth[i] === c), xbins: { start: 0, end: 1, size: 0.05 },
      marker: { color: c ? '#1f77b4' : '#2ca02c' }, opacity: 0.7 })), { barmode: 'overlay', shapes: [vline(st.t, '#C0632B', 'solid')], height: 440,
      xaxis: { title: { text: 'Model probability of ER-positive' } }, yaxis: { title: { text: 'Test tumours' }, type: 'log' } });
    const conf = []; for (let i = 0; i < n; i++) if ((p[i] > 0.9 && !truth[i]) || (p[i] < 0.1 && truth[i])) conf.push(i);
    readout(root, `<div class="chips">${chip('AUC', rocs[st.layer].auc.toFixed(3), 'good')}${chip('Sensitivity (ER+ found)', (tpr * 100).toFixed(1) + ' %')}${chip('Specificity (ER− cleared)', ((1 - fpr) * 100).toFixed(1) + ' %')}
      ${chip('True +', tp)}${chip('False +', fp, fp ? 'warn' : '')}${chip('True −', tn)}${chip('False −', fn, fn ? 'warn' : '')}</div>
      <p class="muted">${n} test tumours. Confident disagreements with the pathologist (P &gt; 0.9 but IHC negative, or P &lt; 0.1 but IHC positive): <b>${conf.length}</b>
      ${conf.length ? '– ' + conf.map(i => `<code>${d.ids[i]}</code> (IHC ${truth[i] ? '+' : '−'}, P ${p[i].toFixed(2)}, PAM50: ${SUBNAME[d.pam50[i]] || d.pam50[i]})`).join(', ') : ''}</p>`);
  }
  update();
};

// ---------- NB4: leakage ------------------------------------------------------------------------------------------------
W.leak = async root => {
  const d = await J('nb4_leak.json');
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [
    { key: 'n', type: 'steps', label: 'Tumours in the study', values: d.ns, value: 100 },
    { key: 'k', type: 'steps', label: 'Genes selected (of all ' + fmt(d.n_genes) + ')', values: d.ks, value: 20 },
  ], update);
  function update() {
    const run = d.runs[`${st.n}|${st.k}`], R = rng(5);
    const tr = [['wrong', 'WRONG: select genes on all samples, then cross-validate', '#C0632B'], ['right', 'RIGHT: selection inside each training fold', '#1F7A6E']].map(([key, name, color], j) => ({
      type: 'scatter', mode: 'markers', name, x: run[key].map(() => j + (R() - 0.5) * 0.3), y: run[key], marker: { size: 10, color, opacity: 0.75 },
      text: d.seeds.map(s => `repeat ${s}`), hovertemplate: '%{text}: %{y:.2f}<extra></extra>' }));
    tr.push({ type: 'scatter', mode: 'markers', name: 'mean of 10 repeats', x: [0, 1], y: [mean(run.wrong), mean(run.right)], marker: { symbol: 'line-ew-open', size: 40, color: '#000', line: { width: 3 } } });
    plot(p1, tr, { height: 430, shapes: [hline(0.25, '#555', 'dot')], annotations: [{ xref: 'paper', x: 1, y: 0.25, text: 'chance', showarrow: false, xanchor: 'right', yshift: 9 }],
      xaxis: { tickvals: [0, 1], ticktext: ['Wrong procedure', 'Right procedure'], range: [-0.5, 1.5] }, yaxis: { title: { text: 'Cross-validated balanced accuracy (labels are random!)' }, range: [0, 1] } });
    readout(root, `<div class="chips">${chip('Wrong way, mean of 10', mean(run.wrong).toFixed(2), 'bad')}${chip('Right way, mean of 10', mean(run.right).toFixed(2), 'good')}${chip('True answer', '0.25 (chance)')}
      ${st.n === 100 && st.k === 20 ? chip('Notebook run (repeat 1)', `${run.wrong[0].toFixed(2)} vs ${run.right[0].toFixed(2)}`) : ''}</div>`);
  }
  update();
};

// ---------- NB4: unlabelled tumours -----------------------------------------------------------------------------------
W.unlab = async root => {
  const d = await J('nb4_unlabelled.json'); const n = d.ids.length;
  const pred = d.proba.map(p => p.indexOf(Math.max(...p))), conf = d.proba.map(p => Math.max(...p));
  const [p1] = plotsIn(root, 1);
  const st = controls(root, [{ key: 't', type: 'range', label: 'Trust a prediction if its probability is above', min: 0.3, max: 0.99, step: 0.01, value: 0.8 }], update);
  function update() {
    const cls = d.classes, sure = cls.map(() => 0), unsure = cls.map(() => 0);
    pred.forEach((c, i) => (conf[i] > st.t ? sure : unsure)[c]++);
    plot(p1, [{ type: 'bar', name: 'confident', x: cls.map(c => SUBNAME[c]), y: sure, marker: { color: cls.map(c => COL[c]) } },
      { type: 'bar', name: 'uncertain', x: cls.map(c => SUBNAME[c]), y: unsure, marker: { color: '#d5d8da' } }], { barmode: 'stack', height: 360, yaxis: { title: { text: 'Tumours without a PAM50 label' } } });
    const b = cls.indexOf('Basal'); let bAll = 0, bNeg = 0, bS = 0, bSNeg = 0;
    pred.forEach((c, i) => { if (c === b) { bAll++; const neg = d.er_ihc[i] === 'Negative'; if (neg) bNeg++; if (conf[i] > st.t) { bS++; if (neg) bSNeg++; } } });
    readout(root, `<div class="chips">${chip('Confident predictions', `${sum(sure)} of ${n}`, 'good')}${chip('Uncertain', sum(unsure))}
      ${chip('Predicted Basal & ER-negative by IHC', `${bNeg} of ${bAll}`)}${chip('…among confident Basal calls', `${bSNeg} of ${bS}`)}</div>`);
  }
  update();
};

// ------------------------------------------------------------------ boot: build each widget when it comes near the screen
const started = new WeakSet();
const io = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting || started.has(e.target)) return; started.add(e.target);
  const name = e.target.dataset.w; e.target.classList.add('loading');
  W[name](e.target).then(() => e.target.classList.remove('loading')).catch(err => { e.target.classList.remove('loading');
    $('.readout', e.target).innerHTML = `<p class="found">Could not load this figure (${err.message}). Open the page through its web link, not as a local file.</p>`; console.error(err); });
}), { rootMargin: '800px 0px' });
document.querySelectorAll('[data-w]').forEach(el => io.observe(el));

// sidebar highlight
const links = [...document.querySelectorAll('nav#side a[href^="#"]')];
const obs = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) {
  links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  const act = document.querySelector('nav#side a.active'), nav = document.getElementById('side');
  if (act) { const r = act.offsetTop; if (r < nav.scrollTop + 60 || r > nav.scrollTop + nav.clientHeight - 60) nav.scrollTop = r - nav.clientHeight / 3; } } }),
  { rootMargin: '0px 0px -75% 0px' });
links.map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean).forEach(t => obs.observe(t));
