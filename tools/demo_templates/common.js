// Shared drawing helpers (inlined into every demo page).
const BUTTONS = ["forward", "back", "strafe L", "strafe R", "turn L", "turn R", "attack", "use", "weapon"];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const $ = id => document.getElementById(id);

function fit(canvas, ratio) {
  const w = canvas.clientWidth || 600, h = Math.round(w * ratio), d = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * d) || canvas.height !== Math.round(h * d)) { canvas.width = Math.round(w * d); canvas.height = Math.round(h * d); canvas.style.height = h + "px"; }
  const g = canvas.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, w, h);
  return [g, w, h];
}
function projector(bounds, w, h, pad = 16) {
  const [x0, y0, x1, y1] = bounds, s = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
  const ox = (w - s * (x1 - x0)) / 2, oy = (h - s * (y1 - y0)) / 2;
  const P = (x, y) => [ox + (x - x0) * s, h - oy - (y - y0) * s];
  P.scale = s; return P;
}
function drawMap(g, P, geo, open = []) {
  const isOpen = new Set(open);
  for (const [x1, y1, x2, y2, kind, door] of geo.lines) {
    if (kind === "edge") continue;
    g.strokeStyle = door >= 0 && isOpen.has(door) ? css("--open") : css(kind === "wall" ? "--wall" : kind === "door" ? "--door" : "--locked");
    g.lineWidth = door >= 0 ? 3 : 1.3;
    g.beginPath(); g.moveTo(...P(x1, y1)); g.lineTo(...P(x2, y2)); g.stroke();
  }
  for (const [x, y] of geo.exits) star(g, ...P(x, y), 8, css("--good"));
}
function dot(g, x, y, r, fill, alpha = 1) { g.globalAlpha = alpha; g.fillStyle = fill; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); g.globalAlpha = 1; }
function star(g, x, y, r, fill) { g.fillStyle = fill; g.beginPath(); for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, q = i % 2 ? r / 2.3 : r; g.lineTo(x + q * Math.cos(a), y + q * Math.sin(a)); } g.fill(); }
function cross(g, x, y, r, colour, width = 3) { g.strokeStyle = colour; g.lineWidth = width; g.beginPath(); g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x - r, y + r); g.lineTo(x + r, y - r); g.stroke(); }
function player(g, P, x, y, degrees, colour, size = 9) {
  const [px, py] = P(x, y), a = -degrees * Math.PI / 180;
  g.fillStyle = colour; g.beginPath();
  g.moveTo(px + size * Math.cos(a), py + size * Math.sin(a));
  g.lineTo(px + size * .8 * Math.cos(a + 2.5), py + size * .8 * Math.sin(a + 2.5));
  g.lineTo(px + size * .8 * Math.cos(a - 2.5), py + size * .8 * Math.sin(a - 2.5));
  g.fill();
}
function path(g, P, points, upto, colour, width, dash = [], alpha = 1) {
  g.strokeStyle = colour; g.lineWidth = width; g.setLineDash(dash); g.globalAlpha = alpha; g.beginPath();
  for (let i = 0; i <= upto && i < points.length; i++) { const [x, y] = P(points[i][0], points[i][1]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
function buttonsHtml(bits) { return BUTTONS.map((n, i) => `<span class="${bits[i] === "1" ? "on" : ""}">${n}</span>`).join(""); }

// Play / pause / scrub / speed. `draw(t)` renders frame t.
function transport(length, draw, fps) {
  const bar = $("scrub"), play = $("play"), speed = $("speed");
  let t = 0, running = false, last = 0, carry = 0;
  bar.max = length - 1;
  const set = v => { t = Math.max(0, Math.min(length - 1, v)); bar.value = t; draw(t); };
  const tick = now => {
    if (!running) return;
    carry += (now - last) / 1000 * fps * Number(speed.value); last = now;
    const n = Math.floor(carry); carry -= n;
    if (n) set(t + n);
    if (t >= length - 1) { running = false; play.textContent = "Replay"; return; }
    requestAnimationFrame(tick);
  };
  play.onclick = () => { running = !running; play.textContent = running ? "Pause" : "Play"; if (running) { if (t >= length - 1) set(0); last = performance.now(); requestAnimationFrame(tick); } };
  bar.oninput = () => set(Number(bar.value));
  document.addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); play.click(); } if (e.code === "ArrowRight") set(t + 1); if (e.code === "ArrowLeft") set(t - 1); });
  addEventListener("resize", () => draw(t));
  // "page.html#t=90" opens at decision 91: handy for jumping to a moment during a live demo.
  const asked = typeof location === "undefined" ? null : location.hash.match(/t=(\d+)/);
  set(asked ? Number(asked[1]) : 0);
  return set;
}
