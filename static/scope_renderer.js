    // ------------------------------------------------- client-side rendering
    // A port of scope_bake.render_luma. Same algorithm: box-resample the
    // luminance to a grid, tone-map it, walk a serpentine path spending
    // samples in proportion to brightness. Dwell IS the brightness, which is
    // why the walk is weighted rather than uniform.
    //
    // It runs HERE rather than on the server because the server has no sound
    // card, and because the browser's AudioContext rate decides the sample
    // budget -- rendering server-side would mean resampling, and resampling a
    // trace is a filter applied to a picture.
    const lumImg = document.getElementById("luma-src");
    let actx = null, worklet = null, lumCanvas = null, lumCtx = null;
    let localRunning = false, tracesBuilt = 0;

    function boxGrid(data, w, h, rows, cols) {
      // area-average each cell, matching _box()
      const g = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const y0 = Math.floor(r * h / rows), y1 = Math.max(y0 + 1, Math.floor((r + 1) * h / rows));
        for (let c = 0; c < cols; c++) {
          const x0 = Math.floor(c * w / cols), x1 = Math.max(x0 + 1, Math.floor((c + 1) * w / cols));
          let sum = 0, n = 0;
          for (let y = y0; y < y1; y++) {
            const off = y * w;
            for (let x = x0; x < x1; x++) { sum += data[(off + x) * 4]; n++; }
          }
          g[r * cols + c] = n ? sum / (n * 255) : 0;
        }
      }
      return g;
    }

    function buildTrace(g, rows, cols, n, trim, gamma, w, h, level) {
      // serpentine path + per-cell weight, then walk it by cumulative weight
      const px = [], py = [], wt = [];
      // Aspect, matching render_luma: the SHORT side is scaled down so the
      // picture keeps its proportions. Without this a 128x171 source came out
      // nearly square -- x reached +-0.874 where it should reach +-0.674 --
      // so the face was stretched sideways.
      const L = (level === undefined ? 0.9 : level);
      const sx = L * (w >= h ? 1.0 : w / h);
      const sy = L * (h >= w ? 1.0 : h / w);
      // Contrast stretch, matching render_luma: map the 2nd..98th percentile
      // of the LIT cells onto 0..1. Without it the picture is flat and much
      // too bright -- measured 72.8% of pixels lit against python's 55.1%.
      const lit = [];
      for (let i = 0; i < g.length; i++) if (g[i] > 0.01) lit.push(g[i]);
      if (lit.length > 16) {
        lit.sort((p, q) => p - q);
        const lo = lit[Math.floor(lit.length * 0.02)];
        const hi = lit[Math.floor(lit.length * 0.98)];
        if (hi > lo) {
          for (let i = 0; i < g.length; i++)
            g[i] = Math.min(1, Math.max(0, (g[i] - lo) / (hi - lo)));
        }
      }

      let prevRight = false;
      for (let r = 0; r < rows; r++) {
        let a = -1, b = -1;
        for (let c = 0; c < cols; c++) if (g[r * cols + c] > trim) { if (a < 0) a = c; b = c; }
        if (a < 0) continue;
        const y = 1 - 2 * (r + 0.5) / rows;              // y up on the scope
        const fwd = !prevRight;
        for (let k = 0; k <= (b - a); k++) {
          const c = fwd ? a + k : b - k;
          px.push((2 * (c + 0.5) / cols - 1) * sx);
          py.push(y * sy);
          wt.push(Math.pow(Math.max(g[r * cols + c], 1e-9), gamma));
        }
        prevRight = fwd;
      }
      const m = px.length;
      const out = new Float32Array(n * 2);
      if (m < 2) return out;
      // cumulative weight -> sample positions. More weight = more samples =
      // the beam lingers = brighter. This is the whole display mechanism.
      const cum = new Float64Array(m);
      let tot = 0;
      for (let i = 0; i < m; i++) { tot += wt[i]; cum[i] = tot; }
      if (tot <= 0) return out;
      let j = 0;
      for (let i = 0; i < n; i++) {
        const target = (i + 0.5) / n * tot;
        while (j < m - 1 && cum[j] < target) j++;
        // Interpolate WITHIN the segment. Snapping to the nearest vertex puts
        // every sample exactly on a cell centre, which is what made the output
        // read as a dot grid rather than continuous scanlines.
        const c0 = j > 0 ? cum[j - 1] : 0;
        const span = cum[j] - c0;
        const t = span > 1e-12 ? Math.min(1, Math.max(0, (target - c0) / span)) : 0;
        const jp = j > 0 ? j - 1 : 0;
        out[i * 2] = px[jp] + (px[j] - px[jp]) * t;
        out[i * 2 + 1] = py[jp] + (py[j] - py[jp]) * t;
      }
      return out;
    }

    function buildYtTrace(g, rows, cols, n, trim, gamma, w, h, triggerSamples, border) {
      // Same fixed timeline as scope_bake.render_yt_grid; g is tone-mapped.
      const marker = Math.max(0, triggerSamples | 0);
      let borderN = Math.floor(n * Math.min(0.5, Math.max(0, border || 0)) + 0.5);
      if (borderN < 5) borderN = 0;
      const retrace = Math.max(2, Math.floor(n * 0.004 + 0.5));
      const returnStart = n - retrace, pictureEnd = returnStart - borderN;
      const pictureN = pictureEnd - marker;
      if (pictureN < 2 * rows) throw new Error("Y-T: not enough samples per row; reduce rows/trigger/border");
      const sx = 0.9 * Math.min(1, w / h), sy = 0.9 * Math.min(1, h / w);
      const out = new Float32Array(n * 2), pedestal = -0.936;
      for (let i = 0; i < n; i++) { out[2*i] = pedestal; out[2*i+1] = sy; }
      for (let r = 0; r < rows; r++) {
        const begin = marker + Math.floor(r * pictureN / rows);
        const end = marker + Math.floor((r + 1) * pictureN / rows);
        const y = rows > 1 ? sy - 2 * sy * r / (rows - 1) : sy;
        let a = -1, b = -1;
        for (let c = 0; c < cols; c++) if (g[r*cols+c] > Math.max(0,trim)) {
          if (a < 0) a = c; b = c;
        }
        for (let i = begin; i < end; i++) out[2*i+1] = y;
        if (a < 0) continue;
        const xx = [], vv = [];
        for (let k = 0; k <= b-a; k++) {
          const c = r % 2 ? b-k : a+k;
          xx.push(cols > 1 ? -sx + 2*sx*c/(cols-1) : -sx);
          vv.push(g[r*cols+c]);
        }
        if (trim > 0 && xx.length > 1) {
          const step = 2*sx/(cols-1), left = r%2 ? xx.length-1 : 0, right = r%2 ? 0 : xx.length-1;
          if (a>0 && g[r*cols+a]>g[r*cols+a-1]) xx[left] -= step*Math.max(0,Math.min(1,(g[r*cols+a]-trim)/(g[r*cols+a]-g[r*cols+a-1])));
          if (b<cols-1 && g[r*cols+b]>g[r*cols+b+1]) xx[right] += step*Math.max(0,Math.min(1,(g[r*cols+b]-trim)/(g[r*cols+b]-g[r*cols+b+1])));
        }
        const cum = [0];
        for (let j = 1; j < vv.length; j++) cum.push(cum[j-1] + Math.pow(Math.max(vv[j],0.012),gamma));
        let j = 0;
        for (let i = begin; i < end-1; i++) {
          const target = (i-begin) * cum[cum.length-1] / (end-begin-1);
          while (j < xx.length-2 && cum[j+1] <= target) j++;
          out[2*i] = xx.length === 1 ? xx[0] : xx[j] + (xx[j+1]-xx[j]) * (target-cum[j])/(cum[j+1]-cum[j]);
        }
        out[2*begin] = xx[0]; out[2*(end-1)] = xx[xx.length-1];
      }
      if (borderN) {
        const corners = [[-sx,-sy],[sx,-sy],[sx,sy],[-sx,sy],[-sx,-sy]];
        const lengths = [2*sx,2*sy,2*sx,2*sy];
        const cum = [0]; for (const length of lengths) cum.push(cum[cum.length-1]+length);
        let j = 0;
        for (let i = 0; i < borderN; i++) {
          const target = i*cum[4]/borderN;
          while (j < 3 && cum[j+1] <= target) j++;
          const t = (target-cum[j])/lengths[j];
          for (let c=0;c<2;c++) out[2*(pictureEnd+i)+c] = corners[j][c]+t*(corners[j+1][c]-corners[j][c]);
        }
      }
      for (let i = 0; i < retrace; i++) {
        const t = i/(retrace-1);
        for (let c = 0; c < 2; c++) out[2*(returnStart+i)+c] = (1-t)*out[2*(returnStart-1)+c]+t*out[c];
      }
      return out;
    }

    function markerWindow(count, shape) {
      // Must stay bit-comparable with scope_out.marker_window.
      const n = Math.max(2, count | 0), level = 0.99;
      const out = new Float32Array(n * 2);
      if (shape === "step") {
        const split = Math.max(1, Math.floor(n / 2));
        for (let i = 0; i < n; i++) {
          out[2*i] = i < split ? -level : level;
          out[2*i+1] = level;
        }
        return out;
      }
      const hold = Math.max(1, Math.floor(n / 8));
      const span = Math.max(1, n - 2 * hold - 1);
      for (let i = 0; i < n; i++) {
        const t = Math.min(1, Math.max(0, (i - hold) / span));
        out[2*i] = -level + 2 * level * t;
        out[2*i+1] = level;
      }
      return out;
    }

    function prependMarker(xy, n, sampleRate, triggerUs, shape) {
      const marker = Math.max(4, Math.min(
        Math.max(4, n >> 3),
        Math.round(sampleRate * Number(triggerUs || 250) / 1000000)));
      const w = markerWindow(marker, shape);
      const out = new Float32Array((n + marker) * 2);
      out.set(w, 0);
      for (let i = 0; i < n; i++) {
        out[2*(marker+i)]   = Math.min(0.9, Math.max(-0.98, xy[2*i]));
        out[2*(marker+i)+1] = Math.min(1.0, Math.max(-1.0, xy[2*i+1]));
      }
      return out;
    }

    function addYtTrigger(xy, n, sampleRate, triggerUs, shape) {
      // Must stay bit-comparable with scope_out.trigger_frame; the parity
      // test in test_scope_web.py runs both and diffs them.
      const out = new Float32Array(xy);
      const markerUs = Number(triggerUs || 250);
      const marker = Math.max(4, Math.min(
        n, Math.round(sampleRate * markerUs / 1000000)));
      const level = 0.99;
      // Picture content must stay clear of the trigger threshold, or a
      // filtered highlight becomes a second edge and the scope will not hold.
      for (let i = 0; i < n; i++) {
        out[i * 2] = Math.min(0.9, Math.max(-0.98, out[i * 2]));
      }
      if (shape === "step") {
        const split = Math.max(2, Math.floor(marker / 2));
        for (let i = 0; i < n; i++) {
          if (i < split) out[i * 2] = -level;
          else if (i < marker) out[i * 2] = level;
        }
        return out;
      }
      // ramp: one monotonic crossing, and Y parked on the rail so the whole
      // marker sits outside the +-0.9 picture box an XY display is showing.
      const hold = Math.max(1, Math.floor(marker / 8));
      const span = Math.max(1, marker - 2 * hold - 1);
      for (let i = 0; i < marker && i < n; i++) {
        const t = Math.min(1, Math.max(0, (i - hold) / span));
        out[i * 2] = -level + 2 * level * t;
        out[i * 2 + 1] = level;
      }
      return out;
    }

    const WORKLET_SRC = `
      class ScopeOut extends AudioWorkletProcessor {
        constructor(){ super(); this.buf=null; this.pos=0;
          this.port.onmessage = e => { this.buf = e.data; this.pos = 0; }; }
        process(_i, outputs){
          const L=outputs[0][0], R=outputs[0][1];
          if(!this.buf){ L.fill(0); R.fill(0); return true; }
          const b=this.buf, n=b.length>>1;
          for(let k=0;k<L.length;k++){
            L[k]=b[this.pos*2]; R[k]=b[this.pos*2+1];
            this.pos++; if(this.pos>=n) this.pos=0;   // loop, never silence
          }
          return true;
        }
      }
      registerProcessor('scope-out', ScopeOut);
    `;

    function setStatus() {
      if (!localRunning || !actx) return;
      const ytFixed = lastData.scope_yt_timing === "fixed";
      // Use the picture budget, not the marker-inclusive refresh rate.
      const serverRate = Number(lastData.scope_samplerate);
      const pictureSamples = Number(lastData.scope_samples_per_trace);
      const rate = ytFixed && serverRate > 0 && pictureSamples > 0
        ? serverRate / pictureSamples : 30;
      const n = Math.max(64, Math.round(actx.sampleRate / rate));
      document.getElementById("local-status").textContent =
        `Running at ${actx.sampleRate} Hz \u2014 ${n} samples per trace, `
        + `${tracesBuilt} built. Your rate, your sound card.`;
    }

    // MJPEG in an <img> fires `load` ONCE. The browser keeps replacing the
    // displayed frame as new parts arrive, but there is no event per frame,
    // so anything hung off onload runs exactly once -- which rendered a
    // single trace and then stopped forever.
    //
    // Instead, sample the element on a clock. drawImage() always copies
    // whatever frame is currently showing, so a timer gets the live picture
    // without needing an event that does not exist.
    function renderFromLuma() {
      if (!localRunning || !actx || !worklet) return;
      const w = lumImg.naturalWidth, h = lumImg.naturalHeight;
      if (!w || !h) return;
      if (lumCanvas.width !== w) { lumCanvas.width = w; lumCanvas.height = h; }
      lumCtx.drawImage(lumImg, 0, 0);
      const px = lumCtx.getImageData(0, 0, w, h).data;
      const hh = frameHash(px, w, h);
      if (hh !== lastHash) {
        lastHash = hh;
        lastChange = Date.now();
        backoff = 1000;                 // healthy again, reset the backoff
      } else if (Date.now() - lastChange > STALL_MS) {
        reconnectLuma("feed stalled");
        return;
      }
      // sample budget comes from THIS browser's rate, not the server's
      const ytFixed = lastData.scope_yt_timing === "fixed";
      // Use the picture budget, not the marker-inclusive refresh rate.
      const serverRate = Number(lastData.scope_samplerate);
      const pictureSamples = Number(lastData.scope_samples_per_trace);
      const rate = ytFixed && serverRate > 0 && pictureSamples > 0
        ? serverRate / pictureSamples : 30;
      const n = Math.max(64, Math.round(actx.sampleRate / rate));
      const cells = Math.max(64, n);
      let cols = Math.max(8, Math.round(Math.sqrt(cells / (h / w))));
      let rows = Math.max(6, Math.round(cols * h / w));
      cols = Math.min(cols, w); rows = Math.min(rows, h);
      if (ytFixed && lastData.scope_yt_grid) {
        // Browser-local audio is progressive; it does not cycle server fields.
        rows = Math.min(h, lastData.scope_yt_grid[0]);
        cols = Math.min(w, lastData.scope_yt_grid[1]);
      }
      const trim = parseFloat(document.getElementById("local-trim").value);
      const gamma = parseFloat(document.getElementById("local-gamma").value);
      const g = boxGrid(px, w, h, rows, cols);
      let trace;
      if (ytFixed) {
        const levels = lastData.scope_yt_levels;
        if (levels && levels[1] > levels[0]) {
          for (let i=0;i<g.length;i++) g[i] = Math.max(0,Math.min(1,(g[i]-levels[0])/(levels[1]-levels[0])));
        }
        try {
          // triggerSamples 0: the marker is prepended by prependMarker below,
          // as scope_out.show_frame does, so the row timeline must not reserve
          // a second window for it.
          trace = buildYtTrace(g, rows, cols, n, trim, gamma, w, h, 0, lastData.scope_border);
        } catch (err) {
          document.getElementById("local-status").textContent = err.message;
          return;
        }
      } else {
        trace = buildTrace(g, rows, cols, n, trim, gamma, w, h);
      }
      if (lastData.scope_trigger !== false) {
        // PREPEND, matching scope_out.show_frame: the marker gets its own
        // samples so the picture arrives whole. Overwriting the head cost the
        // first ~24 samples of every trace and the exact beam handoff with it.
        trace = prependMarker(trace, n, actx.sampleRate,
                              lastData.scope_yt_trigger_us,
                              lastData.scope_trigger_shape);
      }
      worklet.port.postMessage(trace);
      if ((++tracesBuilt % 15) === 0) setStatus();
    }

    // IPS, not the trace rate: the picture only changes when the index does,
    // and rebuilding faster would burn CPU redrawing identical content. The
    // worklet loops whatever it was last given, so the beam keeps sweeping
    // between updates regardless.
    let lumTimer = null;
    function startLumaLoop() {
      if (lumTimer) clearInterval(lumTimer);
      lumTimer = setInterval(renderFromLuma, 1000 / 30);
    }
    function stopLumaLoop() {
      if (lumTimer) { clearInterval(lumTimer); lumTimer = null; }
    }
    // ---------------------------------------------------- keepalive
    // An MJPEG connection is not forever. nginx cuts it at
    // proxy_read_timeout, the server's own loop returns after 30 s with no
    // new frame, and a sleeping laptop or a network blip ends it too. None
    // of those raise an error the page can see: the <img> simply stops
    // updating, and the worklet keeps looping its last trace, so the audio
    // carries on sounding correct while the picture is frozen. That is the
    // worst possible failure -- it looks like it is working.
    //
    // So watch the CONTENT, not the connection. If the frame stops changing,
    // reconnect regardless of whether anything reported a problem.
    const STALL_MS = 6000;
    let lastHash = -1, lastChange = Date.now(), reconnects = 0, backoff = 1000;

    function frameHash(px, w, h) {
      // ~100 scattered pixels: enough to notice a changed picture, far
      // cheaper than hashing the frame, and it runs every render.
      let acc = 0;
      const step = Math.max(4, Math.floor((w * h) / 100)) * 4;
      for (let i = 0; i < px.length; i += step) acc = (acc * 31 + px[i]) | 0;
      return acc;
    }

    function reconnectLuma(why) {
      if (!localRunning) return;
      reconnects++;
      document.getElementById("local-status").textContent =
        `Reconnecting (${why})\u2026 attempt ${reconnects}`;
      lumImg.removeAttribute("src");
      setTimeout(() => {
        if (!localRunning) return;
        lumImg.src = "scope/luma.mjpg?t=" + Date.now();
        lastChange = Date.now();
        // back off so a server that is down is not hammered, but cap it:
        // this is an installation, it should recover on its own eventually
        backoff = Math.min(backoff * 2, 15000);
      }, backoff);
    }

    lumImg.onerror = () => reconnectLuma("feed error");
