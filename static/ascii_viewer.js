/* One shared master frame; each tab chooses its own grid. Never sends resize. */
(function () {
    'use strict';
    const config = window.ASCII_VIEWER_CONFIG;
    const grid = window.AsciiResolution;
    const appearance = window.AsciiAppearance;
    const initialAppearance = appearance.normalize(config.appearance);
    let picture = {...initialAppearance};
    let colourTransform = appearance.transform(picture);
    // Image only: picture and zoom controls are never shown; zoom stays on gestures and keys.
    let hideControls = true;
    let backgroundBasis = appearance.toHsl(picture.background);
    const FONT_FAMILY = '"Interleave Schoolbook Mono", monospace';
    let sourceCols = config.sourceCols || 120, sourceRows = config.sourceRows || 80;
    let maxCols = grid.clamp(sourceCols, 15, 160), maxRows = grid.clamp(sourceRows, 10, 100);
    const defaults = [grid.clamp(config.cols, 15, maxCols), grid.clamp(config.rows, 10, maxRows)];
    const stage = document.getElementById('terminal-stage');
    const terminalElement = document.getElementById('terminal');
    const status = document.getElementById('resolution');
    const more = document.getElementById('zoom-out');
    const less = document.getElementById('zoom-in');
    let resolution = defaults[1];
    let cols = defaults[0], rows = defaults[1];
    let term = null, ws = null, reconnectTimer = null;
    let frame = null, revision = 0, painted = -1, writing = false, scheduled = false;
    let spareFrame = null;
    let pendingCompressed = null, inflating = null;
    let layoutDirty = true;
    let rendererLayoutVersion = 0;
    const aspect = defaults[0] * config.fontRatio / defaults[1];

    function fit() {
        if (!layoutDirty) return;
        // Fit the stage itself: on compact screens it stops above the open panel.
        const boxW = Math.min(stage.clientWidth || window.innerWidth, (stage.clientHeight || window.innerHeight) * aspect);
        const boxH = boxW / aspect;
        if (term && term.fitViewport) {
            try { term.fitViewport(boxW,boxH,picture); layoutDirty=false; }
            catch (error) { fallbackRenderer(error); }
            return;
        }
        const screen = terminalElement.querySelector('.xterm-screen');
        if (!screen || screen.clientWidth < 1 || screen.clientHeight < 1) return;
        const renderedRows = terminalElement.querySelectorAll('.xterm-rows > div');
        // This bundle uses xterm's DOM renderer. Move row origins rather than
        // lineHeight, whose minimum of 1 would prevent truly tighter spacing.
        // Always derive from original cell geometry: repeated fits cannot drift.
        const spacing = renderedRows.length === term.rows ? picture.lineSpacing : 1;
        const layout = appearance.rowLayout(term.rows, screen.clientHeight, spacing);
        renderedRows.forEach((row, index) => {
            row.style.transform = spacing === 1 ? '' : `translateY(${layout.offsets[index]}px)`;
        });
        terminalElement.style.width = screen.clientWidth + 'px';
        terminalElement.style.height = layout.height + 'px';
        terminalElement.style.overflow = 'hidden';
        // Fixed picture extent: enlarge cells, not a crop of the master picture.
        terminalElement.style.transform = `scale(${boxW / screen.clientWidth}, ${boxH / layout.height})`;
        layoutDirty = false;
    }
    function paint() {
        scheduled = false;
        if (document.hidden || !term || !frame || writing || painted === revision) return;
        writing = true;
        const version = revision;
        try {
            if (term.cols !== cols || term.rows !== rows) { term.resize(cols, rows); layoutDirty = true; }
            if (term.fitViewport) {
                if (term.dpr !== (window.devicePixelRatio || 1)) layoutDirty=true;
                const renderer=term;
                fit();
                if (renderer!==term) { writing=false; schedule(); return; }
            }
            if (term.renderGrid) {
                term.renderGrid(frame, cols, rows, colourTransform);
                if (rendererLayoutVersion !== term.layoutVersion) {
                    rendererLayoutVersion = term.layoutVersion; layoutDirty = true;
                }
                writing = false; painted = version; fit();
                return;
            }
            const output = grid.encode(frame, cols, rows, colourTransform);
            term.write(output, () => {
                writing = false; painted = version;
                fit();
                if (painted !== revision) schedule();
            });
        } catch (error) {
            writing=false;
            if (term.fitViewport) fallbackRenderer(error);
            else throw error;
        }
    }
    function schedule() {
        if (!document.hidden && !scheduled && !writing) { scheduled = true; requestAnimationFrame(paint); }
    }
    function updateResolution(value) {
        if (term) term.clearSelection();
        resolution = grid.clamp(value, 10, maxRows);
        [cols, rows] = grid.dimensions(resolution, defaults[0], defaults[1], maxCols, maxRows);
        status.textContent = `${cols}×${rows}`;
        more.disabled = rows === maxRows;
        less.disabled = rows === 10;
        refreshSettingsPreview();
        revision++;
        schedule();
    }
    function resetResolution() { updateResolution(defaults[1]); }
    function exportSettings() {
        return JSON.stringify({resolution: {cols, rows}, appearance: {...picture}, hideControls}, null, 2);
    }
    function refreshSettingsPreview() {
        const output = document.getElementById('settings-json');
        if (!output.hidden) output.value = exportSettings();
        document.getElementById('copy-settings').textContent = 'Copy settings';
    }
    function setControlsHidden(hidden) {
        hideControls = hidden === true;
        document.getElementById('picture-controls').hidden = hideControls;
        document.getElementById('resolution-controls').hidden = hideControls;
        document.getElementById('hide-controls').checked = hideControls;
        if (stage.classList) stage.classList.toggle('controls-open', !hideControls);
        layoutDirty = true; revision++; schedule();
        refreshSettingsPreview();
    }
    function applyAppearance() {
        if (term) {
            if (term.options.theme.background !== picture.background) {
                term.options.theme = {...term.options.theme, background: picture.background, black: '#000000'};
            }
            if (term.options.fontWeight !== picture.fontWeight) term.options.fontWeight = picture.fontWeight;
        }
        terminalElement.style.setProperty('--terminal-bg', picture.background);
        terminalElement.style.setProperty('--glyph-stroke', picture.fontStroke + 'px');
        colourTransform = appearance.transform(picture);
        layoutDirty = true;
        revision++; schedule(); refreshSettingsPreview();
    }
    for (const name of Object.keys(appearance.limits)) {
        const slider = document.getElementById(name + '-range');
        const number = document.getElementById(name + '-number');
        const format = value => name === 'fontWeight' ? String(value) : value.toFixed(2);
        slider.value = String(picture[name]); number.value = format(picture[name]);
        function update(source, final = false) {
            if (source.value.trim() === '' || !Number.isFinite(Number(source.value))) {
                if (final) source.value = format(picture[name]);
                return;
            }
            picture = appearance.normalize({...picture, [name]: Number(source.value)});
            slider.value = String(picture[name]);
            if (source !== number || final) number.value = format(picture[name]);
            applyAppearance();
        }
        slider.addEventListener('input', () => update(slider));
        number.addEventListener('input', () => update(number));
        number.addEventListener('change', () => update(number, true));
    }
    const bgSlider = document.getElementById('background-level');
    const bgNumber = document.getElementById('background-number');
    const bgColour = document.getElementById('background-colour');
    function updateBackgroundFields(level = null) {
        const value = level === null ? Math.round(appearance.toHsl(picture.background)[2] * 100) : level;
        bgSlider.value = String(value); bgNumber.value = String(value);
        bgColour.value = picture.background;
        document.getElementById('background-hex').textContent = picture.background;
    }
    function updateBackgroundLevel(source, final = false) {
        if (source.value.trim() === '' || !Number.isFinite(Number(source.value))) {
            if (final) updateBackgroundFields();
            return;
        }
        const level = Math.round(grid.clamp(Number(source.value), 0, 100));
        picture.background = appearance.fromHsl(backgroundBasis[0], backgroundBasis[1], level / 100);
        updateBackgroundFields(level); applyAppearance();
    }
    bgSlider.addEventListener('input', () => updateBackgroundLevel(bgSlider));
    bgNumber.addEventListener('input', () => updateBackgroundLevel(bgNumber));
    bgNumber.addEventListener('change', () => updateBackgroundLevel(bgNumber, true));
    bgColour.addEventListener('input', () => {
        picture = appearance.normalize({...picture, background: bgColour.value});
        backgroundBasis = appearance.toHsl(picture.background);
        updateBackgroundFields(); applyAppearance();
    });
    document.getElementById('picture-reset').addEventListener('click', () => {
        picture = {...initialAppearance};
        backgroundBasis = appearance.toHsl(picture.background);
        for (const name of Object.keys(appearance.limits)) {
            document.getElementById(name + '-range').value = String(picture[name]);
            document.getElementById(name + '-number').value = name === 'fontWeight' ?
                String(picture[name]) : picture[name].toFixed(2);
        }
        updateBackgroundFields(); applyAppearance();
    });
    document.getElementById('hide-controls').addEventListener('change', event => setControlsHidden(event.target.checked));
    document.getElementById('copy-settings').addEventListener('click', async () => {
        const output = document.getElementById('settings-json');
        output.hidden = false; output.value = exportSettings();
        const copied = output.value;
        try {
            await navigator.clipboard.writeText(copied);
            if (output.value === copied) document.getElementById('copy-settings').textContent = 'Copied';
        } catch (_) {
            output.focus(); output.select();
            document.getElementById('copy-settings').textContent = 'Select settings below';
        }
    });
    updateBackgroundFields(); setControlsHidden(hideControls);
    more.addEventListener('click', () => updateResolution(Math.floor(resolution) + 1));
    less.addEventListener('click', () => updateResolution(Math.ceil(resolution) - 1));
    document.getElementById('zoom-reset').addEventListener('click', resetResolution);
    stage.addEventListener('wheel', event => {
        event.preventDefault();
        const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
        const delta = grid.clamp(event.deltaY * unit, -200, 200);
        updateResolution(resolution * Math.exp(delta * (event.ctrlKey ? .005 : .0015)));
    }, { passive: false });
    const pointers = new Map();
    let pinch = null;
    function distance() {
        const points = Array.from(pointers.values());
        return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
    }
    stage.addEventListener('pointerdown', event => {
        event.preventDefault(); // This is a picture: gestures must not select terminal cells.
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        stage.setPointerCapture(event.pointerId);
        if (pointers.size === 2) pinch = { distance: distance(), resolution };
    });
    stage.addEventListener('pointermove', event => {
        if (!pointers.has(event.pointerId)) return;
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (pointers.size === 2 && pinch && pinch.distance > 0) {
            const current = distance();
            if (current > 0) updateResolution(pinch.resolution * pinch.distance / current);
        }
    });
    function release(event) { pointers.delete(event.pointerId); pinch = null; }
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => stage.addEventListener(name, release));
    window.addEventListener('keydown', event => {
        if (event.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) return;
        if (event.key === '+' || event.key === '=') {
            event.preventDefault(); updateResolution(Math.ceil(resolution) - 1);
        } else if (event.key === '-' || event.key === '_') {
            event.preventDefault(); updateResolution(Math.floor(resolution) + 1);
        } else if (event.key === '0') { event.preventDefault(); resetResolution(); }
    });
    window.addEventListener('resize', () => { layoutDirty = true; revision++; schedule(); });
    if (window.visualViewport) window.visualViewport.addEventListener('resize', () => { layoutDirty = true; revision++; schedule(); });

    function receiveText(text) {
        if (text[0] === '{') {
            try {
                const message = JSON.parse(text);
                if (message.type === 'resize' && Number.isInteger(message.cols) &&
                    Number.isInteger(message.rows) && message.cols > 0 && message.cols <= 160 &&
                    message.rows > 0 && message.rows <= 100) {
                    sourceCols = message.cols; sourceRows = message.rows;
                    maxCols = grid.clamp(sourceCols, 15, 160); maxRows = grid.clamp(sourceRows, 10, 100);
                    updateResolution(resolution);
                }
            } catch (_) { /* Ignore unknown control messages. */ }
            return;
        }
        const decoded = grid.decode(text, sourceCols, sourceRows, spareFrame);
        if (decoded) { spareFrame = frame; frame = decoded; revision++; schedule(); }
    }
    // One picture in flight: the server sends the next only after this.
    function acknowledge(socket, id) {
        if (ws === socket && !document.hidden && Number.isInteger(id) &&
                socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: 'ack', id}));
    }
    async function inflateLatest() {
        const socket = ws;
        if (!socket || inflating === socket) return;
        inflating = socket;
        try {
            // One in flight and ONE replaceable pending compressed frame.
            // An old, suspended connection must never consume or block a
            // resumed connection's pending frame.
            while (pendingCompressed && ws === socket && !document.hidden) {
                const item = pendingCompressed; pendingCompressed = null;
                const stream = new Blob([new Uint8Array(item.data,1)]).stream()
                    .pipeThrough(new DecompressionStream('gzip'));
                const text = await new Response(stream).text();
                if (ws === socket && !document.hidden) receiveText(text);
                acknowledge(item.socket, item.id);
            }
        } catch (_) {
            if (ws === socket) { pendingCompressed = null; socket.close(); }
        } finally { if (inflating === socket) inflating = null; }
    }

    function connect() {
        if (document.hidden) return;
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
        clearTimeout(reconnectTimer);
        const protocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
        const url = (!location.port || location.port === '80' || location.port === '443') ?
            protocol + location.host + '/ascii_ws/' : protocol + location.hostname + ':' + config.websocketPort;
        const socket = new WebSocket(url);
        let pictureId = null;
        socket.binaryType = 'arraybuffer'; ws = socket;
        socket.onopen = () => {
            if (ws !== socket || document.hidden) { socket.close(); return; }
            // Older browsers keep the identical uncompressed master stream.
            if (typeof DecompressionStream === 'function') {
                socket.send(JSON.stringify({type: 'encoding', encoding: 'gzip'}));
            }
            // Opt in to one-picture-in-flight delivery: each picture is acknowledged.
            socket.send(JSON.stringify({type: 'flow'}));
        };
        socket.onmessage = event => {
            if (ws !== socket) return;
            if (typeof event.data === 'string' && event.data[0] === '{') {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === 'frame' && Number.isInteger(message.id)) {
                        pictureId = message.id;
                        return;
                    }
                } catch (_) { /* receiveText handles other controls. */ }
            }
            if (event.data instanceof ArrayBuffer && new Uint8Array(event.data)[0] === 103) {
                // A picture replaced here is never decoded; acknowledge it now.
                if (pendingCompressed) acknowledge(pendingCompressed.socket, pendingCompressed.id);
                pendingCompressed = {socket, id: pictureId, data: event.data};
                pictureId = null;
                inflateLatest(); return;
            }
            const text = typeof event.data === 'string' ? event.data : new TextDecoder().decode(event.data);
            receiveText(text);
            if (text[0] !== '{') { acknowledge(socket, pictureId); pictureId = null; }
        };
        socket.onclose = () => {
            if (ws !== socket) return;
            ws = null;
            reconnectTimer = setTimeout(connect, 1000);
        };
        socket.onerror = () => socket.close();
    }
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            clearTimeout(reconnectTimer);
            pendingCompressed = null;
            const previous = ws; ws = null;
            if (previous) previous.close();
        } else { connect(); schedule(); }
    });

    function fallbackRenderer(error) {
        if (!term || !term.fitViewport) return;
        const options=term.options;
        term.dispose();
        term=new window.AsciiGridRenderer(options);
        term.open(terminalElement);
        terminalElement.dataset.renderer='grid';
        terminalElement.dataset.rendererFailure=String(error.message || error);
        rendererLayoutVersion=0; layoutDirty=true; writing=false;
        painted=-1; revision++; schedule();
    }
    function initialize() {
        const probe = document.createElement('span');
        probe.style.cssText = `font:${picture.fontWeight} 12px ${FONT_FAMILY};position:absolute;visibility:hidden;white-space:nowrap`;
        probe.textContent = 'M'.repeat(20); document.body.appendChild(probe);
        const measured = probe.getBoundingClientRect(); probe.remove();
        const lineHeight = Math.max(.1, measured.width / 20 / (measured.height * config.fontRatio));
        const Renderer = config.renderer === 'atlas' && window.AsciiAtlasRenderer ? window.AsciiAtlasRenderer :
            ((config.renderer === 'grid' || config.renderer === 'atlas') && window.AsciiGridRenderer ? window.AsciiGridRenderer : Terminal);
        term = new Renderer({ cols, rows, fontFamily: FONT_FAMILY, fontSize: 12,
            monospaced: true,
            onFailure: fallbackRenderer,
            allowSoftwareGL: config.allowSoftwareGL === true,
            lineHeight, letterSpacing: 1, fontWeight: picture.fontWeight, cursorBlink: false, disableStdin: true,
            scrollback: 0, theme: { background: '#000000', black: '#000000', foreground: '#e6e6e6',
                selectionBackground: '#00000000', selectionInactiveBackground: '#00000000' },
            allowTransparency: false });
        try { term.open(terminalElement); }
        catch (error) {
            if (!term.fitViewport) throw error;
            fallbackRenderer(error);
        }
        terminalElement.dataset.renderer=term.backend || 'grid';
        term.onSelectionChange(() => { if (term.hasSelection()) term.clearSelection(); });
        applyAppearance();
        resetResolution(); requestAnimationFrame(fit); connect();
    }
    Promise.all([document.fonts.load('400 16px ' + FONT_FAMILY),
        document.fonts.load('700 16px ' + FONT_FAMILY)]).then(initialize, initialize);
})();
