/* Read-only grid renderer. Reuses coloured text runs instead of rebuilding a
 * terminal buffer/ANSI parser/DOM tree for every picture. Cell metrics and run
 * spacing match the bundled xterm DOM renderer, including proportional fonts.
 */
(function (root) {
    'use strict';
    class AsciiGridRenderer {
        constructor(options) {
            this.options = options;
            this.cols = options.cols; this.rows = options.rows;
            this.states = []; this.widths = new Map(); this.mapping = null;
            this.lastWeight = null;
        }
        open(host) {
            this.host = host;
            this.element = document.createElement('div');
            this.element.className = 'terminal xterm ascii-readonly-grid';
            this.element.setAttribute('role','img');
            this.element.setAttribute('aria-label','Live ASCII animation');
            this.screen = document.createElement('div'); this.screen.className = 'xterm-screen';
            this.container = document.createElement('div'); this.container.className = 'xterm-rows';
            this.container.style.cssText = `font-family:${this.options.fontFamily};font-size:${this.options.fontSize}px;font-kerning:none;font-variant-ligatures:none;white-space:pre;line-height:normal`;
            this.screen.appendChild(this.container); this.element.appendChild(this.screen); host.appendChild(this.element);
            const probe = document.createElement('span');
            probe.className = 'xterm-char-measure-element';
            probe.style.cssText = 'display:inline-block;visibility:hidden;position:absolute;top:0;left:-9999em;line-height:normal';
            probe.style.fontFamily = this.options.fontFamily;
            probe.style.fontSize = this.options.fontSize + 'px';
            probe.style.fontWeight = '400'; probe.style.fontKerning = 'none';
            probe.style.whiteSpace = 'pre'; probe.textContent = 'W'.repeat(32);
            this.element.appendChild(probe);
            this.charWidth = (this.options.monospaced ? probe.getBoundingClientRect().width : probe.offsetWidth) / 32;
            this.charHeight = Math.ceil(probe.offsetHeight);
            probe.remove();
            if (!this.options.monospaced) {
                this.measure = document.createElement('div');
                this.measure.style.cssText = `position:absolute;top:-50000px;width:50000px;white-space:pre;font-kerning:none;font-family:${this.options.fontFamily};font-size:${this.options.fontSize}px`;
                document.body.appendChild(this.measure);
            }
            this.resize(this.cols,this.rows);
        }
        resize(cols,rows) {
            this.cols = cols; this.rows = rows; this.mapping = null;
            const dpr = window.devicePixelRatio || 1;
            this.dpr = dpr; this.layoutVersion = (this.layoutVersion || 0) + 1;
            const width = Math.round((this.charWidth * dpr + Math.round(this.options.letterSpacing || 0)) * cols / dpr);
            const height = Math.round(Math.floor(Math.ceil(this.charHeight * dpr) * Math.max(1,this.options.lineHeight)) * rows / dpr);
            this.cellWidth = width / cols; this.cellHeight = height / rows;
            if (this.options.monospaced) this.container.style.letterSpacing = (this.cellWidth-this.charWidth)+'px';
            this.screen.style.width = width + 'px'; this.screen.style.height = height + 'px';
            while (this.states.length > rows) this.states.pop().element.remove();
            while (this.states.length < rows) {
                const element = document.createElement('div'); this.container.appendChild(element);
                this.states.push({element,nodes:[],texts:[],colours:[],spacing:[]});
            }
            for (const row of this.states) row.element.style.cssText =
                `width:${width}px;height:${this.cellHeight}px;line-height:${this.cellHeight}px;overflow:hidden;contain:strict`;
        }
        clearSelection() {}
        hasSelection() { return false; }
        onSelectionChange() {}
        prepareWidths(frame) {
            if (this.lastWeight !== this.options.fontWeight) {
                this.lastWeight = this.options.fontWeight;
                this.container.style.fontWeight = String(this.lastWeight);
                if (this.measure) this.measure.style.fontWeight = String(this.lastWeight);
                this.widths.clear();
            }
            if (this.options.monospaced) return;
            const missing = new Set();
            for (const index of this.mapping) {
                const glyph = frame.glyphs[index];
                if (!this.widths.has(glyph)) missing.add(glyph);
            }
            if (!missing.size) return;
            if (this.widths.size + missing.size > 512) {
                this.widths.clear();
                for (const index of this.mapping) missing.add(frame.glyphs[index]);
            }
            // Batch measurement writes before reads: at most one font-layout
            // flush for new glyphs, not one forced layout for every new cell.
            const probes = [];
            for (const glyph of missing) {
                const span = document.createElement('span'); span.textContent = glyph.repeat(32);
                this.measure.appendChild(span); probes.push([glyph,span]);
            }
            for (const [glyph,span] of probes) this.widths.set(glyph,span.offsetWidth / 32);
            this.measure.replaceChildren();
        }
        renderGrid(frame,cols,rows,transform) {
            if (cols !== this.cols || rows !== this.rows || (window.devicePixelRatio || 1) !== this.dpr) this.resize(cols,rows);
            if (!this.mapping || this.sourceWidth !== frame.width || this.sourceHeight !== frame.height) {
                this.sourceWidth = frame.width; this.sourceHeight = frame.height;
                this.mapping = new Uint16Array(cols * rows);
                for (let y=0;y<rows;y++) for (let x=0;x<cols;x++) {
                    this.mapping[y*cols+x] = Math.floor((y+.5)*frame.height/rows)*frame.width +
                        Math.floor((x+.5)*frame.width/cols);
                }
            }
            this.prepareWidths(frame);
            const adjusted = transform ? new Map() : null;
            for (let y=0;y<rows;y++) {
                const state = this.states[y];
                let count=0, previous=-1, previousSpacing=NaN, text='';
                const commit = () => {
                    let span = state.nodes[count];
                    if (!span) {
                        span = document.createElement('span');
                        span.style.cssText = 'display:inline-block;height:100%;vertical-align:top';
                        state.element.appendChild(span); state.nodes.push(span);
                    }
                    if (state.texts[count] !== text) { span.textContent=text; state.texts[count]=text; }
                    if (state.colours[count] !== previous) {
                        span.style.color='#'+previous.toString(16).padStart(6,'0'); state.colours[count]=previous;
                    }
                    if (!this.options.monospaced && state.spacing[count] !== previousSpacing) {
                        span.style.letterSpacing=previousSpacing+'px'; state.spacing[count]=previousSpacing;
                    }
                    count++;
                };
                for (let x=0;x<cols;x++) {
                    const index=this.mapping[y*cols+x], glyph=frame.glyphs[index];
                    let colour=frame.colours[index];
                    if (transform) {
                        const original=colour;
                        if (adjusted.has(original)) colour=adjusted.get(original);
                        else { colour=transform(original); adjusted.set(original,colour); }
                    }
                    const spacing=this.options.monospaced ? 0 : this.cellWidth-this.widths.get(glyph);
                    if (colour!==previous || (!this.options.monospaced && spacing!==previousSpacing)) {
                        if (text) commit();
                        text=''; previous=colour; previousSpacing=spacing;
                    }
                    text+=glyph;
                }
                if (text) commit();
                while (state.nodes.length>count) state.nodes.pop().remove();
                state.texts.length=state.colours.length=state.spacing.length=count;
            }
        }
        dispose() { if (this.measure) this.measure.remove(); this.element.remove(); this.states=[]; this.widths.clear(); }
    }
    root.AsciiGridRenderer = AsciiGridRenderer;
})(typeof window === 'object' ? window : globalThis);
