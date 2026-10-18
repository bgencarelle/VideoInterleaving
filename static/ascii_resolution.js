/* Pure client-side grid selection. No server resize requests or image filtering. */
(function (root) {
    'use strict';
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const basic = [0x000000, 0xcd0000, 0x00cd00, 0xcdcd00, 0x0000ee, 0xcd00cd,
        0x00cdcd, 0xe5e5e5, 0x7f7f7f, 0xff0000, 0x00ff00, 0xffff00,
        0x5c5cff, 0xff00ff, 0x00ffff, 0xffffff];
    function indexed(n) {
        if (n < 16) return basic[n];
        if (n >= 232) { const v = 8 + 10 * (n - 232); return (v << 16) | (v << 8) | v; }
        const levels = [0, 95, 135, 175, 215, 255];
        n -= 16;
        return (levels[Math.floor(n / 36)] << 16) |
            (levels[Math.floor(n / 6) % 6] << 8) | levels[n % 6];
    }
    function decode(text, width, height, reuse = null) {
        if (typeof text !== 'string' || text.length > 1024 * 1024 ||
            !Number.isInteger(width) || !Number.isInteger(height) ||
            width < 1 || width > 160 || height < 1 || height > 100) return null;
        const count = width * height;
        const frame = reuse && reuse.width === width && reuse.height === height ? reuse :
            {width,height,glyphs:new Array(count),colours:new Uint32Array(count),seen:new Uint8Array(count)};
        const {glyphs,colours} = frame;
        if (!frame.seen) frame.seen = new Uint8Array(count);
        const seen = frame.seen; seen.fill(0);
        let x = 0, y = 0, colour = 0xe6e6e6, populated = 0;
        const values = frame.parameters || (frame.parameters = new Int32Array(24));
        // Scan once: no regexp token array, split/map arrays or RGB slices per
        // colour instruction. At 90x60 there can be thousands per publication.
        let position = 0;
        while (position < text.length) {
            const code = text.charCodeAt(position++);
            if (code === 27) {
                if (text.charCodeAt(position++) !== 91) return null;
                if (text.charCodeAt(position) === 63) position++;
                let size = 0, number = 0, command = 0;
                while (position < text.length) {
                    const c = text.charCodeAt(position++);
                    if (c >= 48 && c <= 57) {
                        number = number * 10 + c - 48;
                        if (number > 1000000) return null;
                    } else if (c === 59) {
                        if (size >= values.length-1) return null;
                        values[size++] = number; number = 0;
                    } else if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) {
                        values[size++] = number; command = c; break;
                    } else return null;
                }
                if (!command) return null;
                if (command === 72) { y = (values[0] || 1)-1; x = (size > 1 ? values[1] || 1 : 1)-1; }
                if (command === 74 && values[0] === 2) { seen.fill(0); populated = 0; }
                if (command === 109) {
                    for (let i = 0; i < size; i++) {
                        const value = values[i];
                        if (value === 0 || value === 39) colour = 0xe6e6e6;
                        else if (value >= 30 && value <= 37) colour = basic[value - 30];
                        else if (value >= 90 && value <= 97) colour = basic[value - 90 + 8];
                        else if ((value === 38 || value === 48) && i+2 < size && values[i+1] === 5) {
                            const n = values[i + 2];
                            if (n > 255) return null;
                            if (value === 38) colour = indexed(n);
                            i += 2;
                        } else if ((value === 38 || value === 48) && i+4 < size && values[i+1] === 2) {
                            const r=values[i+2],g=values[i+3],b=values[i+4];
                            if (r > 255 || g > 255 || b > 255) return null;
                            if (value === 38) colour = (r<<16)|(g<<8)|b;
                            i += 4;
                        } else if (value === 38 || value === 48) {
                            return null;
                        }
                    }
                }
            } else {
                if (code === 13) { x = 0; continue; }
                if (code === 10) { y++; continue; }
                if (code < 32) continue;
                let glyph;
                if (code >= 0xd800 && code <= 0xdbff) {
                    const next = text.charCodeAt(position);
                    if (!(next >= 0xdc00 && next <= 0xdfff)) return null;
                    glyph = text.slice(position-1,++position);
                } else glyph = text[position-1];
                if (x >= width) { x = 0; y++; }
                if (x < 0 || y < 0 || y >= height) return null;
                const index = y * width + x++;
                if (!seen[index]) { populated++; seen[index] = 1; }
                glyphs[index] = glyph; colours[index] = colour;
            }
        }
        return populated === count ? frame : null;
    }
    function dimensions(rows, defaultCols, defaultRows, maxCols = 120, maxRows = 80) {
        maxCols = clamp(maxCols, 15, 160); maxRows = clamp(maxRows, 10, 100);
        rows = Math.round(clamp(rows, 10, maxRows));
        defaultRows = clamp(defaultRows, 10, maxRows);
        defaultCols = clamp(defaultCols, 15, maxCols);
        const width = rows === defaultRows ? defaultCols : rows < defaultRows ?
            15 + (defaultCols - 15) * (rows - 10) / (defaultRows - 10) :
            defaultCols + (maxCols - defaultCols) * (rows - defaultRows) / (maxRows - defaultRows);
        return [Math.round(width), rows];
    }
    function encode(frame, cols, rows, transform = null) {
        // SGR 40 selects ANSI palette black, which may be charcoal (Tango).
        // SGR 49 uses the explicitly black terminal theme background instead.
        const output = ['\x1b[?25l\x1b[?7l\x1b[0m\x1b[49m'];
        let previous = -1;
        // Bound colour work/cache to this one picture, not the stream lifetime.
        const adjusted = transform ? new Map() : null;
        for (let y = 0; y < rows; y++) {
            output.push(`\x1b[${y + 1};1H`);
            const sy = Math.min(frame.height - 1, Math.floor((y + 0.5) * frame.height / rows));
            for (let x = 0; x < cols; x++) {
                const sx = Math.min(frame.width - 1, Math.floor((x + 0.5) * frame.width / cols));
                const index = sy * frame.width + sx;
                let colour = frame.colours[index];
                if (transform) {
                    const original = colour;
                    if (adjusted.has(original)) colour = adjusted.get(original);
                    else { colour = transform(original); adjusted.set(original, colour); }
                }
                if (colour !== previous) {
                    output.push(`\x1b[38;2;${colour >> 16};${(colour >> 8) & 255};${colour & 255}m`);
                    previous = colour;
                }
                output.push(frame.glyphs[index]);
            }
        }
        return output.join('');
    }
    const api = { clamp, decode, dimensions, encode, indexed };
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.AsciiResolution = api;
})(typeof window === 'object' ? window : globalThis);
