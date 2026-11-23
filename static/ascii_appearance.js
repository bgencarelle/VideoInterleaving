/* Local display adjustments, applied to master RGB tints, never cumulatively. */
(function (root) {
    'use strict';
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const defaults = Object.freeze({brightness:1, contrast:1, gamma:1, saturation:1,
        fontWeight:900, fontStroke:0, lineSpacing:1.05, background:'#000000'});
    const limits = Object.freeze({brightness:[0,3], contrast:[0,3], gamma:[.25,3],
        saturation:[0,3], fontWeight:[100,900], fontStroke:[0,1], lineSpacing:[.5,2]});
    function normalize(input) {
        const values = {...defaults};
        for (const key of Object.keys(limits)) {
            if (input && Number.isFinite(input[key])) values[key] = clamp(input[key], ...limits[key]);
        }
        values.fontWeight = Math.round(values.fontWeight / 100) * 100;
        if (input && /^#[0-9a-f]{6}$/i.test(input.background)) values.background = input.background.toLowerCase();
        return values;
    }
    function transform(values) {
        const s = normalize(values);
        if (s.brightness === 1 && s.contrast === 1 && s.gamma === 1 && s.saturation === 1) return null;
        const channel = value => Math.round(255 * Math.pow(
            clamp(((value - 127.5) * s.contrast + 127.5) * s.brightness / 255, 0, 1), 1 / s.gamma));
        return colour => {
            const r = colour >> 16, g = (colour >> 8) & 255, b = colour & 255;
            const luma = .2126 * r + .7152 * g + .0722 * b;
            return (channel(luma + (r - luma) * s.saturation) << 16) |
                (channel(luma + (g - luma) * s.saturation) << 8) |
                channel(luma + (b - luma) * s.saturation);
        };
    }
    function toHsl(hex) {
        const rgb = [1,3,5].map(i => parseInt(hex.slice(i,i+2),16) / 255);
        const [r,g,b] = rgb, hi = Math.max(...rgb), lo = Math.min(...rgb);
        const lightness = (hi + lo) / 2, delta = hi - lo;
        if (!delta) return [0,0,lightness];
        let hue = hi === r ? ((g-b)/delta + (g<b ? 6 : 0)) : hi === g ? (b-r)/delta+2 : (r-g)/delta+4;
        return [hue/6, delta/(1-Math.abs(2*lightness-1)), lightness];
    }
    function fromHsl(hue, saturation, lightness) {
        const l = clamp(lightness,0,1), s = clamp(saturation,0,1);
        const a = s * Math.min(l, 1-l);
        const channel = n => {
            const k = (n+hue*12)%12;
            return Math.round(255*(l-a*Math.max(-1,Math.min(k-3,9-k,1))));
        };
        return '#' + [0,8,4].map(n => channel(n).toString(16).padStart(2,'0')).join('');
    }
    function rowLayout(rows, height, spacing) {
        if (!Number.isInteger(rows) || rows < 1 || !Number.isFinite(height) || height <= 0) {
            throw new RangeError('Positive row count and screen height required');
        }
        spacing = Number.isFinite(spacing) ? clamp(spacing,.5,2) : 1;
        const cell = height / rows;
        return {height:cell * (1 + (rows-1) * spacing),
            offsets:Array.from({length:rows}, (_,i) => i === 0 ? 0 : i * cell * (spacing-1))};
    }
    const api = {defaults, limits, normalize, transform, toHsl, fromHsl, rowLayout};
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.AsciiAppearance = api;
})(typeof window === 'object' ? window : globalThis);
