'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const grid = require('../static/ascii_resolution.js');
const appearance = require('../static/ascii_appearance.js');
assert.equal(appearance.transform(appearance.defaults), null);
assert.equal(appearance.defaults.fontWeight,900);
assert.equal(appearance.defaults.fontStroke,0);
assert.equal(appearance.defaults.lineSpacing,1.05);
assert.equal(appearance.transform({background:'#445566', fontWeight:700}), null);
assert.equal(appearance.transform({fontStroke:.25}), null);
assert.equal(appearance.transform({lineSpacing:.75}), null);
for (const rows of [1,10,40,60]) for (const spacing of [.5,.75,1,1.5,2]) {
    const layout = appearance.rowLayout(rows,rows*16,spacing);
    assert.equal(layout.height,16*(1+(rows-1)*spacing));
    assert.equal(layout.offsets[0],0);
    assert.equal((rows-1)*16+layout.offsets.at(-1)+16,layout.height);
    assert.deepEqual(appearance.rowLayout(rows,rows*16,spacing),layout);
}
assert.throws(() => appearance.rowLayout(0,640,1),RangeError);
assert.equal(appearance.transform({brightness:0})(0xffffff), 0);
assert.equal(appearance.transform({contrast:0})(0x123456), 0x808080);
assert.equal(appearance.transform({gamma:2})(0x808080), 0xb5b5b5);
assert.equal(appearance.transform({saturation:0})(0x00ff00), 0xb6b6b6);
assert.equal(appearance.transform({gamma:2})(0xffffff), 0xffffff);
assert.equal(appearance.transform({gamma:2})(0), 0);
assert.equal(appearance.normalize({gamma:NaN}).gamma, 1);
assert.equal(appearance.normalize({gamma:-10}).gamma, .25);
assert.equal(appearance.normalize({brightness:Infinity}).brightness, 1);
assert.equal(appearance.normalize({background:'bad'}).background, '#000000');
assert.equal(appearance.fromHsl(0,0,0), '#000000');
assert.equal(appearance.fromHsl(0,0,1), '#ffffff');
assert.equal(appearance.fromHsl(0,0,.5), '#808080');
for (const colour of ['#000000','#ffffff','#123456','#ffaaff','#ff0000','#00ff00','#0000ff']) {
    assert.equal(appearance.fromHsl(...appearance.toHsl(colour)), colour);
}

function source(width, height, glyph = 'M') {
    return '\x1b[H' + Array.from({length: height}, (_, y) =>
        Array.from({length: width}, (_, x) => `\x1b[38;2;${x % 256};${y % 256};120m${glyph}`).join(''))
        .join('\r\n') + '\x1b[0m';
}
assert.deepEqual(grid.dimensions(40, 60, 40), [60, 40]);
assert.deepEqual(grid.dimensions(32, 50, 32), [50, 32]);
assert.deepEqual(grid.dimensions(-100, 60, 40), [15, 10]);
assert.deepEqual(grid.dimensions(1000, 60, 40), [120, 80]);
assert.deepEqual(grid.dimensions(1000, 60, 40, 90, 60), [90, 60]);
for (let row = 10; row <= 80; row++) {
    const size = grid.dimensions(row, 60, 40);
    assert.deepEqual(size, [Math.round(row * 1.5), row]);
}
const master = grid.decode(source(120, 80), 120, 80);
assert(master);
const unchanged = Array.from(master.colours);
const adjusted = grid.decode(grid.encode(master,60,40,appearance.transform({brightness:0})),60,40);
assert(adjusted.colours.every(value => value === 0));
assert.deepEqual(Array.from(master.colours), unchanged);
assert.equal(grid.encode(master,60,40), grid.encode(master,60,40,appearance.transform(appearance.defaults)));
let transformCalls = 0;
const uniform = grid.decode(source(1,1),1,1);
grid.encode(uniform,90,60,value => { transformCalls++; return value; });
assert.equal(transformCalls,1);
assert(grid.encode(master, 60, 40).includes('\x1b[49m'));
assert(!grid.encode(master, 60, 40).includes('\x1b[40m'));
for (let rows = 10; rows <= 80; rows++) {
    const [cols] = grid.dimensions(rows, 60, 40);
    const output = grid.decode(grid.encode(master, cols, rows), cols, rows);
    assert(output);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const sx = Math.floor((x + .5) * 120 / cols), sy = Math.floor((y + .5) * 80 / rows);
        assert.equal(output.colours[y * cols + x], master.colours[sy * 120 + sx]);
        assert.equal(output.glyphs[y * cols + x], master.glyphs[sy * 120 + sx]);
    }
}
assert.equal(grid.decode(source(90, 60), 60, 40), null);
assert.equal(grid.decode('incomplete', 90, 60), null);
assert.equal(grid.decode('\x1b[38;2;999;0;0mM', 1, 1), null);
assert.equal(grid.decode('\x1b[38;5;999mM', 1, 1), null);
assert.equal(grid.decode('M', 0, 1), null);
assert.equal(grid.decode('M', Infinity, 1), null);
assert.equal(grid.decode('x'.repeat(1024 * 1024 + 1), 90, 60), null);
const reusable=grid.decode(source(9,6),9,6);
const oldColours=reusable.colours;
const oldGlyphs=reusable.glyphs;
assert.equal(grid.decode(source(9,6,'Z'),9,6,reusable),reusable);
assert.equal(reusable.colours,oldColours);assert.equal(reusable.glyphs,oldGlyphs);
assert(reusable.glyphs.every(glyph=>glyph==='Z'));
assert.equal(grid.decode('\x1b[38;2;1mM',1,1),null);
assert.equal(grid.decode('\x1b[999999999999H',9,6),null);
assert.equal(grid.decode('\x1b[38;2;1;2;3m\x1b[48;2;30;31;32mM',1,1).colours[0],0x010203);
const indexed = grid.decode('\x1b[38;5;196mM\x1b[38;5;244m.\x1b[38;5;16m ', 3, 1);
assert.deepEqual(Array.from(indexed.colours), [0xff0000, 0x808080, 0]);
const unicode = grid.decode('\x1b[38;2;4;5;6mé', 1, 1);
assert.equal(grid.decode(grid.encode(unicode, 1, 1), 1, 1).glyphs[0], 'é');
assert.deepEqual(Array.from(grid.decode(grid.encode(master, 120, 80), 120, 80).colours),
    Array.from(master.colours));

// Browser-controller test with a deliberately slow xterm and no network sends.
class Element {
    constructor() {
        this.style = {setProperty(name,value) { this[name] = value; }};
        this.listeners = {}; this.textContent = ''; this.tagName = 'DIV';
        this.hidden = false; this.value = ''; this.dataset = {};
    }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    setPointerCapture() {}
    querySelector() { return {clientWidth: 480, clientHeight: 640}; }
    querySelectorAll() {
        const rows = this.terminal ? this.terminal.rows : 0;
        if (!this.rowElements || this.rowElements.length !== rows) {
            this.rowElements = Array.from({length:rows}, () => new Element());
        }
        return this.rowElements;
    }
    getBoundingClientRect() { return {width: 144, height: 12}; }
    appendChild() {}
    remove() {}
    focus() {}
    select() {}
}
const controlIds = ['terminal-stage','terminal','resolution','zoom-out','zoom-in','zoom-reset',
    'resolution-controls','picture-controls','show-controls','hide-controls','settings-json',
    'background-level','background-number','background-colour','background-hex','picture-reset','copy-settings',
    ...Object.keys(appearance.limits).flatMap(name => [name+'-range',name+'-number'])];
const ids = Object.fromEntries(controlIds
    .map(id => [id, new Element()]));
ids['settings-json'].hidden = true;
const raf = [], sockets = [], writes = [], callbacks = [], terminals = [];
class MockTerminal {
    constructor(options) {
        this.cols = options.cols; this.rows = options.rows; this.options = options; terminals.push(this);
    }
    open(element) { element.terminal = this; }
    clearSelection() { this.selected = false; }
    hasSelection() { return this.selected || false; }
    onSelectionChange(callback) { this.selectionChanged = callback; }
    resize(cols, rows) { this.cols = cols; this.rows = rows; }
    write(output, callback) { writes.push(output); callbacks.push(callback); }
}
class MockSocket {
    static OPEN = 1;
    static CONNECTING = 0;
    constructor(url) { this.url = url; this.readyState = 1; this.sent = []; sockets.push(this); }
    send(value) {
        const data = JSON.parse(value);
        assert.deepEqual(data, {type:'encoding', encoding:'gzip'});
        this.sent.push(value); // No resize messages are permitted.
    }
    close() { this.readyState = 3; this.onclose(); }
}
const window = {AsciiResolution: grid, AsciiAppearance: appearance,
    ASCII_VIEWER_CONFIG: {cols: 50, rows: 32, sourceCols: 90, sourceRows: 60,
        fontRatio: .5, websocketPort: 2424},
    innerWidth: 800, innerHeight: 600, listeners: {},
    addEventListener(name, handler) { this.listeners[name] = handler; }};
const fontLoads = [];
const document = {body: new Element(), hidden: false, listeners: {},
    getElementById: id => ids[id], createElement: () => new Element(),
    addEventListener(name, handler) { this.listeners[name] = handler; },
    fonts: {load: font => {fontLoads.push(font);return Promise.resolve();}}};
const inflateCompletions = [];
let inflations = 0;
class ObservedResponse extends Response {
    constructor(stream) { super(stream); inflations++; }
    text() {
        return super.text().then(text => {
            const complete = inflateCompletions.shift();
            if (complete) queueMicrotask(complete);
            return text;
        });
    }
}
const sandbox = {window, document, WebSocket: MockSocket, Terminal: MockTerminal,
    location: {protocol:'https:', port:'', host:'example.test', hostname:'example.test'},
    TextDecoder, ArrayBuffer, Uint8Array, Blob, DecompressionStream, Response:ObservedResponse,
    requestAnimationFrame: callback => raf.push(callback), setTimeout: () => 1,
    clearTimeout() {}};
let copied = '';
sandbox.navigator = {clipboard: {async writeText(value) { copied = value; }}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/ascii_viewer.js'), 'utf8'), sandbox);
function flush() { while (raf.length) raf.shift()(); }
function wheel(deltaY) { ids['terminal-stage'].listeners.wheel({deltaY, deltaMode: 0,
    ctrlKey: true, preventDefault() {}}); }
function key(value) { window.listeners.keydown({key:value, target:ids.terminal, preventDefault() {}}); }
(async () => {
    await new Promise(resolve => setImmediate(resolve)); // Both font faces finish before initial fitting.
    assert.equal(sockets.length, 1);
    assert.equal(fontLoads.length,2);
    assert(fontLoads.every(font=>font.includes('Interleave Schoolbook Mono')));
    assert.equal(terminals[0].options.monospaced,true);
    assert.equal(sockets[0].url, 'wss://example.test/ascii_ws/');
    assert.equal(ids.resolution.textContent, '50×32');
    assert.equal(terminals[0].options.theme.background, '#000000');
    assert.equal(terminals[0].options.theme.black, '#000000');
    assert.equal(terminals[0].options.fontWeight,900);
    assert.equal(ids['fontStroke-number'].value,'0.00');
    assert.equal(ids['lineSpacing-number'].value,'1.05');
    assert.equal(ids['brightness-number'].value, '1.00');
    assert.equal(ids['gamma-number'].value, '1.00');
    assert.equal(ids['picture-controls'].hidden, false);
    assert.equal(ids['show-controls'].hidden, true);
    sockets[0].onmessage({data: JSON.stringify({type:'resize', cols:90, rows:60})});
    for (const glyph of ['A','B','C']) sockets[0].onmessage({data: source(90, 60, glyph)});
    flush();
    assert.equal(writes.length, 1);
    assert(writes[0].includes('C') && !writes[0].includes('A') && !writes[0].includes('B'));
    for (let i = 0; i < 10; i++) wheel(-200);
    assert.equal(ids.resolution.textContent, '15×10');
    assert(ids['zoom-in'].disabled);
    sockets[0].onmessage({data:source(90,60,'D')});
    sockets[0].onmessage({data:source(90,60,'E')});
    flush();
    assert.equal(writes.length, 1); // No queued xterm history while a write is pending.
    callbacks.shift()(); flush();
    assert.equal(writes.length, 2);
    assert.equal(terminals[0].cols, 15); assert.equal(terminals[0].rows, 10);
    assert(writes[1].includes('E') && !writes[1].includes('D'));
    callbacks.shift()(); flush();
    for (let i = 0; i < 10; i++) wheel(200);
    assert.equal(ids.resolution.textContent, '90×60');
    assert(ids['zoom-out'].disabled);
    key('0'); assert.equal(ids.resolution.textContent, '50×32');
    key('+'); assert.equal(ids.resolution.textContent, '48×31');
    key('-'); assert.equal(ids.resolution.textContent, '50×32');
    const events = ids['terminal-stage'].listeners;
    let prevented = 0;
    events.pointerdown({pointerId:1, clientX:0, clientY:0, preventDefault() { prevented++; }});
    events.pointerdown({pointerId:2, clientX:100, clientY:0, preventDefault() { prevented++; }});
    assert.equal(prevented, 2);
    terminals[0].selected = true;
    terminals[0].selectionChanged();
    assert.equal(terminals[0].hasSelection(), false);
    events.pointermove({pointerId:2, clientX:200, clientY:0});
    assert.equal(ids.resolution.textContent, '25×16'); // Spread fingers -> fewer, bigger cells.
    events.pointerup({pointerId:2}); events.pointerup({pointerId:1});
    key('0');
    sockets[0].onclose(); document.listeners.visibilitychange();
    assert.equal(sockets.length, 2);
    assert.equal(ids.resolution.textContent, '50×32');
    assert.equal(typeof document.listeners.visibilitychange, 'function');
    assert.equal(sockets[0].sent.length, 0); // All zoom interactions were local.
    while (callbacks.length) { callbacks.shift()(); flush(); }
    sockets[1].onopen();
    assert.equal(sockets[1].sent.length, 1); // One encoding negotiation on reconnect.
    const {gzipSync} = require('node:zlib');
    const firstInflate = new Promise(resolve => inflateCompletions.push(resolve));
    const lastInflate = new Promise(resolve => inflateCompletions.push(resolve));
    for (const glyph of ['F','G','H']) {
        const packed = Buffer.concat([Buffer.from('g'), gzipSync(source(90,60,glyph))]);
        sockets[1].onmessage({data:packed.buffer.slice(packed.byteOffset,packed.byteOffset+packed.byteLength)});
    }
    await Promise.all([firstInflate,lastInflate]);
    await Promise.resolve(); flush();
    assert.equal(inflations, 2); // G was replaced while F inflated; H is next.
    assert(writes.at(-1).includes('H'));
    assert(!writes.some(value => value.includes('G')));
    ids['brightness-range'].value = '1.25'; ids['brightness-range'].listeners.input();
    assert.equal(ids['brightness-number'].value, '1.25');
    ids['contrast-number'].value = '1.40'; ids['contrast-number'].listeners.input();
    assert.equal(ids['contrast-range'].value, '1.4');
    ids['gamma-number'].value = ''; ids['gamma-number'].listeners.change();
    assert.equal(ids['gamma-number'].value, '1.00');
    ids['background-level'].value = '50'; ids['background-level'].listeners.input();
    assert.equal(ids['background-hex'].textContent, '#808080');
    assert.equal(terminals[0].options.theme.background, '#808080');
    assert.equal(terminals[0].options.theme.black, '#000000');
    assert.equal(ids.terminal.style['--terminal-bg'], '#808080');
    ids['background-colour'].value = '#ff0000'; ids['background-colour'].listeners.input();
    ids['background-level'].value = '0'; ids['background-level'].listeners.input();
    ids['background-level'].value = '50'; ids['background-level'].listeners.input();
    assert.equal(ids['background-hex'].textContent, '#ff0000'); // Hue survives black.
    ids['fontWeight-range'].value = '700'; ids['fontWeight-range'].listeners.input();
    assert.equal(terminals[0].options.fontWeight,700);
    ids['fontStroke-range'].value = '0.25'; ids['fontStroke-range'].listeners.input();
    assert.equal(ids.terminal.style['--glyph-stroke'], '0.25px');
    assert.equal(ids['fontStroke-number'].value, '0.25');
    ids['lineSpacing-range'].value = '0.75'; ids['lineSpacing-range'].listeners.input();
    while (callbacks.length) { callbacks.shift()(); flush(); }
    flush();
    assert.equal(ids['lineSpacing-number'].value, '0.75');
    const rowCount = terminals[0].rows;
    const packed = appearance.rowLayout(rowCount,640,.75);
    assert.equal(ids.terminal.style.height,packed.height+'px');
    assert.equal(ids.terminal.rowElements.at(-1).style.transform,
        `translateY(${packed.offsets.at(-1)}px)`);
    const fitBefore = ids.terminal.style.transform;
    assert(fitBefore.includes(String(600/packed.height))); // Same fitted picture height.
    window.listeners.resize(); flush();
    assert.equal(ids.terminal.style.transform,fitBefore); // No repeated-fit feedback.
    await ids['copy-settings'].listeners.click();
    const settings = JSON.parse(copied);
    assert.equal(settings.appearance.brightness,1.25);
    assert.equal(settings.appearance.contrast,1.4);
    assert.equal(settings.appearance.fontWeight,700);
    assert.equal(settings.appearance.fontStroke,.25);
    assert.equal(settings.appearance.lineSpacing,.75);
    assert.equal(settings.appearance.background,'#ff0000');
    assert.equal(settings.hideControls,false);
    assert.equal(ids['settings-json'].hidden,false);
    ids['hide-controls'].checked = true;
    ids['hide-controls'].listeners.change({target:ids['hide-controls']});
    assert.equal(ids['picture-controls'].hidden,true);
    assert.equal(ids['resolution-controls'].hidden,true);
    assert.equal(ids['show-controls'].hidden,false);
    key('h');
    assert.equal(ids['picture-controls'].hidden,false);
    key('H'); ids['show-controls'].listeners.click();
    assert.equal(ids['picture-controls'].hidden,false);
    ids['picture-reset'].listeners.click();
    assert.equal(ids['brightness-number'].value,'1.00');
    assert.equal(ids['contrast-number'].value,'1.00');
    assert.equal(terminals[0].options.theme.background,'#000000');
    assert.equal(terminals[0].options.fontWeight,900);
    assert.equal(ids.terminal.style['--glyph-stroke'], '0px');
    flush();
    while (callbacks.length) { callbacks.shift()(); flush(); }
    flush();
    assert.equal(ids['lineSpacing-number'].value,'1.05');
    const resetLayout=appearance.rowLayout(terminals[0].rows,640,1.05);
    assert.equal(ids.terminal.style.height,resetLayout.height+'px');
    assert.equal(ids.terminal.rowElements.at(-1).style.transform,`translateY(${resetLayout.offsets.at(-1)}px)`);
    assert.equal(window.ASCII_VIEWER_CONFIG.appearance,undefined);
    assert.equal(sockets[1].sent.length,1); // No server requests from any picture control.
    document.hidden=true;document.listeners.visibilitychange();
    assert.equal(sockets[1].readyState,3);
    const previousWrites=writes.length;flush();assert.equal(writes.length,previousWrites);
    document.hidden=false;document.listeners.visibilitychange();
    assert.equal(sockets.length,3); // Fresh stream on return, no hidden-tab replay.
    assert.equal(ids.resolution.textContent,'50×32');
    // Explicit startup boolean hides both panels without hiding the restore button.
    const hiddenIds = Object.fromEntries(controlIds.map(id => [id,new Element()]));
    hiddenIds['settings-json'].hidden = true;
    const hiddenWindow = {...window, ASCII_VIEWER_CONFIG:{...window.ASCII_VIEWER_CONFIG, hideControls:true}};
    const hiddenDocument = {...document, getElementById:id => hiddenIds[id]};
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/ascii_viewer.js'),'utf8'),
        {...sandbox,window:hiddenWindow,document:hiddenDocument});
    await Promise.resolve();
    assert.equal(hiddenIds['picture-controls'].hidden,true);
    assert.equal(hiddenIds['resolution-controls'].hidden,true);
    assert.equal(hiddenIds['show-controls'].hidden,false);
    await new Promise(resolve=>setImmediate(resolve));flush();
    // Canvas/atlas draws synchronously and must retain latest-only scheduling
    // and all local settings when a context loss switches to the DOM fallback.
    class MockGrid extends MockTerminal {
        constructor(options){super(options);this.draws=[];this.layoutVersion=1;}
        renderGrid(frame,cols,rows,transform){this.draws.push({frame,cols,rows,transform});}
        dispose(){this.disposed=true;}
    }
    class MockAtlas extends MockGrid {
        constructor(options){super(options);this.backend='canvas';}
        open(element){super.open(element);if(MockAtlas.failOpen)throw Error('No canvas');}
        fitViewport(width,height,picture){this.fitted={width,height,picture};this.dpr=1;}
    }
    async function atlasCase(failOpen=false){
        const elements=Object.fromEntries(controlIds.map(id=>[id,new Element()]));
        elements['settings-json'].hidden=true;
        const localWindow={...window,listeners:{},AsciiAtlasRenderer:MockAtlas,AsciiGridRenderer:MockGrid,
            ASCII_VIEWER_CONFIG:{...window.ASCII_VIEWER_CONFIG,renderer:'atlas'}};
        const localDocument={...document,listeners:{},getElementById:id=>elements[id]};
        MockAtlas.failOpen=failOpen;
        const count=terminals.length;
        vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../static/ascii_viewer.js'),'utf8'),
            {...sandbox,window:localWindow,document:localDocument});
        await new Promise(resolve=>setImmediate(resolve));flush();
        return {elements,localWindow,localDocument,renderer:terminals[count],socket:sockets.at(-1)};
    }
    const canvas=await atlasCase();
    assert.equal(canvas.elements.terminal.dataset.renderer,'canvas');
    for(const glyph of ['A','B','C'])canvas.socket.onmessage({data:source(90,60,glyph)});
    flush();assert.equal(canvas.renderer.draws.length,1);
    assert(canvas.renderer.draws[0].frame.glyphs.every(glyph=>glyph==='C'));
    canvas.elements['fontWeight-range'].value='700';canvas.elements['fontWeight-range'].listeners.input();
    canvas.elements['brightness-range'].value='1.2';canvas.elements['brightness-range'].listeners.input();
    canvas.elements['lineSpacing-range'].value='.75';canvas.elements['lineSpacing-range'].listeners.input();
    flush();assert.equal(canvas.renderer.options.fontWeight,700);
    const draws=canvas.renderer.draws.length;
    canvas.localWindow.listeners.resize();flush();assert.equal(canvas.renderer.draws.length,draws+1);
    canvas.renderer.options.onFailure(new Error('context lost'));flush();
    const recovered=terminals.at(-1);
    assert(canvas.renderer.disposed);assert(recovered instanceof MockGrid);
    assert.equal(canvas.elements.terminal.dataset.renderer,'grid');
    assert.equal(recovered.options.fontWeight,700);
    assert(recovered.draws.at(-1).frame.glyphs.every(glyph=>glyph==='C'));
    assert(recovered.draws.at(-1).transform); // Tone transform survives fallback.
    assert.equal(canvas.elements['lineSpacing-number'].value,'0.75');
    canvas.socket.onmessage({data:JSON.stringify({type:'resize',cols:120,rows:80})});
    canvas.socket.onmessage({data:source(120,80,'Q')});
    for(let i=0;i<80;i++)canvas.elements['zoom-out'].listeners.click();
    flush();assert.equal(canvas.elements.resolution.textContent,'120×80');
    assert.equal(recovered.draws.at(-1).cols,120);assert.equal(recovered.draws.at(-1).rows,80);
    assert(recovered.draws.at(-1).frame.glyphs.every(glyph=>glyph==='Q'));
    assert(canvas.elements['zoom-out'].disabled);
    // Old 90x60 bakes / servers cap the view instead of inventing new detail.
    canvas.socket.onmessage({data:JSON.stringify({type:'resize',cols:90,rows:60})});
    canvas.socket.onmessage({data:source(90,60,'R')});flush();
    assert.equal(canvas.elements.resolution.textContent,'90×60');
    const unavailable=await atlasCase(true);
    assert.equal(unavailable.elements.terminal.dataset.renderer,'grid');
    unavailable.socket.onmessage({data:source(90,60,'Z')});flush();
    assert(terminals.at(-1).draws.at(-1).frame.glyphs.every(glyph=>glyph==='Z'));
    console.log('ASCII zoom, appearance values/reset/export, background hue, hideControls, latest-only gzip: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
