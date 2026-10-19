const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const root = path.join(__dirname, '..');

function browser() {
    let next = 0;
    const timers = new Map(), intervals = new Map(), elements = new Map();
    const classes = () => ({add(){},remove(){},toggle(){},replace(){},contains(){return false;}});
    const element = id => {
        if (!elements.has(id)) elements.set(id, {
            src:'', value:id==='preview-exposure'?'1':'512', checked:true,
            style:{},classList:classes(), writes:[], listeners:{},
            set src(value){this._src=value;this.writes.push(value);},
            get src(){return this._src||'';},
            removeAttribute(name){if(name==='src')this._src='';},
            getAttribute(name){return name==='src'?this.src:null;},
            addEventListener(name,fn){this.listeners[name]=fn;}
        });
        return elements.get(id);
    };
    const document = {hidden:false,listeners:{},body:{classList:classes()},
        getElementById:element,addEventListener(name,fn){this.listeners[name]=fn;}};
    const context = vm.createContext({document,console:{log(){}},Date,Float32Array,Math,
        LiveImageStream: class {
            constructor(image){this.image=image;}
            start(url){this.image.src=url;}
            stop(){this.image.removeAttribute('src');}
        },
        window:{addEventListener(){},matchMedia(){return {matches:false};}},
        setTimeout(fn){const id=++next;timers.set(id,fn);return id;},
        clearTimeout(id){timers.delete(id);},
        setInterval(fn){const id=++next;intervals.set(id,fn);return id;},
        clearInterval(id){intervals.delete(id);}});
    return {context,document,element,timers,intervals,
        run(code){return vm.runInContext(code,context);},
        hide(){document.hidden=true;document.listeners.visibilitychange();},
        show(){document.hidden=false;document.listeners.visibilitychange();}};
}

test('web MJPEG drops hidden stream and rejects old status replies',async()=>{
    const b=browser();const replies=[];
    b.context.fetch=()=>new Promise(resolve=>replies.push(resolve));
    const html=fs.readFileSync(path.join(root,'templates/index.html'),'utf8');
    assert(!/<img[^>]*onerror=/.test(html),'image errors must not call a not-yet-defined inline handler');
    b.run(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
    const img=b.element('video');
    assert.match(img.src,/video_feed/);
    b.run('handleDisconnect()');
    assert.equal(replies.length,1);
    b.hide();assert.equal(img.src,'');
    b.run('handleDisconnect()');assert.equal(replies.length,1);
    b.show();assert.match(img.src,/video_feed/);
    const writes=img.writes.length;
    replies[0]({ok:true,json:async()=>({current:1,max:20,full:false})});
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(img.writes.length,writes,'old polling response restarted a healthy stream');
});

for(const page of ['scope.html','scope2.html']) {
    test(`${page}: fresh preview/luma on return, no stale retry`,()=>{
        const b=browser();
        b.run(fs.readFileSync(path.join(root,'static/scope_renderer.js'),'utf8'));
        const html=fs.readFileSync(path.join(root,'templates',page),'utf8');
        const start=html.indexOf('    const img = document.getElementById("scope-img");');
        const end=html.indexOf('\n    startPreview();',start)+'\n    startPreview();'.length;
        assert(start>0&&end>start);
        b.run(html.slice(start,end));
        const img=b.element('scope-img'),luma=b.element('luma-src');
        assert.match(img.src,/scope\/stream/);
        b.run('localRunning=true;lumImg.src="old-luma"');
        img.onerror();img.onerror();
        assert.equal(b.timers.size,1,'preview errors accumulated retries');
        b.run('reconnectLuma("test");reconnectLuma("test")');
        assert.equal(b.timers.size,2,'luma errors accumulated retries');
        b.hide();assert.equal(img.src,'');assert.equal(b.timers.size,0);
        b.run('startPreview();reconnectLuma("hidden")');
        assert.equal(img.src,'');assert.equal(b.timers.size,0);
        b.show();assert.match(img.src,/scope\/stream/);
        assert.match(luma.src,/scope\/luma/);
        assert.notEqual(luma.src,'old-luma');
        assert.equal(b.run('localRunning'),true,'visibility changed audio ownership');
        b.run('reconnectLuma("test");stopLumaLoop()');
        assert.equal(b.timers.size,0,'stopping left a pending luma reconnect');
    });
}
