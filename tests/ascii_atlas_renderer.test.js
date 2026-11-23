'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const baseAppearance=require('../static/ascii_appearance.js');
// Neutral geometry fixture; the production defaults are pinned by controller tests.
const appearance={...baseAppearance,defaults:{...baseAppearance.defaults,fontWeight:400,fontStroke:0,lineSpacing:1}};
let created=0,mode='hardware',software=false,compile=true,maxTexture=8192;
class Native {
    constructor(){this.calls=[];}
    save(){} restore(){} beginPath(){} rect(){} clip(){} setTransform(){}
    fillRect(...args){this.calls.push(['background',...args]);}
    fillText(...args){this.calls.push(['fill',...args]);}
    strokeText(...args){this.calls.push(['stroke',...args]);}
}
function gpu(){
    const calls=[],state={lost:false,uploads:[],draws:0,calls};
    const values={NO_ERROR:0,MAX_TEXTURE_SIZE:'maxTexture',MAX_RENDERBUFFER_SIZE:'maxBuffer'};
    return new Proxy(state,{get(object,key){
        if(key in object)return object[key]; if(key in values)return values[key];
        if(key==='getParameter')return name=>name==='maxTexture'?maxTexture:name==='maxBuffer'?16384:software?'SwiftShader':'Hardware GPU';
        if(key==='getExtension')return name=>name==='WEBGL_debug_renderer_info'?{UNMASKED_RENDERER_WEBGL:'renderer'}:
            name==='WEBGL_lose_context'?{loseContext(){state.lost=true;}}:null;
        if(key==='isContextLost')return ()=>state.lost;
        if(key==='getShaderParameter')return ()=>compile;
        if(key==='getProgramParameter')return ()=>true;
        if(key==='getError')return ()=>0;
        if(key==='getUniformLocation')return (program,name)=>name;
        if(key==='getShaderInfoLog')return ()=>'test shader error';
        if(key==='bufferSubData')return (target,offset,data)=>{state.uploads.push(Array.from(data));};
        if(key==='drawArraysInstanced')return (...args)=>{state.draws++;state.lastDraw=args;};
        if(key.startsWith('create'))return ()=>({id:++created});
        if(/^[A-Z_0-9]+$/.test(key))return key;
        return (...args)=>calls.push([key,...args]);
    }});
}
class Element {
    constructor(tag='div'){created++;this.tag=tag;this.children=[];this.style={};this.listeners={};this._text='';}
    appendChild(node){this.children.push(node);node.parent=this;return node;}
    remove(){if(this.parent)this.parent.children.splice(this.parent.children.indexOf(this),1);}
    setAttribute(){}
    addEventListener(name,fn){this.listeners[name]=fn;}
    removeEventListener(name){delete this.listeners[name];}
    get offsetHeight(){return 14;}
    getBoundingClientRect(){return {width:this._text.length*10,top:this.tag==='span'&&!this._text?12:0};}
    set textContent(text){this._text=text;} get textContent(){return this._text;}
    getContext(type){
        if(type==='2d')return this.native || (this.native=new Native());
        if(type==='webgl2')return mode==='none'||mode==='canvas'?null:(this.gpu || (this.gpu=gpu()));
    }
}
const document={body:new Element(),createElement:tag=>new Element(tag)};
const window={devicePixelRatio:1,AsciiAppearance:appearance};
vm.runInNewContext(fs.readFileSync('static/ascii_atlas_renderer.js','utf8'),{window,document});
const options=()=>({cols:3,rows:2,fontFamily:'mono',fontSize:12,lineHeight:1.2,
    letterSpacing:1,fontWeight:400,theme:{background:'#000000'}});
const frame={width:9,height:6,glyphs:new Array(54).fill('M'),colours:new Uint32Array(54).fill(0x123456)};
frame.glyphs[13]='i';frame.colours[13]=0xabcdef;
const original=Array.from(frame.colours);
const host=new Element();
const renderer=new window.AsciiAtlasRenderer(options());renderer.open(host);
assert.equal(renderer.backend,'atlas');
renderer.fitViewport(60,80,appearance.defaults);
assert.equal(renderer.stats.atlasBuilds,1);
assert.equal(renderer.canvas.width,60);assert.equal(renderer.canvas.height,80);
renderer.renderGrid(frame,3,2,null);
assert.deepEqual(Array.from(renderer.mapping),[10,13,16,37,40,43]);
assert.deepEqual(renderer.gl.uploads[0],[45,73,45,45,45,45]); // M / i atlas ids.
assert.deepEqual(renderer.gl.uploads[1].slice(0,6),[0x12,0x34,0x56,0xab,0xcd,0xef]);
assert.equal(renderer.gl.lastDraw[3],6);
assert.equal(renderer.stats.uploadedBytes,6*7);
const stable=created;
for(let i=0;i<40;i++)renderer.renderGrid(frame,3,2,null);
assert.equal(created,stable);assert.equal(renderer.stats.atlasBuilds,1);
assert.equal(renderer.gl.draws,41);
let transforms=0;renderer.renderGrid(frame,3,2,()=>{transforms++;return 0xfedcba;});
assert.equal(transforms,2);assert.deepEqual(renderer.gl.uploads.at(-1).slice(0,3),[0xfe,0xdc,0xba]);
assert.deepEqual(Array.from(frame.colours),original);
renderer.fitViewport(60,80,{...appearance.defaults,brightness:1.2,background:'#ff0000'});
assert.equal(renderer.stats.atlasBuilds,1); // Tone/background never rebuild glyphs.
renderer.options.fontWeight=700;
renderer.fitViewport(60,80,{...appearance.defaults,fontWeight:700,fontStroke:.3,lineSpacing:.75});
assert.equal(renderer.stats.atlasBuilds,2);
renderer.resize(6,4);window.devicePixelRatio=2;
renderer.fitViewport(60,80,appearance.defaults);
assert.equal(renderer.canvas.width,120);assert.equal(renderer.canvas.height,160);
assert.equal(renderer.glyphs.length,24);assert.equal(renderer.stats.atlasBuilds,3);
renderer.renderGrid(frame,6,4,null);
const beforeClear=renderer.gl.calls.filter(c=>c[0]==='clear').length;
const unicode={...frame,glyphs:new Array(54).fill('é')};
assert.throws(()=>renderer.renderGrid(unicode,6,4,null),/fallback/);
assert.equal(renderer.gl.calls.filter(c=>c[0]==='clear').length,beforeClear);
let losses=0;renderer.options.onFailure=()=>losses++;
renderer.canvas.listeners.webglcontextlost({preventDefault(){}});assert.equal(losses,1);
renderer.gl.lost=true;assert.throws(()=>renderer.renderGrid(frame,6,4,null),/lost/);
renderer.dispose();renderer.dispose();assert.equal(host.children.length,0);
assert(renderer.gl.calls.some(c=>c[0]==='deleteTexture'));

// Software-GPU detection selects native Canvas instead of expensive emulation.
software=true;window.devicePixelRatio=1;
const native=new window.AsciiAtlasRenderer(options());native.open(host);
assert.equal(native.backend,'canvas');native.fitViewport(60,80,appearance.defaults);
native.renderGrid(frame,3,2,null);native.renderGrid(frame,3,2,null);
assert.equal(native.stats.atlasBuilds,0);assert.equal(native.stats.textCalls,12);
assert.equal(native.context.calls.filter(c=>c[0]==='fill').length,12);
assert.equal(native.stats.fontBuilds,1);
native.fitViewport(60,80,{...appearance.defaults,fontStroke:.3});native.renderGrid(frame,3,2,null);
assert.equal(native.context.calls.filter(c=>c[0]==='stroke').length,6);
let nativeLosses=0;native.options.onFailure=()=>nativeLosses++;
native.canvas.listeners.contextlost({preventDefault(){}});assert.equal(nativeLosses,1);
native.dispose();assert.equal(host.children.length,0);
const forced=new window.AsciiAtlasRenderer({...options(),allowSoftwareGL:true});forced.open(host);
assert.equal(forced.backend,'atlas');forced.dispose();

mode='canvas';const noGL=new window.AsciiAtlasRenderer(options());noGL.open(host);
assert.equal(noGL.backend,'canvas');noGL.dispose();
mode='hardware';software=false;compile=false;
const bad=new window.AsciiAtlasRenderer(options());assert.throws(()=>bad.open(host),/shader/);bad.dispose();
assert.equal(host.children.length,0);compile=true;
const budget=new window.AsciiAtlasRenderer(options());budget.open(host);
assert.throws(()=>budget.fitViewport(100000,800,appearance.defaults),/budget/);
maxTexture=4;assert.throws(()=>budget.fitViewport(60,80,appearance.defaults),/budget/);
budget.dispose();assert.equal(document.body.children.length,0);
console.log('Atlas batching, exact cells/RGB, cache budgets, software Canvas, controls, loss and disposal: passed');
