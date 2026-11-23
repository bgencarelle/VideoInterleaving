'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
let created=0,textWrites=0,styleWrites=0;
class Element {
    constructor(){created++;this.children=[];this.parent=null;this._text='';
        this.style=new Proxy({}, {set(object,key,value){styleWrites++;object[key]=value;return true;}});}
    appendChild(child){this.children.push(child);child.parent=this;return child;}
    remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null;}}
    replaceChildren(){for(const child of this.children)child.parent=null;this.children=[];}
    setAttribute(){}
    set textContent(value){textWrites++;this._text=value;}
    get textContent(){return this._text || this.children.map(child=>child.textContent).join('');}
    get offsetWidth(){return [...this.textContent].reduce((sum,c)=>sum+(c==='W'||c==='M'?10:7),0);}
    get offsetHeight(){return 14;}
    getBoundingClientRect(){return {width:this.offsetWidth,height:this.offsetHeight};}
}
const document={createElement:()=>new Element(),body:new Element()};
const window={devicePixelRatio:1};
vm.runInNewContext(fs.readFileSync('static/ascii_grid_renderer.js','utf8'),{window,document});
const renderer=new window.AsciiGridRenderer({cols:6,rows:3,fontFamily:'serif',fontSize:12,
    lineHeight:1.2,letterSpacing:1,fontWeight:400,theme:{background:'#000000'}});
const host=new Element();renderer.open(host);
function frame(glyph,colour,width=6,height=3){return {width,height,glyphs:new Array(width*height).fill(glyph),
    colours:new Uint32Array(width*height).fill(colour)};}
renderer.renderGrid(frame('M',0x123456),6,3,null);
renderer.renderGrid(frame('W',0x234567),6,3,null);
const count=created;
for(let i=0;i<100;i++)renderer.renderGrid(frame(i%2?'M':'W',i%2?0x123456:0x234567),6,3,null);
assert.equal(created,count); // No replacement spans or measurements after warmup.
assert.equal(renderer.states.length,3);
assert(renderer.states.every(row=>row.nodes.length===1));
assert.equal(renderer.widths.size,2);
const writes=[textWrites,styleWrites];
renderer.renderGrid(frame('M',0x123456),6,3,null);
assert.deepEqual([textWrites,styleWrites],writes); // Unchanged picture does not touch DOM.
assert.equal(renderer.states[0].nodes[0].textContent,'MMMMMM');
assert.equal(renderer.states[0].nodes[0].style.color,'#123456');
assert.equal(renderer.states[0].nodes[0].style.letterSpacing,'1px');
const mixed=frame('M',0x123456);
mixed.glyphs[1]='.';mixed.colours[2]=0xabcdef;
renderer.renderGrid(mixed,6,3,null);
assert.equal(renderer.states[0].nodes.length,4);
assert.equal(renderer.states[0].nodes.map(n=>n.textContent).join(''),'M.MMMM');
assert.equal(mixed.colours[2],0xabcdef);
let transforms=0;
renderer.renderGrid(mixed,6,3,value=>{transforms++;return 0;});
assert.equal(transforms,2);
assert(renderer.states.every(row=>row.nodes.every(n=>n.style.color==='#000000')));
renderer.resize(3,2);renderer.renderGrid(frame('M',0xffffff,6,3),3,2,null);
assert.equal(renderer.states.length,2);assert.equal(renderer.mapping.length,6);
assert(renderer.states.every(row=>row.nodes.map(n=>n.textContent).join('')==='MMM'));
const oldVersion=renderer.layoutVersion;
window.devicePixelRatio=2;renderer.renderGrid(frame('M',0xffffff,6,3),3,2,null);
assert.equal(renderer.dpr,2);assert.equal(renderer.layoutVersion,oldVersion+1);
renderer.dispose();assert.equal(host.children.length,0);assert.equal(document.body.children.length,0);
const mono=new window.AsciiGridRenderer({cols:6,rows:3,fontFamily:'mono',fontSize:12,
    monospaced:true,lineHeight:1.2,letterSpacing:1,fontWeight:400,theme:{background:'#000000'}});
mono.open(host);
const monoFrame=frame('M',0x123456);monoFrame.glyphs[1]='.';monoFrame.glyphs[2]='i';
mono.renderGrid(monoFrame,6,3,null);
assert.equal(mono.widths.size,0);assert.equal(mono.measure,undefined);
assert.equal(mono.states[0].nodes.length,1); // Different glyph widths NEVER split colour runs.
assert.equal(mono.states[0].nodes[0].textContent,'M.iMMM');
assert.equal(mono.states[0].nodes[0].style.letterSpacing,undefined);
monoFrame.colours[2]=0xabcdef;mono.renderGrid(monoFrame,6,3,null);
assert.equal(mono.states[0].nodes.length,3);
mono.options.fontWeight=900;mono.renderGrid(monoFrame,6,3,null);
assert.equal(mono.container.style.fontWeight,'900');assert.equal(mono.widths.size,0);
mono.renderGrid(monoFrame,6,3,()=>0);
assert(mono.states.every(row=>row.nodes.length===1)); // One tint means exactly one run per row.
mono.dispose();assert.equal(host.children.length,0);assert.equal(document.body.children.length,0);
console.log('Persistent grid spans, unchanged DOM, bounds, run colours, resize and disposal: passed');
