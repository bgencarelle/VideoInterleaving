/* Read-only RGB text without per-frame DOM cells/layout or GPU readback.
 * Hardware: reusable alpha atlas and one instanced draw. Software GPUs:
 * native Canvas text, avoiding costly WebGL emulation. DOM remains a fallback.
 */
(function (root) {
    'use strict';
    const ATLAS_COLS = 16, GLYPHS = 95;
    const VERTEX = `#version 300 es
        precision highp float;
        layout(location=0) in vec2 corner;
        layout(location=1) in float glyph;
        layout(location=2) in vec3 colour;
        uniform vec2 viewportSize;
        uniform vec2 cellSize;
        uniform float rowStep;
        uniform int columns;
        uniform float padding;
        uniform vec2 tileSize;
        uniform vec2 atlasSize;
        out vec2 uv;
        out vec3 tint;
        void main() {
            vec2 origin = vec2(float(gl_InstanceID % columns)*cellSize.x,
                               float(gl_InstanceID / columns)*rowStep);
            vec2 quad = vec2(cellSize.x+2.0*padding, cellSize.y);
            vec2 pixel = origin + corner*quad - vec2(padding,0.0);
            gl_Position = vec4(pixel/viewportSize*vec2(2.0,-2.0)+vec2(-1.0,1.0),0.0,1.0);
            vec2 tile = vec2(mod(glyph,16.0),floor(glyph/16.0));
            uv = (tile*tileSize+vec2(0.0,1.0)+corner*quad)/atlasSize;
            tint = colour;
        }`;
    const FRAGMENT = `#version 300 es
        precision highp float;
        uniform sampler2D atlas;
        in vec2 uv;
        in vec3 tint;
        out vec4 result;
        void main() {
            float alpha = texture(atlas,uv).a;
            if (alpha == 0.0) discard;
            result = vec4(tint*alpha,alpha);
        }`;

    function shader(gl, type, source) {
        const value = gl.createShader(type);
        gl.shaderSource(value,source); gl.compileShader(value);
        if (!gl.getShaderParameter(value,gl.COMPILE_STATUS)) {
            const message = gl.getShaderInfoLog(value); gl.deleteShader(value);
            throw new Error('ASCII atlas shader: '+message);
        }
        return value;
    }

    class AsciiAtlasRenderer {
        constructor(options) {
            this.options=options; this.cols=options.cols; this.rows=options.rows;
            this.stats={draws:0,atlasBuilds:0,fontBuilds:0,uploadedBytes:0,textCalls:0};
            this.mapping=null; this.atlasKey=null; this.layoutVersion=0;
            this.resources=[]; this.disposed=false;
        }
        open(host) {
            this.host=host;
            this.canvas=document.createElement('canvas');
            this.canvas.className='ascii-atlas-canvas';
            this.canvas.style.cssText='display:block;width:100%;height:100%';
            this.gl=this.canvas.getContext('webgl2', {alpha:false,antialias:false,
                depth:false,stencil:false,premultipliedAlpha:false,preserveDrawingBuffer:false,
                failIfMajorPerformanceCaveat:!this.options.allowSoftwareGL});
            if(this.gl && !this.options.allowSoftwareGL) {
                const info=this.gl.getExtension('WEBGL_debug_renderer_info');
                const name=info ? this.gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : '';
                if(/swiftshader|llvmpipe|softpipe|software|basic render driver|\bwarp\b/i.test(name)) {
                    const loss=this.gl.getExtension('WEBGL_lose_context');
                    if(loss) loss.loseContext();
                    this.gl=null;
                }
            }
            this.backend=this.gl ? 'atlas' : 'canvas';
            if(!this.gl) {
                this.canvas=document.createElement('canvas'); this.canvas.className='ascii-atlas-canvas';
                this.canvas.style.cssText='display:block;width:100%;height:100%';
                this.context=this.canvas.getContext('2d',{alpha:false});
                if(!this.context) throw new Error('Canvas rendering unavailable');
            }
            const gl=this.gl;
            this.lost = event => {
                event.preventDefault();
                if (!this.disposed && this.options.onFailure) this.options.onFailure(new Error('Canvas rendering context lost'));
            };
            this.lossEvent=gl ? 'webglcontextlost' : 'contextlost';
            this.canvas.addEventListener(this.lossEvent,this.lost);
            this.element=document.createElement('div');
            this.element.className='terminal xterm ascii-atlas-grid';
            this.element.setAttribute('role','img');
            this.element.setAttribute('aria-label','Live ASCII animation');
            this.screen=document.createElement('div'); this.screen.className='xterm-screen';
            this.screen.appendChild(this.canvas); this.element.appendChild(this.screen); host.appendChild(this.element);
            // Match the DOM renderer's original, unscaled font/row metrics once.
            const probe=document.createElement('span');
            probe.style.cssText=`position:absolute;visibility:hidden;white-space:pre;font:400 ${this.options.fontSize}px ${this.options.fontFamily};font-kerning:none`;
            probe.textContent='W'.repeat(32); this.element.appendChild(probe);
            this.charWidth=probe.getBoundingClientRect().width/32;
            this.charHeight=Math.ceil(probe.offsetHeight); probe.remove();
            if (!(this.charWidth>0 && this.charHeight>0)) throw new Error('Invalid atlas font metrics');
            if(!gl) { this.resize(this.cols,this.rows); return; }
            const vertex=shader(gl,gl.VERTEX_SHADER,VERTEX);
            let fragment;
            try { fragment=shader(gl,gl.FRAGMENT_SHADER,FRAGMENT); }
            catch(error) { gl.deleteShader(vertex); throw error; }
            this.program=gl.createProgram(); this.resources.push(['Program',this.program]);
            gl.attachShader(this.program,vertex); gl.attachShader(this.program,fragment); gl.linkProgram(this.program);
            gl.deleteShader(vertex); gl.deleteShader(fragment);
            if (!gl.getProgramParameter(this.program,gl.LINK_STATUS)) throw new Error('ASCII atlas link: '+gl.getProgramInfoLog(this.program));
            this.vao=gl.createVertexArray(); this.resources.push(['VertexArray',this.vao]); gl.bindVertexArray(this.vao);
            this.corners=gl.createBuffer(); this.resources.push(['Buffer',this.corners]);
            gl.bindBuffer(gl.ARRAY_BUFFER,this.corners);
            gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([0,0,1,0,0,1,1,1]),gl.STATIC_DRAW);
            gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
            this.glyphBuffer=gl.createBuffer(); this.resources.push(['Buffer',this.glyphBuffer]);
            gl.bindBuffer(gl.ARRAY_BUFFER,this.glyphBuffer); gl.enableVertexAttribArray(1);
            gl.vertexAttribPointer(1,1,gl.FLOAT,false,0,0); gl.vertexAttribDivisor(1,1);
            this.colourBuffer=gl.createBuffer(); this.resources.push(['Buffer',this.colourBuffer]);
            gl.bindBuffer(gl.ARRAY_BUFFER,this.colourBuffer); gl.enableVertexAttribArray(2);
            gl.vertexAttribPointer(2,3,gl.UNSIGNED_BYTE,true,0,0); gl.vertexAttribDivisor(2,1);
            this.texture=gl.createTexture(); this.resources.push(['Texture',this.texture]);
            gl.bindTexture(gl.TEXTURE_2D,this.texture);
            gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
            gl.useProgram(this.program);
            this.uniforms={};
            for(const name of ['viewportSize','cellSize','rowStep','columns','padding','tileSize','atlasSize','atlas'])
                this.uniforms[name]=gl.getUniformLocation(this.program,name);
            gl.uniform1i(this.uniforms.atlas,0);
            gl.disable(gl.DEPTH_TEST); gl.disable(gl.DITHER);
            gl.enable(gl.BLEND); gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
            this.resize(this.cols,this.rows);
        }
        resize(cols,rows) {
            this.cols=cols; this.rows=rows; this.mapping=null;
            this.glyphs=new Float32Array(cols*rows);
            this.colours=new Uint8Array(cols*rows*3);
            if(this.gl) {
                const gl=this.gl;
                gl.bindBuffer(gl.ARRAY_BUFFER,this.glyphBuffer); gl.bufferData(gl.ARRAY_BUFFER,this.glyphs.byteLength,gl.DYNAMIC_DRAW);
                gl.bindBuffer(gl.ARRAY_BUFFER,this.colourBuffer); gl.bufferData(gl.ARRAY_BUFFER,this.colours.byteLength,gl.DYNAMIC_DRAW);
            }
            this.atlasKey=null; this.layoutVersion++;
        }
        clearSelection() {}
        hasSelection() { return false; }
        onSelectionChange() {}
        fitViewport(width,height,picture) {
            this.picture=picture;
            const dpr=window.devicePixelRatio || 1;
            const key=[width,height,dpr,this.cols,this.rows,picture.fontWeight,picture.fontStroke,picture.lineSpacing].join('/');
            this.host.style.width=width+'px'; this.host.style.height=height+'px';
            this.host.style.transform=''; this.host.style.overflow='hidden';
            this.screen.style.width=width+'px'; this.screen.style.height=height+'px';
            if(key===this.atlasKey) return;
            this.dpr=dpr;
            const gl=this.gl, pixelW=Math.max(1,Math.round(width*dpr)), pixelH=Math.max(1,Math.round(height*dpr));
            const limit=gl ? gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) : 16384;
            if(pixelW>limit || pixelH>limit || pixelW*pixelH>16*1024*1024) throw new Error('ASCII canvas exceeds GPU budget');
            // Preserve the DOM path's natural geometry and row-spacing fit.
            const naturalW=Math.round((this.charWidth*dpr+Math.round(this.options.letterSpacing || 0))*this.cols/dpr);
            const naturalH=Math.round(Math.floor(Math.ceil(this.charHeight*dpr)*Math.max(1,this.options.lineHeight))*this.rows/dpr);
            const cellH=naturalH/this.rows;
            const layout=root.AsciiAppearance.rowLayout(this.rows,naturalH,picture.lineSpacing);
            this.scaleX=pixelW/naturalW; this.scaleY=pixelH/layout.height;
            this.naturalW=naturalW; this.naturalCellH=cellH;
            this.cellW=pixelW/this.cols; this.cellH=cellH*this.scaleY;
            this.rowStep=this.cellH*picture.lineSpacing;
            this.canvas.width=pixelW; this.canvas.height=pixelH;
            this.baseline=this.measureBaseline(cellH);
            this.stats.fontBuilds++;
            if(!gl) {
                const ctx=this.context;
                ctx.font=`${this.options.fontWeight} ${this.options.fontSize}px ${this.options.fontFamily}`;
                ctx.fontKerning='none'; ctx.textBaseline='alphabetic';
                ctx.lineWidth=picture.fontStroke;
                this.atlasKey=key; this.layoutVersion++;
                return;
            }
            gl.viewport(0,0,pixelW,pixelH);
            this.buildAtlas();
            gl.useProgram(this.program);
            gl.uniform2f(this.uniforms.viewportSize,pixelW,pixelH);
            gl.uniform2f(this.uniforms.cellSize,this.cellW,this.cellH);
            gl.uniform1f(this.uniforms.rowStep,this.rowStep);
            gl.uniform1i(this.uniforms.columns,this.cols);
            this.atlasKey=key; this.layoutVersion++;
        }
        measureBaseline(cellH) {
            // One baseline layout read per atlas rebuild; never during frame work.
            const line=document.createElement('div'), marker=document.createElement('span');
            line.style.cssText=`position:absolute;left:-50000px;top:0;visibility:hidden;white-space:pre;font:${this.options.fontWeight} ${this.options.fontSize}px ${this.options.fontFamily};line-height:${cellH}px;font-kerning:none`;
            line.textContent='W'; marker.style.cssText='display:inline-block;width:0;height:0;vertical-align:baseline';
            line.appendChild(marker); document.body.appendChild(line);
            const baseline=marker.getBoundingClientRect().top-line.getBoundingClientRect().top;
            line.remove();
            return baseline;
        }
        buildAtlas() {
            const gl=this.gl;
            const padding=Math.ceil(Math.max(1,this.picture.fontStroke)*Math.max(this.scaleX,this.scaleY))+2;
            const tileW=Math.ceil(this.cellW+2*padding)+1, tileH=Math.ceil(this.cellH)+2;
            const width=ATLAS_COLS*tileW, height=Math.ceil(GLYPHS/ATLAS_COLS)*tileH;
            const limit=gl.getParameter(gl.MAX_TEXTURE_SIZE);
            if(width>limit || height>limit || width*height>8*1024*1024) throw new Error('ASCII atlas exceeds GPU budget');
            const atlas=document.createElement('canvas'); atlas.width=width; atlas.height=height;
            const ctx=atlas.getContext('2d');
            if(!ctx) throw new Error('Glyph atlas canvas unavailable');
            for(let index=0;index<GLYPHS;index++) {
                const x=(index%ATLAS_COLS)*tileW, y=Math.floor(index/ATLAS_COLS)*tileH;
                ctx.save(); ctx.beginPath();
                // The old DOM renderer clips each row, not each glyph column.
                ctx.rect(x,y+1,tileW,this.cellH); ctx.clip();
                ctx.setTransform(this.scaleX,0,0,this.scaleY,x+padding,y+1);
                ctx.font=`${this.options.fontWeight} ${this.options.fontSize}px ${this.options.fontFamily}`;
                ctx.fontKerning='none'; ctx.textBaseline='alphabetic';
                ctx.fillStyle=ctx.strokeStyle='#ffffff';
                if(this.picture.fontStroke>0) {
                    ctx.lineWidth=this.picture.fontStroke;
                    ctx.strokeText(String.fromCharCode(index+32),0,this.baseline);
                }
                ctx.fillText(String.fromCharCode(index+32),0,this.baseline); ctx.restore();
            }
            gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.texture);
            gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);
            gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,atlas);
            if(gl.getError()!==gl.NO_ERROR) throw new Error('ASCII atlas texture allocation failed');
            gl.uniform1f(this.uniforms.padding,padding);
            gl.uniform2f(this.uniforms.tileSize,tileW,tileH);
            gl.uniform2f(this.uniforms.atlasSize,width,height);
            this.stats.atlasBuilds++;
        }
        renderGrid(frame,cols,rows,transform) {
            if(this.disposed || !this.atlasKey) throw new Error('ASCII atlas is not fitted');
            if(cols!==this.cols || rows!==this.rows) throw new Error('ASCII atlas resize needs fitting');
            const gl=this.gl;
            if(gl && gl.isContextLost()) throw new Error('WebGL context lost');
            if(!this.mapping || this.sourceWidth!==frame.width || this.sourceHeight!==frame.height) {
                this.sourceWidth=frame.width; this.sourceHeight=frame.height;
                this.mapping=new Uint16Array(cols*rows);
                for(let y=0;y<rows;y++) for(let x=0;x<cols;x++)
                    this.mapping[y*cols+x]=Math.floor((y+.5)*frame.height/rows)*frame.width+Math.floor((x+.5)*frame.width/cols);
            }
            const adjusted=transform ? new Map() : null;
            for(let cell=0;cell<this.mapping.length;cell++) {
                const source=this.mapping[cell], glyph=frame.glyphs[source];
                const id=glyph.charCodeAt(0)-32;
                // Never replace an unsupported character with a '?' or blank.
                if(glyph.length!==1 || id<0 || id>=GLYPHS) throw new Error('Glyph needs DOM font fallback');
                this.glyphs[cell]=id;
                let colour=frame.colours[source];
                if(transform) {
                    const original=colour;
                    if(adjusted.has(original)) colour=adjusted.get(original);
                    else { colour=transform(original); adjusted.set(original,colour); }
                }
                const offset=cell*3;
                this.colours[offset]=colour>>16; this.colours[offset+1]=(colour>>8)&255; this.colours[offset+2]=colour&255;
            }
            const background=this.options.theme.background;
            if(!gl) { this.drawNative(frame,background); this.stats.draws++; return; }
            gl.clearColor(parseInt(background.slice(1,3),16)/255,parseInt(background.slice(3,5),16)/255,
                parseInt(background.slice(5,7),16)/255,1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.useProgram(this.program); gl.bindVertexArray(this.vao);
            gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.texture);
            gl.bindBuffer(gl.ARRAY_BUFFER,this.glyphBuffer); gl.bufferSubData(gl.ARRAY_BUFFER,0,this.glyphs);
            gl.bindBuffer(gl.ARRAY_BUFFER,this.colourBuffer); gl.bufferSubData(gl.ARRAY_BUFFER,0,this.colours);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,cols*rows);
            this.stats.draws++; this.stats.uploadedBytes+=this.glyphs.byteLength+this.colours.byteLength;
        }
        drawNative(frame,background) {
            const ctx=this.context;
            ctx.setTransform(1,0,0,1,0,0); ctx.fillStyle=background;
            ctx.fillRect(0,0,this.canvas.width,this.canvas.height);
            ctx.setTransform(this.scaleX,0,0,this.scaleY,0,0);
            const cellW=this.naturalW/this.cols, cellH=this.naturalCellH;
            for(let y=0;y<this.rows;y++) {
                const top=y*cellH*this.picture.lineSpacing;
                ctx.save(); ctx.beginPath(); ctx.rect(0,top,this.naturalW,cellH); ctx.clip();
                let previous=-1;
                for(let x=0;x<this.cols;x++) {
                    const index=y*this.cols+x, offset=index*3;
                    const colour=(this.colours[offset]<<16)|(this.colours[offset+1]<<8)|this.colours[offset+2];
                    if(colour!==previous) {
                        ctx.fillStyle=ctx.strokeStyle='#'+colour.toString(16).padStart(6,'0'); previous=colour;
                    }
                    const glyph=frame.glyphs[this.mapping[index]];
                    if(this.picture.fontStroke>0) ctx.strokeText(glyph,x*cellW,top+this.baseline);
                    ctx.fillText(glyph,x*cellW,top+this.baseline); this.stats.textCalls++;
                }
                ctx.restore();
            }
        }
        dispose() {
            if(this.disposed) return;
            this.disposed=true;
            if(this.canvas && this.lost) this.canvas.removeEventListener(this.lossEvent,this.lost);
            if(this.gl) {
                for(const [kind,value] of this.resources) this.gl['delete'+kind](value);
                const loss=this.gl.getExtension('WEBGL_lose_context');
                if(loss && !this.gl.isContextLost()) loss.loseContext();
            }
            if(this.element) this.element.remove();
            this.resources=[]; this.mapping=null; this.glyphs=null; this.colours=null;
        }
    }
    root.AsciiAtlasRenderer=AsciiAtlasRenderer;
})(typeof window === 'object' ? window : globalThis);
