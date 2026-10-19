const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {test}=require('node:test');

const turn=()=>new Promise(resolve=>setImmediate(resolve));
function harness(){
    let counter=0,held=null,holdNext=false;
    const blobs=new Map(),revoked=[],errors=[],sources=[];
    const image={src:'',removeAttribute(){this.src='';}};
    class FakeImage{
        set src(url){
            sources.push(url);
            const complete=()=>this.onload();
            if(holdNext){holdNext=false;held=complete;}else queueMicrotask(complete);
        }
    }
    const context=vm.createContext({window:{},Blob,Uint8Array,TextDecoder,AbortController,Image:FakeImage,
        URL:{createObjectURL(blob){const url='blob:'+ ++counter;blobs.set(url,blob);return url;},
             revokeObjectURL(url){revoked.push(url);}},
        fetch:async(_url,{signal})=>{
            let controller;
            const body=new ReadableStream({start(c){controller=c;}});
            signal.addEventListener('abort',()=>controller.error(Error('aborted')));
            context.push=bytes=>controller.enqueue(bytes);
            return {ok:true,body};
        }});
    vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/live_image_stream.js'),'utf8'),context);
    const stream=new context.window.LiveImageStream(image,{onError:e=>errors.push(e.message)});
    return {context,image,stream,blobs,revoked,errors,sources,
        hold(){holdNext=true;},release(){held();},
        async value(){return image.src?blobs.get(image.src).text():null;}};
}
function part(value,first=false){
    return new TextEncoder().encode((first?'--frame\r\n':'\r\n--frame\r\n')+
        'Content-Type: image/jpeg\r\nContent-Length: '+value.length+'\r\n\r\n'+value);
}

test('complete image presents without a subsequent frame or boundary',async()=>{
    const h=harness();h.stream.start('/stream');await turn();
    const bytes=part('first',true);
    for(const byte of bytes)h.context.push(Uint8Array.of(byte));
    await turn();assert.equal(await h.value(),'first');
    assert.deepEqual(h.errors,[]);h.stream.stop();await turn();assert.deepEqual(h.errors,[]);
});

test('one in-flight image and only newest pending image are decoded',async()=>{
    const h=harness();h.hold();h.stream.start('/stream');await turn();
    h.context.push(part('one',true));await turn();
    h.context.push(part('two'));h.context.push(part('newest'));await turn();
    assert.equal(h.sources.length,1);
    h.release();await turn();await turn();
    assert.equal(h.sources.length,2);assert.equal(await h.value(),'newest');
    assert.equal(h.revoked.length,1);h.stream.stop();
});

test('old decode cannot block or overwrite a reconnected stream',async()=>{
    const h=harness();h.hold();h.stream.start('/old');await turn();
    h.context.push(part('old',true));await turn();
    h.stream.stop();h.stream.start('/current');await turn();
    h.context.push(part('current',true));await turn();
    assert.equal(await h.value(),'current');h.release();await turn();
    assert.equal(await h.value(),'current');assert.deepEqual(h.errors,[]);h.stream.stop();
});

test('oversized frame is rejected before buffering its payload',async()=>{
    const h=harness();h.stream.start('/stream');await turn();
    h.context.push(new TextEncoder().encode('--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 999999999\r\n\r\n'));
    await turn();assert.equal(h.errors.length,1);assert.equal(h.sources.length,0);h.stream.stop();
});
