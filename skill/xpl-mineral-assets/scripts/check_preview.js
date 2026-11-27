/* Execute preview logic with a minimal DOM/canvas test double. This is not browser layout testing. */
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('assert');
const reference=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const results=[];
for(const sample of reference){
  const html=fs.readFileSync(sample.preview,'utf8');
  const code=html.match(/<script>([\s\S]*)<\/script>/)[1];
  let latest,scheduled=[];
  const nodes={};
  function node(id){if(!nodes[id])nodes[id]={value:id==='mode'?'optics':'0',handlers:{},addEventListener(type,fn){this.handlers[type]=fn},click(){}};return nodes[id]}
  node('view').getContext=()=>({putImageData(im){latest=im.data}});
  const context={document:{getElementById:node,createElement:()=>({click(){}})},window:{},
    atob:s=>Buffer.from(s,'base64').toString('binary'),
    ImageData:class{constructor(data,w,h){this.data=data}},
    requestAnimationFrame:fn=>scheduled.push(fn)};
  vm.createContext(context);vm.runInContext(code,context,{timeout:30000});
  let maxError=0;
  for(const frame of sample.frames){
    node('angle').value=String(frame.angle);node('angle').handlers.input();
    for(const p of frame.pixels){for(let k=0;k<4;k++)maxError=Math.max(maxError,Math.abs(latest[p.index*4+k]-p.rgba[k]))}
  }
  assert(maxError<=1,'Preview/Python mismatch: '+maxError);
  const optics=Buffer.from(latest);node('mode').value='domains';node('mode').handlers.change();
  assert(!optics.equals(Buffer.from(latest)),'Domain switch failed');
  node('mode').value='overlay';node('mode').handlers.change();
  node('mode').value='mask';node('mode').handlers.change();
  node('play').onclick();assert(scheduled.length===1,'Play callback missing');
  const old=Number(node('angle').value);scheduled.shift()(1000);
  assert(Number(node('angle').value)!==old,'Play did not advance angle');node('play').onclick();
  results.push({asset:sample.asset,rgba_samples:sample.frames.reduce((n,f)=>n+f.pixels.length,0),max_error_8bit:maxError,
    mode_handlers:true,animation_handler:true,browser_layout_tested:false});
}
console.log(JSON.stringify({ok:true,method:'Node VM with DOM/canvas test double',assets:results},null,2));
