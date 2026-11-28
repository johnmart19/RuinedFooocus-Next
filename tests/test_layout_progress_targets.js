/* Execute the pinned native tracker and AppTree snapshot methods with inert UI adapters. */
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assets = process.argv[2];
const contract = {
  "6.31.0": {
    tracker: "statustracker-DQF-KcRT.js",
    core: "core-Bdx9080F.js",
    blocks: "Blocks-D1GWYZdr.js",
    editor: "Index-CsUOoOcz.js",
    editorBase: "Fe",
    dataStart: "async function j(){if(!S(te))try{return await k()}",
    dataEnd: "function ee(e){",
  },
  "6.32.0": {
    tracker: "statustracker-s2_3vshN.js",
    core: "core-BXD5setz.js",
    blocks: "Blocks-WDOEXx6W.js",
    editor: "Index-CQwd_G3b.js",
    editorBase: "Pe",
    dataStart: "async function j(){if(!S(te))try{return await A()}",
    dataEnd: "function M(e){",
  },
}[process.argv[3]];
if (!contract) throw Error("Unreviewed native Gradio fixture version: " + process.argv[3]);
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const configurations = Array.isArray(request) ? request : [];
const editorProjection = request.editorProjection;
const native = (name) => fs.readFileSync(path.join(assets, name), "utf8");
const tracker = native(contract.tracker);
const core = native(contract.core);
const blocks = native(contract.blocks);
function between(source, start, end) {
  const first = source.indexOf(start),
    last = source.indexOf(end, first);
  if (first < 0 || last < 0) throw Error("Pinned native method shape changed: " + start);
  return source.slice(first, last);
}
const nativeTracker = between(tracker, "var $=class{current={}", "},Ge=") + "};";
const update = between(core, "async update_state(e,t,n=!0){", "async update_visibility(");
const visibility = between(core, "async update_visibility(e,t){", "async#m(");
const snapshots = between(core, "async#m(e,t){", "#g(e,t){");
const broadcast = between(
  blocks,
  "async update_loading_stati_state(){",
  "dispatch_state_change_events(",
);
let editorMethods = "";
if (editorProjection) {
  const editor = native(contract.editor);
  editorMethods = [
    between(editor, `class o extends ${contract.editorBase}{async get_data(){`, "let s=new o(i,"),
    between(editor, "async function m(){if(S(l)){", "let h=f(!1)"),
    between(editor, contract.dataStart, contract.dataEnd),
  ].join("\n");
}
const { File, Blob } = require("node:buffer");
const scope = { performance, configurations, editorProjection, File, Blob };
vm.createContext(scope);
const program = `
${nativeTracker}
let reads=0;
const Te=['value'];
const Z=(node,id)=>node.id===id?node:node.children.map(child=>Z(child,id)).find(Boolean);
const ke=(node,values)=>Object.assign(node.props.props,values);
const oe=(a,b)=>a===b;
const C=()=>Promise.resolve();
class NativeTreeMethods {
  #c=new Map(); #d=new Set(); #l=new Map(); #s=new Map(); #u(){} #g(){} #r={};
  root={id:-1,children:[],props:{props:{},shared_props:{}}};
  constructor(ids,children=true) {
    for(const id of ids) {
      const node={id,type:'group',children:[],component:true,props:{props:{},shared_props:{}}};
      this.root.children.push(node);
      this.#s.set(id,()=>({value:0}));
      for(let n=0;children&&n<2;n++) {
        const child={id:100000+id*2+n,type:'textbox',children:[],props:{props:{},shared_props:{}}};
        node.children.push(child);
        this.#s.set(child.id,()=>{reads++;return {value:0};});
      }
    }
  }
  setGetter(id,getter){this.#s.set(id,getter);}
  ${update}
  ${visibility}
  ${snapshots}
}
class NativeBroadcast {
  constructor(status,tree){this.loading_stati=status;this.update_state_cb=tree.update_state.bind(tree);}
  ${broadcast}
}
(async()=>{
  // Positive control: full work still retains its actual layout progress and the
  // native completed entry is replayed by a later unrelated pending/complete.
  const status=new $, tree=new NativeTreeMethods([1]), dispatcher=new NativeBroadcast(status,tree);
  status.register(1,[1],[],'full');status.register(2,[2],[],'hidden');
  status.update({fn_index:1,status:'pending'});status.update({fn_index:1,status:'complete'});
  for(const stage of ['pending','complete']) {
    status.update({fn_index:2,status:stage});await dispatcher.update_loading_stati_state();
  }
  const positiveControlReads=reads;
  const allOutputs=configurations.flatMap(row=>row.outputs);
  const metadata=new $, metadataTree=new NativeTreeMethods(allOutputs);
  const metadataDispatcher=new NativeBroadcast(metadata,metadataTree);
  let fullLeafProgressIds=0;
  configurations.forEach((row,id)=>{
    const targets=row.show_progress_on||row.outputs;
    metadata.register(id,targets,[],row.show_progress);
    metadata.update({fn_index:id,status:'pending'});metadata.update({fn_index:id,status:'complete'});
    if(row.show_progress==='full')fullLeafProgressIds+=targets.length;
  });
  // The Python fixture models every output as a node, so only descendants of
  // actual configured layout targets count toward this regression.
  const layoutIds=new Set(configurations.flatMap(row=>row.layout_ids));
  const retainedLayoutIds=Object.keys(metadata.current).map(Number).filter(id=>layoutIds.has(id));
  // Test only the retained layout IDs through the exact native snapshot path;
  // scalar output progress remains registered and is checked separately.
  const retained=new $;retained.current=Object.fromEntries(retainedLayoutIds.map(id=>[id,metadata.current[id]]));
  retained.register(99999,[99999],[],'hidden');
  const retainedDispatcher=new NativeBroadcast(retained,metadataTree);
  reads=0;
  for(const stage of ['pending','complete']) {
    retained.update({fn_index:99999,status:stage});await retainedDispatcher.update_loading_stati_state();
  }
  let imageEditor;
  if(editorProjection){
    let exports=0,uploads=0;
    const getEditor=(()=>{
      class ${contract.editorBase}{async get_data(){return {value:null};}}
      let l=null,te=true;
      const S=value=>value,L=()=>{},R=()=>{},E=Boolean,D=Boolean,ie=async files=>files,ae=ie;
      const C={get_blobs:async()=>{exports++;return {
        background:new Blob(['background']),layers:[new Blob(['layer'])],composite:new Blob(['composite'])
      };}};
      const n={root:'/inert',upload:async files=>{uploads++;return files.map(file=>({path:file.name}));}};
      ${editorMethods}
      const c={get_data:j};
      const controller=new o();
      return ()=>controller.get_data();
    })();
    async function replay(targets){
      exports=0;uploads=0;
      const id=editorProjection.editorId,status=new $,tree=new NativeTreeMethods([id],false);
      tree.setGetter(id,getEditor);
      const dispatcher=new NativeBroadcast(status,tree);
      status.register(1,targets,[],'hidden');status.register(2,[99999],[],'hidden');
      status.update({fn_index:1,status:'pending'});status.update({fn_index:1,status:'complete'});
      for(const stage of ['pending','complete']){
        status.update({fn_index:2,status:stage});await dispatcher.update_loading_stati_state();
      }
      return {exports,uploads};
    }
    imageEditor={before:await replay([editorProjection.editorId]),after:await replay(editorProjection.show_progress_on||editorProjection.outputs)};
  }
  globalThis.result={positiveControlReads,metadataLayoutReads:reads,metadataLayoutIds:retainedLayoutIds,fullLeafProgressIds,imageEditor};
})()`;
vm.runInContext(program, scope)
  .then(() => process.stdout.write(JSON.stringify(scope.result)))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
