"""Execute the shared card renderer in a bounded DOM fixture without a browser."""

from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ruinedfooocus/ui/assets/adapter_cards.js"
HARNESS = r"""
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8');
const scenario=process.argv[2];
let document;
class Node {
  constructor(tag='div') {
    this.tagName=tag; this.children=[]; this.dataset={}; this.attrs={}; this.listeners={};
    this.classList={toggle(){},add(){},remove(){}}; this.scrollTop=0; this.scrollLeft=0;
    this._text=''; this.sourceChanges=[];
  }
  set textContent(value) { this._text=value; this.children.forEach(node=>node.parent=null); this.children=[]; }
  get textContent() { return this._text+this.children.map(node=>node.textContent).join(''); }
  set src(value) { this.attrs.src=value; this.sourceChanges.push(value); }
  get src() { return this.attrs.src; }
  setAttribute(name,value) { this.attrs[name]=value; }
  getAttribute(name) { return this.attrs[name]??null; }
  addEventListener(name,fn) { this.listeners[name]=fn; }
  append(...nodes) { for(const node of nodes) { node.remove(); node.parent=this; this.children.push(node); } }
  replaceChildren(...nodes) { this.textContent=''; this.append(...nodes); }
  contains(node) { return this.children.some(child=>child===node||child.contains(node)); }
  remove() { if(this.parent) { this.parent.children=this.parent.children.filter(child=>child!==this); this.parent=null; } }
  insertBefore(node,next) { node.remove(); node.parent=this; this.children.splice(next?this.children.indexOf(next):this.children.length,0,node); }
  focus() { document.activeElement=this; }
  matches(selector) {
    if(selector==='img') return this.tagName==='img';
    if(selector==='button[data-adapter]') return this.tagName==='button'&&!!this.dataset.adapter;
    if(selector==='[data-adapter]') return !!this.dataset.adapter;
    return selector.startsWith('.')&&this.className?.split(' ').includes(selector.slice(1));
  }
  querySelectorAll(selector) { return this.children.flatMap(child=>[...(child.matches(selector)?[child]:[]),...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0]||null; }
}
const host=new Node(),heading=new Node(),grid=new Node(),callbacks=[];
heading.className='adapter-card-heading';grid.className='adapter-card-grid';host.append(heading,grid);
document={activeElement:null,createElement:tag=>new Node(tag)};
const localized='Типовий попередній перегляд';
const card=(fields={})=>({key:'one',label:'Example model',kind:'Image model',artworkKey:'owner:sha',
  preview:{url:'category.jpg'},fallback:{url:'category.jpg'},preview_is_default:true,...fields});
const packet=(fields={})=>({heading:'Models',empty:'Empty',selected:'one',defaultPreviewLabel:localized,cards:[card()],...fields});
const props={value:packet()};
const context={props,element:host,document,window:{__rfModelUIRequests:{}},watch:(name,fn)=>callbacks.push(fn),trigger(){}};
vm.runInNewContext(source,context);
const update=value=>{props.value=value;callbacks.forEach(fn=>fn());};
const button=()=>grid.querySelector('button[data-adapter]');
const image=()=>button().querySelector('img');
const badge=()=>button().querySelector('.adapter-card-default');
if(scenario==='default-and-decode') {
  assert.equal(badge().hidden,false);assert.equal(badge().textContent,localized);
  assert.equal(image().alt,'');assert.equal(button().type,'button');
  assert.equal(button().getAttribute('aria-pressed'),'true');
  assert.equal(badge().getAttribute('aria-hidden'),null);
  update(packet({cards:[card({preview:{url:'publisher.jpg'},preview_is_default:false})]}));
  assert.equal(badge().hidden,true);image().onerror();
  assert.equal(image().src,'category.jpg');assert.equal(badge().hidden,false);
  const changes=image().sourceChanges.length;image().onerror();
  assert.equal(image().sourceChanges.length,changes); // A failed fallback never loops.
  update(packet({cards:[card({preview:{url:'flagged.jpg'},preview_is_default:true})]}));
  assert.equal(badge().hidden,false); // Explicit helper flag also covers a distinct cache URL.
  update(packet({cards:[card({preview:null,preview_is_default:false})]}));
  assert.equal(image().src,'category.jpg');assert.equal(badge().hidden,false);
} else if(scenario==='incremental-patches') {
  const originalButton=button(),originalImage=image();grid.scrollTop=42;grid.scrollLeft=4;originalButton.focus();
  context.window.__rfModelUIRequests={'acquisition-view':2};
  update({requestIdentity:{channel:'acquisition-view',version:1},previewPatch:[{key:'one',artworkKey:'owner:sha',preview:{url:'stale.jpg'}}]});
  assert.equal(image().src,'category.jpg');assert.equal(badge().hidden,false);
  update({previewPatch:[{key:'one',artworkKey:'wrong owner',preview:{url:'wrong.jpg'}}]});
  assert.equal(image().src,'category.jpg');assert.equal(badge().hidden,false);
  update({previewPatch:[{key:'one',artworkKey:'owner:sha',preview:{url:'publisher.jpg'}}]});
  assert.equal(image(),originalImage);assert.equal(button(),originalButton);assert.equal(badge().hidden,true);
  assert.equal(document.activeElement,originalButton);assert.equal(grid.scrollTop,42);assert.equal(grid.scrollLeft,4);
  image().onerror();assert.equal(badge().hidden,false);
  update({previewPatch:[{key:'one',artworkKey:'owner:sha',preview:{url:'new-publisher.jpg'}}]});
  assert.equal(badge().hidden,true);
  update({previewPatch:[{key:'one',artworkKey:'owner:sha',preview:{url:'flagged.jpg'},preview_is_default:true}]});
  assert.equal(badge().hidden,false);
  update({previewPatch:[{key:'one',artworkKey:'owner:sha',preview:{url:'category.jpg'}}]});
  assert.equal(badge().hidden,false);
} else if(scenario==='locale-and-legacy') {
  const text='<img src=x onerror=alert(1)>';
  update(packet({defaultPreviewLabel:text}));
  assert.equal(badge().textContent,text);assert.equal(badge().children.length,0);
  assert.equal(grid.querySelectorAll('img').length,1);
  update(packet({defaultPreviewLabel:'Aperçu par défaut'}));
  assert.equal(badge().textContent,'Aperçu par défaut');assert.equal(badge().hidden,false);
  const legacy=packet();delete legacy.defaultPreviewLabel;delete legacy.cards[0].preview_is_default;
  update(legacy);assert.equal(image().src,'category.jpg');assert.equal(badge().hidden,true);
  update(packet({cards:[card({preview:null,fallback:null,preview_is_default:false})]}));
  assert.equal(button().querySelector('.adapter-card-media').textContent,'Image model');
  assert.equal(badge().hidden,true);
} else throw new Error('Unknown scenario');
"""


class CardPreviewBadgeTests(unittest.TestCase):
    def run_scenario(self, scenario):
        result = subprocess.run(
            ["node", "-e", HARNESS, str(SCRIPT), scenario],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_declared_default_and_decode_fallback_have_a_localized_visible_label(self):
        self.run_scenario("default-and-decode")

    def test_matching_publisher_patch_clears_badge_and_preserves_image_focus_and_scroll(self):
        self.run_scenario("incremental-patches")

    def test_locale_text_is_escaped_and_legacy_payloads_keep_working(self):
        self.run_scenario("locale-and-legacy")


if __name__ == "__main__":
    unittest.main()
