import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SOURCE = process.env.LITERATE_AI_DECK_SOURCE || HERE;
const REPO = process.env.LITERATE_AI_REPO || path.resolve(SOURCE, "../../..");
const ROOT = process.env.LITERATE_AI_DECK_WORKSPACE || await fs.mkdtemp(path.join(os.tmpdir(), "literate-ai-application-foundry-"));
const ASSETS = path.join(SOURCE, "assets");
const OUT =
  process.env.LITERATE_AI_DECK_OUTPUT ||
  path.join(
    REPO,
    "docs",
    "presentations",
    "literate-ai-manager-overview",
    "literate-ai-manager-and-engineering-overview.pptx"
  );
const W = 1280, H = 720, TOTAL = 22;
const C = {
  ink: "#101317", panel: "#23282F", steel: "#65707C", fog: "#EEF1F3",
  white: "#FFFFFF", orange: "#FF6B35", orange2: "#FF9B66", blue: "#72B7D6",
  green: "#76B900", green2: "#7BC6A4", red: "#F47C7C", line: "#D8DEE3", muted: "#AAB3BC",
  code: "#1A1F26", codeline: "#333B45"
};
const FONT = "Helvetica Neue";
const MONO = "Courier New";
const p = Presentation.create({ slideSize: { width: W, height: H } });

async function bytes(file) {
  const b = await fs.readFile(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
function shape(slide, geometry, x, y, w, h, fill, opts={}) {
  return slide.shapes.add({ geometry, position:{left:x,top:y,width:w,height:h}, fill,
    line: opts.line || {style:"solid",fill:opts.stroke||"none",width:opts.strokeWidth||0},
    borderRadius: opts.radius, shadow: opts.shadow, opacity: opts.opacity });
}
function text(slide, value, x, y, w, h, size=28, color=C.ink, bold=false, opts={}) {
  const s=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=value; s.text.style={fontFamily:opts.font||FONT,fontSize:size,color,bold,italic:opts.italic||false};
  if(opts.align) s.text.paragraphFormat={alignment:opts.align};
  return s;
}
function mono(slide,value,x,y,w,h,size=12,color=C.fog,bold=false){return text(slide,value,x,y,w,h,size,color,bold,{font:MONO});}
function line(slide,x,y,w,h=4,color=C.orange){return shape(slide,"rect",x,y,w,h,color);}
function dot(slide,x,y,d,color,stroke="none"){return shape(slide,"ellipse",x,y,d,d,color,{stroke,strokeWidth:stroke==="none"?0:2});}
function pill(slide,label,x,y,w,fill=C.orange,color=C.white,size=13){shape(slide,"roundRect",x,y,w,32,fill,{radius:"rounded-full"});text(slide,label,x+10,y+6,w-20,20,size,color,true,{align:"center"});}
function addImage(slide,blob,x=0,y=0,w=W,h=H,fit="cover",alt=""){return slide.images.add({blob,contentType:"image/png",alt,fit,position:{left:x,top:y,width:w,height:h}});}
// NOTE: the artifact-tool build in use ignores the shape `opacity` option, so this
// renders as an OPAQUE ink panel. Only ever place it over the typography side of an
// image-led slide; covering the subject side erases the photograph.
function wash(slide,x=0,y=0,w=W,h=H,opacity=.35){return shape(slide,"rect",x,y,w,h,C.ink,{opacity});}
function title(slide,value,sub,dark,n,tag){pill(slide,"LITERATE-AI",70,42,132,dark?C.white:C.ink,dark?C.ink:C.white);if(tag)pill(slide,tag,214,42,182,C.orange,C.ink,12);text(slide,String(n).padStart(2,"0"),1170,45,40,20,13,dark?C.white:C.steel,true,{align:"right"});text(slide,value,70,104,1120,96,42,dark?C.white:C.ink,true);if(sub)text(slide,sub,72,207,1080,55,19,dark?C.muted:C.steel,false);line(slide,70,277,88,6,C.orange);}
function footer(slide,n,dark=false){text(slide,"LITERATE-AI",70,681,150,18,11,dark?C.muted:C.steel,true);line(slide,1080,687,120,3,dark?C.panel:C.line);line(slide,1080,687,Math.max(8,120*n/TOTAL),3,C.orange);}
function arrow(slide,x1,y,x2,color=C.orange){line(slide,x1,y,Math.max(4,x2-x1-13),4,color);shape(slide,"chevron",x2-19,y-8,19,20,color);}
function notes(slide,lines){slide.speakerNotes.textFrame.setText(lines);}

const assets={};
for(const name of ["cover-hero","manager-section","lifecycle-gates","engineering-section","one-many","legacy-transform","application-foundry","living-portfolio"]){assets[name]=await bytes(path.join(ASSETS,`${name}.png`));}

// 01 — cover
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["cover-hero"],0,0,W,H,"cover","Durable intent branching into complete software targets");wash(s,0,0,720,H,.40);
  pill(s,"LITERATE-AI",70,55,132,C.white,C.ink);text(s,"Software that can\nrebuild itself from intent",70,145,625,170,54,C.white,true);text(s,"A path from isolated code generation to governed creation of complete applications and software portfolios.",74,354,530,90,22,C.fog);line(s,74,488,112,7,C.orange);text(s,"THE APPLICATION FOUNDRY VISION",74,528,390,28,17,C.orange2,true);text(s,"2026",1160,668,50,18,12,C.muted,true,{align:"right"});
  notes(s,["Opening frame. This deck argues for a focused application-layer investment; it is not a product-completion announcement.","Slides 7 through 11 are the technical sequence. Point engineering reviewers there; a decision audience can move past them to slide 12.","Every quantified figure in this deck traces to source-notes.md. Nothing here is a modeled ROI or productivity estimate."]);
}

// 02 — the problem
{
  const s=p.slides.add();s.background.fill=C.fog;title(s,"We still scale software by copying its implementation.","Each product, platform, and repository inherits another version of the same knowledge—and another place for it to drift.",false,2);
  text(s,"ONE PRODUCT",75,352,210,30,18,C.steel,true);line(s,124,412,870,5,C.ink);
  const xs=[95,305,515,725,935];xs.forEach((x,i)=>{dot(s,x,383,58,i===0?C.orange:C.white,i===0?C.orange:C.line);text(s,String(i+1),x+19,400,20,20,17,i===0?C.white:C.ink,true,{align:"center"});if(i>0)line(s,x-150,412,150,3,i<3?C.orange:C.red);});
  ["copy","fork","patch","coordinate"].forEach((v,i)=>text(s,v,260+i*210,454,120,24,16,C.steel,true,{align:"center"}));
  text(s,"The implementation multiplies. The intent does not.",75,548,760,46,30,C.ink,true);footer(s,2);
  notes(s,["The problem statement is qualitative and deliberately carries no figures. Do not attach a drift or duplication statistic to it.","The claim is structural: knowledge is re-expressed per copy, so every copy becomes an independent place for it to diverge."]);
}

// 03 — the inversion
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"Make intent the product. Make implementation renewable.","Literate-AI keeps observable behavior readable and durable, then treats source as a candidate that must earn acceptance.",true,3);
  arrow(s,300,423,474);arrow(s,782,423,956);
  shape(s,"roundRect",80,340,220,170,C.orange,{radius:"rounded-xl"});text(s,"INTENT",108,373,164,30,22,C.ink,true,{align:"center"});text(s,"Readable behavior\n+ constraints",108,428,164,55,20,C.ink,true,{align:"center"});
  shape(s,"roundRect",476,340,306,170,C.panel,{radius:"rounded-xl",stroke:"#3A424B",strokeWidth:1});text(s,"REGENERATE",512,373,234,30,22,C.white,true,{align:"center"});text(s,"Exact plan\n+ bounded execution",512,428,234,55,20,C.muted,true,{align:"center"});
  shape(s,"roundRect",958,340,240,170,C.green2,{radius:"rounded-xl"});text(s,"PROOF",992,373,172,30,22,C.ink,true,{align:"center"});text(s,"Build + test\n+ independent acceptance",986,428,184,55,19,C.ink,true,{align:"center"});
  text(s,"Durable",125,555,130,28,18,C.orange2,true,{align:"center"});text(s,"Replaceable",555,555,150,28,18,C.blue,true,{align:"center"});text(s,"Promotable",1003,555,150,28,18,C.green2,true,{align:"center"});footer(s,3,true);
  notes(s,["This is the thesis slide. The three mechanisms behind it are shown on slides 7, 8 and 9 respectively — do not treat this diagram as the explanation.","\"Candidate\" is precise: generated source holds no build, acceptance, cache-membership, or publication authority until the current gates pass."]);
}

// 04 — implementation velocity
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"The implementation is moving unusually fast.","One authority model now spans language, build, operating-system, and packaging choices while preserving exact target constraints.",false,4);
  const facts=[["5","languages"],["2","build systems"],["3","OS families"],["6","package providers"]];facts.forEach((a,i)=>{const x=72+i*296;text(s,a[0],x,338,240,70,48,i===3?C.orange:C.ink,true,{align:"center"});line(s,x+30,423,180,4,i===3?C.orange:C.line);text(s,a[1],x,448,240,30,18,C.steel,true,{align:"center"});});
  text(s,"Exact planning  →  bounded generation  →  build  →  tests  →  execution  →  independent acceptance  →  receipt",86,535,1108,32,19,C.ink,true,{align:"center"});shape(s,"roundRect",168,595,944,38,"#FFF0E8",{radius:"rounded-full"});text(s,"These four figures measure surface area, not maturity. The control model itself is the claim; slides 7–11 show it.",194,606,892,19,15,C.ink,true,{align:"center"});footer(s,4);
  notes(s,["Current matrix: Python 3.11+, C++17, Rust 2021, JavaScript/Node 20+, and Swift; Bazel and GNU Make; macOS, Linux, and Windows Flavors.","Packaging Flavors are pip, Conan, apt, Homebrew, WinGet, and Chocolatey. Multi-provider planning is implemented; native construction and independent verification are live for pip wheels and Conan cache archives.","The private user-owned test matrix currently configures four runner roles: macOS, Windows 11, Ubuntu 24.04, and Ubuntu 26.04. The deck does not disclose hostnames or claim the full Cartesian matrix is qualified."]);
}

// 05 — real sample applications
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"The samples now prove target composition, not just behavior.","Readable Components still generate, build, run, and pass deterministic acceptance—while selected packaging and OS constraints travel with the same authority.",true,5);
  const apps=[["PIP STARTERS","Python applications + verified wheel bytes",C.orange],["CONAN NATIVE","Rust sample + verified cache archive",C.blue],["MACOS SWIFT","LaunchAgent catalog + Apple toolchain + Brew",C.green2],["COMPONENT DAG","Invoice service + exact-money boundary",C.orange2]];
  apps.forEach((a,i)=>{const x=70+i*286;shape(s,"roundRect",x,326,262,178,C.panel,{radius:"rounded-xl",stroke:a[2],strokeWidth:2});text(s,a[0],x+18,348,226,38,16,a[2],true,{align:"center"});line(s,x+76,405,110,4,a[2]);text(s,a[1],x+24,430,214,52,15,C.fog,false,{align:"center"});});
  arrow(s,215,548,402);arrow(s,505,548,692);arrow(s,795,548,982);text(s,"SPEC",88,535,120,28,17,C.white,true,{align:"center"});text(s,"SOURCE",405,535,120,28,17,C.white,true,{align:"center"});text(s,"BINARY",695,535,120,28,17,C.white,true,{align:"center"});text(s,"ACCEPTANCE",985,535,170,28,17,C.green2,true,{align:"center"});
  footer(s,5,true);
  notes(s,["These are repository samples, not product mockups. The catalog now contains 18 behavior applications, including three explicit OS pins.","The opt-in sample package target constructs and independently verifies pip wheels and Conan cache archives from accepted Standard sample artifacts. Homebrew remains exact selected authority and planning only.","Every listed behavior has known output suitable for end-to-end acceptance. The fan-out mechanism and OS filtering appear on slide 15."]);
}

// 06 — the destination
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["application-foundry"],0,0,W,H,"cover","A governed intent graph becoming complete application experiences");wash(s,0,0,610,H,.36);pill(s,"THE NEXT STEP",70,50,150,C.white,C.ink);text(s,"The goal is not better\ncode generation.",70,126,490,105,45,C.white,true);text(s,"It is the governed creation of complete applications: interfaces, services, resources, tests, packages, releases, and runtime projections.",72,278,440,126,21,C.fog);text(s,"An application foundry.",72,486,400,38,28,C.orange2,true);footer(s,6,true);
  notes(s,["Aspirational framing, explicitly labeled as the next step rather than current behavior.","This slide closes the argument section. The next five slides answer \"how does it actually work\" before the deck returns to consequences."]);
}

// 07 — HOW IT WORKS 1/5 — the durable authority, shown
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"This is the durable authority.","A Component specification is a readable file. Requirements, scenarios, and the public capability contract are the product; the source tree is what gets regenerated from them.",false,7,"HOW IT WORKS · 1/5");
  shape(s,"roundRect",70,300,556,306,C.code,{radius:"rounded-lg"});
  text(s,"components/money-calculation/component.md",88,314,520,18,11,C.orange2,true);
  mono(s,"### Requirement: Exact integer discount calculation\n\nThe Component SHALL accept a non-negative integer\nsubtotal in cents and a discount in basis points\nfrom 0 through 10000, compute the discount as\nfloor((subtotal_cents * discount_bp + 5000) / 10000)\n\n#### Scenario: Half-up discount rounding\n- WHEN a 999-cent subtotal receives a\n  1250-basis-point discount\n- THEN the discount is 125 cents and the\n  total is 874 cents",88,344,520,250,12,C.fog);
  shape(s,"roundRect",654,300,556,306,C.panel,{radius:"rounded-lg"});
  text(s,"interfaces/money-calculation.md  —  the public capability contract",672,314,520,18,11,C.blue,true);
  mono(s,"This contract is language-neutral. The realized\nprovider artifact SHALL expose its portable `run`\nentrypoint through the environment variable\nLITAI_CAPABILITY_MONEY_CALCULATION.\n\nConsumers SHALL use this public process contract\nand SHALL NOT import, copy, or depend on private\nprovider source layout.",672,344,520,180,12,C.fog);
  line(s,672,538,520,1,C.codeline);
  mono(s,"flavor_slots:  build.system · language-ecosystem · platform.os",672,552,520,20,11,C.muted);
  shape(s,"roundRect",70,624,1140,40,C.fog,{radius:"rounded-full"});text(s,"Language, build system, operating system, and the entire source tree sit below this line. They are Flavor choices and regeneration targets.",96,636,1088,20,15,C.ink,true,{align:"center"});footer(s,7);
  notes(s,["Both excerpts are verbatim from the repository sample at components/money-calculation/. Nothing on this slide is illustrative prose.","This is the slide that earns the thesis. A deck claiming readable intent is the product must show the readable intent.","The right panel is the reuse boundary: consumers bind to a public process contract, never to provider source layout. That is what makes a provider's private implementation replaceable without touching its consumers."]);
}

// 08 — HOW IT WORKS 2/5 — the generation key
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"What a generation key binds — and what it refuses to.","Cache identity is the narrow boundary the whole model rests on. Bind too much and reuse dies; bind too little and authority leaks across the line.",true,8,"HOW IT WORKS · 2/5");
  shape(s,"roundRect",70,300,552,212,C.panel,{radius:"rounded-lg",stroke:C.orange,strokeWidth:2});
  text(s,"BINDS",92,316,300,24,17,C.orange,true);
  text(s,"Ordered specification set and document identities\nSelected target and Flavors\nSpecification-to-source Skills, workflow, routing policy\nThe exact model and the locked authored assets\nIts own exported public-interface identities\nThe public interfaces of its direct dependencies",92,350,508,150,14,C.fog);
  shape(s,"roundRect",658,300,552,212,C.panel,{radius:"rounded-lg",stroke:C.steel,strokeWidth:2});
  text(s,"DELIBERATELY DOES NOT BIND",680,316,400,24,17,C.muted,true);
  text(s,"Aggregate Component or authoring revisions\nA dependency's revision or private specifications\nA dependency's source tree, build output, or tests\nA dependency's routing policy or skills",680,350,508,110,14,C.fog);
  text(s,"Binding these would smuggle acceptance-oracle and\ndependency-selection identity across the narrow boundary.",680,468,508,40,12,C.muted,false,{italic:true});
  line(s,70,536,88,5,C.orange);text(s,"THE BLAST RADIUS THAT FOLLOWS",70,552,420,22,14,C.orange2,true);
  const br=[["Private leaf change","invalidates the leaf only"],["Exported interface change","invalidates the leaf and its direct consumers"],["Diamond dependency","the leaf appears once per provider-first layer"],["Deeper private graph","the consumer key correctly stays reusable"]];
  br.forEach((a,i)=>{const x=70+i*288;shape(s,"roundRect",x,584,268,74,C.code,{radius:"rounded-lg"});text(s,a[0],x+16,594,236,20,13,C.white,true);text(s,a[1],x+16,616,236,36,12,C.muted);});
  footer(s,8,true);
  notes(s,["Authority: docs/architecture/component-execution-plans.md, \"Generation-key boundary\". Every line on this slide is a stated rule, not a summary.","The refusal column is the substantive engineering claim. Binding aggregate revisions would be the easy implementation and would destroy both reuse and the acceptance-oracle separation.","ComponentGenerationPlan still retains the exact direct edges for audit; cache membership uses generation_key.identity. The audit plan may legitimately change when a provider revision changes even while the consumer source key remains reusable."]);
}

// 09 — HOW IT WORKS 3/5 — layers, concurrency, budgets
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Layered execution. Bounded concurrency. Exact budgets.","Ordering and cost are decided before any model runs. A plan that cannot be scheduled is a stable planning error, not a failed generation.",false,9,"HOW IT WORKS · 3/5");
  shape(s,"roundRect",70,300,556,330,C.fog,{radius:"rounded-lg"});
  const layers=[["LAYER 0",["A","B","C"],C.orange],["LAYER 1",["D","E"],C.blue],["LAYER 2",["F"],C.green2]];
  layers.forEach((L,i)=>{const y=326+i*100;text(s,L[0],92,y+16,86,20,12,C.steel,true);L[1].forEach((n,j)=>{const x=190+j*118;shape(s,"roundRect",x,y,100,52,C.white,{radius:"rounded-lg",stroke:L[2],strokeWidth:2});text(s,n,x,y+14,100,24,19,C.ink,true,{align:"center"});});if(i<2)shape(s,"rect",238,y+52,4,48,C.line);});
  shape(s,"rect",186,318,4,68,C.orange);text(s,"concurrent, up to an explicit bound",190,392,240,32,11,C.orange,true);
  text(s,"Every applicable edge requires the provider to occupy an earlier layer than its consumer.",92,592,510,32,13,C.steel);
  const facts=[["Emission is deterministic","Results emit in canonical Component-revision order, not completion order."],["Budgets are exact","A reported measurement that exceeds the node's budget fails that node and retains the observed values for diagnosis."],["Absent is not zero","Attempts, wall time, tokens, and cost each stay null when unmeasured. The framework never invents a zero."],["Reuse is a strict predicate","Key, context manifest, budget and decision, prompt, recipe, workspace allocation, and complete typed output must all match."]];
  facts.forEach((a,i)=>{const y=302+i*84;line(s,656,y,7,58,i===3?C.orange:C.ink);text(s,a[0],678,y-4,520,24,17,C.ink,true);text(s,a[1],678,y+22,520,52,13,C.steel);});
  footer(s,9);
  notes(s,["Authority: docs/architecture/component-execution-plans.md, sections \"Stable action layers\" and \"Bounded incremental generation\".","This is the cost-control answer. If asked \"what does a run cost\": concurrency is explicitly bounded, per-node budgets are exact and enforced after the fact, and a cache hit reports no model execution at all.","ComponentExecutionPlan carries one action plan for every canonical phase: generate, validate, build, run, package, deploy.","The planner rechecks closure, cycles, and exact public-interface bindings before a coding CLI can run, and never repairs an interface failure by exposing a dependency's private material."]);
}

// 10 — HOW IT WORKS 4/5 — the negative path
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"What happens when a gate fails.","The negative path is the design, not an afterthought. Failure stays bounded, attributable, and cheap to retry.",true,10,"HOW IT WORKS · 4/5");
  const rows=[["FAILED PROVIDER","Dependent generation nodes are cancelled before their adapters are called. Independent branches continue.",C.red],["KEY MISMATCH","A different generation key fails closed before reuse begins. Explicit regeneration bypasses cache membership entirely.",C.orange],["CACHE HIT","Still runs the complete current index, authorization, build, test, execution, and acceptance sequence. A hit is not republished.",C.blue],["BUDGET OVERRUN","Fails the node and retains the observed values. Historical runtime metrics stay on the input membership, never charged to the hit.",C.green2]];
  rows.forEach((a,i)=>{const y=304+i*80;shape(s,"roundRect",70,y,1140,68,C.panel,{radius:"rounded-lg"});line(s,70,y,6,68,a[2]);text(s,a[0],96,y+24,250,24,16,a[2],true);text(s,a[1],356,y+22,830,44,15,C.fog);});
  shape(s,"roundRect",70,640,1140,40,C.code,{radius:"rounded-full"});text(s,"Reuse means only that the exact source-only output remains eligible to enter the current lifecycle. It grants no build, acceptance, or publication authority.",96,652,1088,20,14,C.muted,true,{align:"center"});footer(s,10,true);
  notes(s,["Authority: docs/architecture/component-execution-plans.md, \"Bounded incremental generation\" and the accepted-source membership path.","The cache-hit row is the one most audiences do not expect. Matching identities authorize reuse of the immutable source artifacts only; the service creates a new candidate, output, and provenance projection bound to the current prepared node and reruns every current gate.","Any prepared recipe lock must equal the current execution-plan lock. A mismatch fails before work starts."]);
}

// 11 — HOW IT WORKS 5/5 — the trust boundary
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"What never crosses the model boundary.","Generated bytes are untrusted input. The build contract has to exist before they do.",false,11,"HOW IT WORKS · 5/5");
  shape(s,"roundRect",70,300,556,250,C.code,{radius:"rounded-lg"});
  text(s,"THE BUILD DECLARATION PRECEDES THE BYTES",92,318,480,22,13,C.orange2,true);
  text(s,"A BuildRequestDeclaration names its builder, toolchain,\nsandbox, privileges, and outputs before any generated\nsource exists.",92,352,500,74,15,C.fog);
  shape(s,"roundRect",92,438,500,44,C.panel,{radius:"rounded-lg"});text(s,"It deliberately cannot name a source bundle.",112,451,460,22,16,C.orange,true,{align:"center"});
  text(s,"A source-bound BuildRequest is realized only after the generated tree, current test manifest, and source SBOM pass admission.",92,496,500,44,13,C.muted);
  text(s,"ORDERED ISOLATION LEVELS",658,306,420,22,13,C.steel,true);
  const lv=[["host-yolo","No containment boundary. Ambient authority of the invoking user.",C.red],["process-limited","Declared process-tree, time, output, and resource limits. Not a security boundary.",C.orange],["os-sandboxed","OS-enforced filesystem, credential, device, network, and process controls.",C.green2]];
  lv.forEach((a,i)=>{const y=338+i*72;shape(s,"roundRect",658,y,552,62,C.fog,{radius:"rounded-lg"});line(s,658,y,6,62,a[2]);text(s,a[0],680,y+8,240,20,15,C.ink,true,{font:MONO});text(s,a[1],680,y+30,510,26,12,C.steel);});
  text(s,"A local decision is explicitly unauthenticated-local: useful for diagnostics, never execution authorization or release evidence.",658,558,552,40,12,C.steel,false,{italic:true});
  shape(s,"roundRect",70,616,1140,64,"#FFF0E8",{radius:"rounded-lg"});text(s,"Current state, stated plainly: the live sample path runs with explicit YOLO authorization as the host user, and source observation has narrower macOS and Linux sandbox\nadapters. Production containment is an architecture contract in this repository — not a shipped backend.",96,630,1088,44,14,C.ink,true,{align:"center"});footer(s,11);
  notes(s,["Authority: docs/architecture/production-containment-threat-model.md and the typed build boundary in component-execution-plans.md.","Do not soften the orange band. The threat model document itself opens by saying it is a contract, not a claim that production containment backends exist.","The declaration-cannot-name-a-source-bundle rule is the concrete expression of \"generated source is untrusted\": the build's authority is fixed before the model produces anything.","A future trusted launcher authorization must precede execution, and authenticated OPS-300 evidence must qualify the resulting report."]);
}

// 12 — evidence
{
  const s=p.slides.add();s.background.fill=C.white;title(s,"Correctness ships with the artifact.","The result is source plus the evidence needed to trust, review, release, and reproduce it.",false,12);
  const pillars=[["BEHAVIOR","Generated tests and bounded execution answer whether the software works.",C.orange],["INTENT","Independent acceptance answers whether software and tests satisfy the specification.",C.blue],["PROVENANCE","Pre-build and post-build CycloneDX SBOMs, exact identities, journals, and receipts answer precisely what produced the result.",C.green2]];
  pillars.forEach((a,i)=>{const x=72+i*390;line(s,x,348,330,9,a[2]);text(s,a[0],x,382,330,32,24,C.ink,true);text(s,a[1],x,438,330,130,18,C.steel);});shape(s,"roundRect",260,575,760,43,C.ink,{radius:"rounded-full"});text(s,"Acceptance is a property of the release—not a memory of the process.",286,588,708,22,16,C.white,true,{align:"center"});footer(s,12);
  notes(s,["Independent acceptance is a separate oracle from the generating model. That separation is exactly why the generation key refuses to bind aggregate authoring revisions (slide 8).","Pre-build and post-build CycloneDX evidence are supported today; authority is docs/architecture/sbom-and-dependency-graph.md.","Each per-node result binds the current context and budget identities and is the common journal, benchmark, and receipt-facing evidence surface."]);
}

// 13 — component DAG
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"Applications are arbitrary Component DAGs—not flattened prompts.","A hierarchy may nest Components to any depth. Planning resolves it into exact public-interface edges and provider-before-consumer layers.",true,13);
  const ns=[[70,390,170,"AUTH + POLICY",C.orange],[330,316,190,"WEB FRONTEND",C.blue],[330,468,190,"CLI",C.blue],[650,390,190,"API SERVICE",C.orange2],[980,316,210,"APPLICATION A",C.green2],[980,468,210,"APPLICATION B",C.green2]];
  [[240,438,330],[240,438,330],[520,364,650],[520,516,650],[840,438,980],[840,438,980]].forEach((e,i)=>{const y=i===0?364:i===1?516:i===2?364:i===3?516:i===4?364:516;arrow(s,e[0],y,e[2],i<2?C.orange:C.steel);});
  ns.forEach(n=>{shape(s,"roundRect",n[0],n[1],n[2],82,n[4]===C.orange?C.orange:C.panel,{radius:"rounded-xl",stroke:n[4],strokeWidth:2});text(s,n[3],n[0]+12,n[1]+25,n[2]-24,34,16,n[4]===C.orange?C.ink:C.white,true,{align:"center"});});
  shape(s,"roundRect",70,584,1120,54,C.code,{radius:"rounded-lg"});text(s,"Only direct contracts cross an edge. Private specifications, source, tests, and transitive context stay behind the provider boundary.",92,600,1076,26,16,C.orange2,true,{align:"center"});footer(s,13,true);
  notes(s,["This is a diamond-shaped example of the implemented arbitrary acyclic graph model: one provider can feed multiple consumers, and multiple applications can share the same downstream capability.","The planner rejects cycles, validates exact interface bindings, emits canonical provider-before-consumer layers, and permits independent nodes in a layer to run concurrently.","Hierarchy is an authoring convenience; flattening is explicitly forbidden. A consumer receives its direct public contracts, never private transitive Component detail."]);
}

// 14 — one intent, many targets
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["one-many"],0,0,W,H,"cover","One governed intent producing cloud, CLI, embedded, mobile, and data targets");wash(s,0,0,585,H,.30);pill(s,"COMPOUNDING REUSE",70,48,202,C.white,C.ink);text(s,"One application intent\ncan serve many products\nand runtimes.",70,122,455,158,44,C.white,true);text(s,"Flavors make language, OS, build, packaging, deployment, and hardware variation explicit—without forking behavior.",72,336,430,112,20,C.fog);
  shape(s,"roundRect",72,462,440,58,C.panel,{radius:"rounded-lg"});text(s,"Packaging composes too: pip + Conan can serve different\ncommunities from the same exact Component authority.",92,474,410,36,13,C.fog);
  text(s,"Variation becomes a first-class product decision.",72,542,440,50,24,C.orange2,true);footer(s,14,true);
  notes(s,["Flavor axes include build.system, implementation.language-ecosystem, platform.os, packaging, toolchain, and documentation.ecosystem.","Packaging is multi-value: compatible providers form independent plans. OS constraints still reject apt, Homebrew, WinGet, or Chocolatey on incompatible targets before generation.","The documentation.ecosystem axis governs format and collaboration surface, never product claims, approval state, build execution, or release authority."]);
}

// 15 — cross-platform fan-out
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"One command fans proof across real operating systems.","The user supplies private runner destinations. Literate-AI selects each platform Flavor, checks out the exact revision, and executes targets in parallel.",true,15);
  shape(s,"roundRect",70,326,250,224,C.orange,{radius:"rounded-xl"});text(s,"FAN-OUT DRIVER",96,354,198,28,20,C.ink,true,{align:"center"});mono(s,"sample: *\nrevision: exact Git SHA\nmode: fail-fast\nresume: checkpoint",96,406,198,100,14,C.ink,true);
  const runners=[[420,310,"macOS","Swift + Homebrew sample",C.blue],[730,310,"Windows 11","C++ + Conan path sample",C.orange2],[420,470,"Ubuntu 24.04","C++ + Conan cgroup sample",C.green2],[730,470,"Ubuntu 26.04","C++ + Conan cgroup sample",C.green2]];
  runners.forEach((a,i)=>{arrow(s,320,i<2?390:486,a[0],a[4]);shape(s,"roundRect",a[0],a[1],250,106,C.panel,{radius:"rounded-xl",stroke:a[4],strokeWidth:2});text(s,a[2],a[0]+20,a[1]+20,210,28,20,C.white,true,{align:"center"});text(s,a[3],a[0]+20,a[1]+58,210,30,13,C.muted,false,{align:"center"});});
  shape(s,"roundRect",1010,348,200,178,C.code,{radius:"rounded-xl",stroke:C.green2,strokeWidth:2});text(s,"COMPACT\nEVIDENCE",1032,376,156,54,19,C.green2,true,{align:"center"});text(s,"passed targets\nfail-fast repair\nfull rerun resets",1032,444,156,60,13,C.fog,false,{align:"center"});
  text(s,"Runner identities never enter the repository. The matrix is operator-owned configuration.",215,606,850,28,18,C.orange2,true,{align:"center"});
  footer(s,15,true);
  notes(s,["Implemented by scripts/fanout_samples.py and documented in docs/user/samples.md. Targets execute concurrently with bounded parallelism and platform Flavors selected from the user-owned matrix.","The catalog retains every sample for readers. Executable discovery admits a pinned sample only when its platform matches the worker, before a coding agent is invoked.","Successful target IDs are checkpointed under OBJ_DIR. A completed repair cycle still requires one clean from-zero rerun for release evidence."]);
}

// 16 — one operating model
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["manager-section"],0,0,W,H,"cover","A coordinated delivery path across teams and evidence gates");wash(s,0,0,620,H,.40);pill(s,"ONE OPERATING MODEL",70,50,205,C.white,C.ink);text(s,"One authority connects\nroadmap to operations.",70,136,500,96,39,C.white,true);text(s,"Product intent, engineering execution, program dependencies, and operational evidence become different views of the same content-identified system.",72,286,460,118,19,C.fog);["ROADMAP","ENGINEERING","RELEASE","OPERATIONS"].forEach((l,i)=>pill(s,l,620+i*150,620,132,i===3?C.green2:C.panel,C.white));footer(s,16,true);
  notes(s,["\"Content-identified\" is literal: identities are content addresses, which is what lets these four views reference the same objects without a synchronization process.","This is a consequence slide. Its mechanism was established on slides 8 and 9."]);
}

// 17 — modernization
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["legacy-transform"],0,0,W,H,"cover","A legacy system mapped into reviewed intent and reusable components");shape(s,"rect",0,0,660,H,C.ink);
  pill(s,"MODERNIZATION",70,50,155,C.white,C.ink);text(s,"Existing systems become\nstarting knowledge.",70,124,540,96,40,C.white,true);
  const steps=[["INDEX","An inert mirror is indexed without executing the mirrored source.","01"],["DRAFT","Reviewable Component and Flavor material is produced, and every model call is journalled.","02"],["QUALIFY","Human review and current qualification evidence decide when authority transfers.","03"]];
  steps.forEach((a,i)=>{const y=278+i*88;shape(s,"roundRect",70,y,520,76,C.panel,{radius:"rounded-lg"});text(s,a[2],88,y+26,44,24,15,C.orange,true);text(s,a[0],138,y+12,200,24,16,C.white,true);text(s,a[1],138,y+36,430,34,13,C.muted);});
  shape(s,"roundRect",70,560,520,60,"#3A2A22",{radius:"rounded-lg"});line(s,70,560,6,60,C.orange);text(s,"Draft extraction is available today. Effective authority remains\nsource-baseline until trusted, current qualification evidence exists.",96,574,480,36,13,C.orange2,true);
  text(s,"Capture intent before replacing implementation.",70,640,520,40,20,C.white,true);footer(s,17,true);
  notes(s,["Authority: docs/architecture/source-promotion.md and the source-qualification schema.","The current qualification model carries a v2-required boundary. Draft generation is explicitly not automatic authority transfer.","Do not claim live repository acquisition as an ordinary generation behavior. That has not become implemented and verified."]);
}

// 18 — litai CLI
{
  const s=p.slides.add();s.background.fill=C.fog;title(s,"`litai` exposes the lifecycle without replacing native tools.","The CLI owns specification, target, package-plan, and release authority; selected build and package ecosystems perform their native work.",false,18);
  const stages=[["AUTHOR","init · component · spec · lock",C.orange],["RESOLVE","verify · plan · catalog",C.blue],["REALIZE","generate · build · test · run",C.green2],["SHIP","package · release · update",C.orange2]];
  stages.forEach((a,i)=>{const x=70+i*286;shape(s,"roundRect",x,328,262,164,C.white,{radius:"rounded-xl",stroke:a[2],strokeWidth:3});text(s,a[0],x+20,350,222,28,18,a[2],true,{align:"center"});line(s,x+72,394,118,4,a[2]);mono(s,a[1],x+22,422,218,48,13,C.ink,true);});
  shape(s,"roundRect",144,536,992,82,C.ink,{radius:"rounded-xl"});mono(s,"litai package plan  →  build --allow-host-execution  →  verify  →  release",174,554,932,26,16,C.orange2,true);text(s,"One accepted closure  •  multiple compatible native formats  •  publication remains separately authorized",174,586,932,22,14,C.fog,false,{align:"center"});footer(s,18);
  notes(s,["Current `litai help` includes release, package, build, test, run, verify, update, reparent, plan, generate, rebuild, and the authoring/maintenance verbs shown here.","Package plan is read-only. Package build reruns the accepted Standard lifecycle before constructing selected pip/Conan bytes beneath OBJ_DIR; package verify independently reopens them. None of these verbs publishes.","Native build and package managers remain delegated execution engines, never specification authority."]);
}

// 19 — the investment horizon
{
  const s=p.slides.add();s.background.fill=C.ink;title(s,"A focused two-week investment horizon can close the application loop.","The framework core is present. The next investment connects it completely to application authoring, release, and runtime surfaces.",true,19);
  const work=[["ORCHESTRATE","Complete durable browser-to-worker lifecycle execution"],["RELEASE","Bind packages, receipts, deployment manifests, and launch"],["COMPOSE","Expand application recipes, resources, interfaces, and shared Components"],["QUALIFY","Prove one demanding application end to end with fresh evidence"]];work.forEach((a,i)=>{const x=58+i*304;line(s,x,345,260,8,i===3?C.green2:C.orange);text(s,a[0],x,382,260,30,22,C.white,true);text(s,a[1],x,434,260,94,17,C.muted);});text(s,"Target horizon, not a readiness claim.",70,592,370,24,16,C.orange2,true);footer(s,19,true);
  notes(s,["The two-week horizon is supplied by the project owner as a target. It is not a measured delivery guarantee, and source-notes.md records it that way.","If asked to commit: the four outcomes are the commitment, the duration is the estimate."]);
}

// 20 — end state
{
  const s=p.slides.add();s.background.fill=C.ink;addImage(s,assets["living-portfolio"],0,0,W,H,"cover","A living capability graph feeding independently evolving application portfolios");wash(s,0,0,575,H,.35);pill(s,"THE END STATE",70,50,145,C.white,C.ink);text(s,"A living portfolio\nof intent.",70,136,430,104,48,C.white,true);text(s,"Shared capabilities evolve once. Products compose them independently. Every change carries its dependency path and proof forward.",72,294,420,118,21,C.fog);text(s,"Software becomes a regeneratable knowledge system.",72,506,440,64,25,C.orange2,true);footer(s,20,true);
  notes(s,["Explicitly the long-term destination, not near-term investment. Keep that distinction audible.","\"Carries its dependency path forward\" is the blast-radius rule from slide 8 stated as an outcome."]);
}

// 21 — the ask
{
  const s=p.slides.add();s.background.fill=C.fog;title(s,"Start with one demanding application. Build for the portfolio.","Use the next investment cycle to prove the complete loop on a real application while preserving the architecture needed for many.",false,21);
  const asks=[["TWO WEEKS","Focused application-layer extension"],["ONE APPLICATION","Enough complexity to force real composition"],["CURRENT GATES","No shortcuts around build, test, acceptance, or evidence"]];asks.forEach((a,i)=>{const x=72+i*390;line(s,x,346,330,8,i===0?C.orange:i===1?C.blue:C.green2);text(s,a[0],x,386,330,34,25,C.ink,true);text(s,a[1],x,444,330,72,18,C.steel);});shape(s,"roundRect",240,570,800,46,C.ink,{radius:"rounded-full"});text(s,"Decision: fund the extension and select the first end-to-end application.",268,584,744,22,17,C.white,true,{align:"center"});footer(s,21);
  notes(s,["This is the decision slide. The requested outcome is funding plus the selection of the first end-to-end application.","The third column is not boilerplate: proving the loop while relaxing the gates would prove nothing, because acceptance is what makes the result a release candidate rather than a code dump."]);
}

// 22 — close
{
  const s=p.slides.add();s.background.fill=C.ink;pill(s,"LITERATE-AI",70,58,132,C.white,C.ink);text(s,"Build the system\nthat builds the software.",70,150,760,132,56,C.white,true);line(s,74,336,120,8,C.orange);text(s,"Durable intent. Renewable implementation. Evidence-backed applications. A portfolio that compounds.",74,384,710,92,25,C.fog);shape(s,"roundRect",74,548,410,54,C.orange,{radius:"rounded-full"});text(s,"ONE APPLICATION → A REGENERATABLE PORTFOLIO",98,565,362,23,16,C.ink,true,{align:"center"});text(s,"22",1162,678,48,18,12,C.muted,true,{align:"right"});
  notes(s,["Close on the decision from slide 21, not on the vision.","For follow-up questions on mechanism, return to slides 7 through 11. The full authority is docs/architecture/component-execution-plans.md and production-containment-threat-model.md."]);
}

await fs.mkdir(`${ROOT}/slides`,{recursive:true});await fs.mkdir(`${ROOT}/layout`,{recursive:true});await fs.mkdir(`${ROOT}/preview`,{recursive:true});await fs.mkdir(`${ROOT}/qa`,{recursive:true});await fs.mkdir(path.dirname(OUT),{recursive:true});
for(const [i,s] of p.slides.items.entries()){const stem=`slide-${String(i+1).padStart(2,"0")}`;const png=await p.export({slide:s,format:"png",scale:1});await fs.writeFile(`${ROOT}/slides/${stem}.png`,new Uint8Array(await png.arrayBuffer()));const layout=await s.export({format:"layout"});await fs.writeFile(`${ROOT}/layout/${stem}.layout.json`,await layout.text());}
const montage=await p.export({format:"webp",montage:true,scale:.35});await fs.writeFile(`${ROOT}/preview/deck-montage.webp`,new Uint8Array(await montage.arrayBuffer()));const pptx=await PresentationFile.exportPptx(p);await pptx.save(OUT);const inspect=await p.inspect({kind:"slide,textbox,shape,image",maxChars:200000});await fs.writeFile(`${ROOT}/qa/inspect.ndjson`,inspect.ndjson);await fs.rm(`${OUT}.inspect.ndjson`,{force:true});

// Capability manifest for `literate-ai.document-pair`. Realizes the local member only:
// this program holds no publication authorization, so `published_location` stays null.
// It is written to the render workspace, never into the repository.
const MANIFEST={
  schema:"literate-ai/document-pair-manifest@1",
  ecosystem:"google-workspace",
  members:{
    presentation:{
      local_artifact:OUT,
      published_location:null,
      publication_authorized:false,
      access:{audience:"organization",principals:[],permission:"view",link_sharing:"organization-restricted"}
    }
  },
  authoring_package:{
    root:SOURCE,
    elements:{
      narrative_specification:"deck-specification.md",
      factual_ledger:"source-notes.md",
      generation_prompts:"prompts",
      build_source:"build_deck.mjs",
      assets:"assets",
      regeneration_entry_point:"regenerate.sh",
      deliverable_links:"current-deliverables.md",
      qa_record:"qa-ledger.md"
    }
  }
};
await fs.writeFile(`${ROOT}/capability-manifest.json`,JSON.stringify(MANIFEST,null,2)+"\n");
console.log(`built ${p.slides.items.length} slides -> ${OUT}`);
console.log(`capability manifest -> ${ROOT}/capability-manifest.json`);
