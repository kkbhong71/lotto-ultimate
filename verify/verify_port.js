/* =====================================================================
 *  Python ↔ JavaScript 이식 교차검증
 *
 *  lotto_ultimate.html 에서 엔진을 직접 추출해 Python 기준값과 대조한다.
 *  엔진 사본을 따로 두지 않으므로 '검증한 코드'와 '배포되는 코드'가
 *  항상 같은 것임이 보장된다.
 *
 *  사용법:
 *    cd verify
 *    python3 crossval_dump.py          # Python 기준값 생성
 *    node verify_port.js               # 대조
 * ===================================================================== */
const fs = require('fs');
global.crypto = require('crypto').webcrypto;

const html = fs.readFileSync(require('path').join(__dirname,'..','index.html'), 'utf8');
const js = html.split('<script>')[1].split('</scr' + 'ipt>')[0];
global.window = { addEventListener: () => {} };
global.document = { querySelector: () => null, addEventListener: () => {} };
const E = new Function(js + `; return { npSum, npMean, npPercentile, npVar,
  safeLoadHistory, analyze, simpleClusters, popularityLog, popularityScore,
  filterAll, decayPowers };`)();

const ref = JSON.parse(fs.readFileSync(require('path').join(__dirname,'pyref.json'), 'utf8'));
const clu = fs.existsSync(require('path').join(__dirname,'pyclu.json')) ? JSON.parse(fs.readFileSync(require('path').join(__dirname,'pyclu.json'),'utf8')) : null;

let FAIL = 0;
const rel = (a,b) => a===b ? 0 : Math.abs(a-b)/Math.max(Math.abs(a),Math.abs(b),Number.MIN_VALUE);
function line(label, maxErr, exact, total, tol) {
  tol = tol === undefined ? 1e-16 : tol;
  const ok = maxErr <= tol; if (!ok) FAIL++;
  console.log(`  ${label.padEnd(20)} 오차 ${maxErr.toExponential(3).padStart(10)}` +
              `  비트동일 ${String(exact).padStart(5)}/${total}  ${ok?'✅':'❌'}`);
}

console.log('━━━ Python ↔ JavaScript 교차검증 ━━━\n');
console.log('[1] 수치 기본 함수 (numpy pairwise summation / percentile 재현)');
let m=0, ex=0, n=0;
for (const u of ref.units) {
  for (const [j,p] of [[E.npSum(u.a),u.sum],[E.npMean(u.a),u.mean],
       [E.npPercentile(u.a,25),u.p25],[E.npPercentile(u.a,50),u.p50],
       [E.npPercentile(u.a,75),u.p75],[E.npVar(u.a),u.var]]) {
    m=Math.max(m,rel(j,p)); if(j===p)ex++; n++;
  }
}
line('sum/mean/pct/var', m, ex, n);

console.log('\n[2] 점수 벡터 11종');
const eng = E.analyze(ref.history);
let vm=0, vex=0, vn=0;
for (const k of Object.keys(ref.vectors)) {
  let mx=0, e2=0;
  for (let i=1;i<=45;i++){ const a=eng[k][i], b=ref.vectors[k][i];
    mx=Math.max(mx,rel(a,b)); if(a===b)e2++; }
  line(k, mx, e2, 45); vm=Math.max(vm,mx); vex+=e2; vn+=45;
}

console.log('\n[3] 인기도 (판정 경로는 로그 스케일 — exp 미사용)');
let pm=0, pex=0;
ref.combos.forEach((c,i)=>{ const v=E.popularityLog(c,eng.last);
  pm=Math.max(pm,rel(v,ref.logs[i])); if(v===ref.logs[i])pex++; });
line('popularityLog', pm, pex, ref.combos.length);
let sm=0, sex=0;
ref.combos.forEach((c,i)=>{ const v=E.popularityScore(c,eng.last);
  sm=Math.max(sm,rel(v,ref.pops[i])); if(v===ref.pops[i])sex++; });
line('popularityScore(표시)', sm, sex, ref.combos.length, 2.3e-16);

console.log('\n[4] 조합 필터 12종');
let mis=0;
ref.combos.forEach((c,i)=>{ if(E.filterAll(c,eng)!==ref.filters[i])mis++; });
console.log(`  ${'필터 판정'.padEnd(20)} 불일치 ${mis}/${ref.combos.length}  ${mis?'❌':'✅'}`);
if (mis) FAIL++;

if (clu) {
  console.log('\n[5] 클러스터 레이어');
  const c = E.simpleClusters(ref.history);
  const topOk = JSON.stringify(c.top)===JSON.stringify(clu.top);
  if(!topOk) FAIL++;
  console.log(`  ${'top_clusters'.padEnd(20)} py=${JSON.stringify(clu.top)} js=${JSON.stringify(c.top)}  ${topOk?'✅':'❌'}`);
  let bid=0,bbo=0;
  clu.probe.forEach((x,i)=>{ if(c.idOf(x)!==clu.ids[i])bid++; if(c.bonusOf(x)!==clu.bonus[i])bbo++; });
  console.log(`  ${'cluster_id'.padEnd(20)} 불일치 ${bid}/${clu.probe.length}  ${bid?'❌':'✅'}`);
  console.log(`  ${'cluster_bonus'.padEnd(20)} 불일치 ${bbo}/${clu.probe.length}  ${bbo?'❌':'✅'}`);
  if(bid||bbo) FAIL++;
}

console.log('\n' + '─'.repeat(58));
console.log(FAIL===0
  ? '  ✅ 전체 통과 — 판정 경로 전 항목 비트 단위 동일 (exp 표시값만 1 ULP)'
  : `  ❌ ${FAIL}개 항목 불일치`);
process.exit(FAIL===0?0:1);
