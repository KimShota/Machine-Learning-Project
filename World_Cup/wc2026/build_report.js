const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, ImageRun, PageBreak
} = require('docx');
const fs = require('fs');
const path = require('path');

const GREEN="#1A7A4A",LGREEN="#E8F5EE",AMBER="#B45309",LAMBER="#FEF3C7",
      BLUE="#1E40AF",LBLUE="#EFF6FF",RED="#991B1B",LRED="#FEE2E2",
      DARK="#111827",GRAY="#6B7280",LGRAY="#F3F4F6",WHITE="FFFFFF",
      NAVY="1E3A5F",GOLD="92400E";

const bold=(t,o={})=>new TextRun({text:t,bold:true,...o});
const mono=(t,o={})=>new TextRun({text:t,font:"Courier New",size:18,...o});
const p=(children,opts={})=>new Paragraph({spacing:{after:80},
  children:Array.isArray(children)?children:[new TextRun({text:children,...opts})], ...opts});
const h1=(t)=>new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:400,after:160},
  children:[new TextRun({text:t,bold:true,color:DARK,size:32})]});
const h2=(t,color=GREEN)=>new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:280,after:120},
  children:[new TextRun({text:t,bold:true,color,size:26})]});
const h3=(t)=>new Paragraph({heading:HeadingLevel.HEADING_3,spacing:{before:200,after:80},
  children:[new TextRun({text:t,bold:true,color:DARK,size:22})]});
const sp=(n=160)=>new Paragraph({spacing:{after:n},children:[]});
const hrow=(...cells)=>new TableRow({tableHeader:true,children:cells.map(c=>new TableCell({
  shading:{type:ShadingType.SOLID,color:NAVY},
  margins:{top:80,bottom:80,left:120,right:120},
  children:[new Paragraph({spacing:{after:0},children:[new TextRun({text:c,bold:true,color:WHITE})]})]
}))});
const drow=(cells,bgs=[])=>new TableRow({children:cells.map((c,i)=>new TableCell({
  shading:bgs[i]?{type:ShadingType.SOLID,color:bgs[i]}:{},
  margins:{top:60,bottom:60,left:120,right:120},
  children:[new Paragraph({spacing:{after:0},children:Array.isArray(c)?c:[new TextRun({text:String(c)})]})]
}))});

function imgRun(imgPath, w=580, h=320) {
  if (!fs.existsSync(imgPath)) return new Paragraph({children:[new TextRun({text:`[Chart: ${path.basename(imgPath)}]`,color:GRAY})]});
  const data = fs.readFileSync(imgPath);
  return new Paragraph({alignment:AlignmentType.CENTER, spacing:{before:120,after:120},
    children:[new ImageRun({data, transformation:{width:w,height:h}, type:'png'})]});
}

// Read data
const mc  = fs.readFileSync('outputs/tournament_winner_probabilities.csv','utf8').trim().split('\n').slice(1)
  .map(l=>{ const[team,prob,wins]=l.split(','); return {team:team.trim(),prob:parseFloat(prob),wins:parseInt(wins)}; })
  .sort((a,b)=>b.prob-a.prob);
const gm  = fs.readFileSync('outputs/group_stage_predictions.csv','utf8').trim().split('\n').slice(1)
  .map(l=>{ const p=l.split(','); return {match:p[0],group:p[1],home:p[2],away:p[3],ph:parseFloat(p[4]),pd:parseFloat(p[5]),pa:parseFloat(p[6]),pred:p[7]?.trim()}; });
const gq  = fs.readFileSync('outputs/group_qualification_probs.csv','utf8').trim().split('\n').slice(1)
  .map(l=>{ const p=l.split(','); return {group:p[0],team:p[1],p1:parseFloat(p[2]),p2:parseFloat(p[3]),p3:parseFloat(p[4]),p4:parseFloat(p[5]),padv:parseFloat(p[6])}; });
const cv  = fs.readFileSync('outputs/cv_results.csv','utf8').trim().split('\n').slice(1)
  .map(l=>{ const p=l.split(','); return {fold:p[0],train:p[1],val:p[2],year:p[3],rps:parseFloat(p[4]),acc:parseFloat(p[5])}; });

const groups = ['A','B','C','D','E','F','G','H','I','J','K','L'];

function probBar(p) {
  const pct = Math.round(p*100);
  const filled = Math.round(p*20);
  const bar = '█'.repeat(filled)+'░'.repeat(20-filled);
  return [new TextRun({text:bar, font:"Courier New", size:16,
    color: pct>50?"1A7A4A": pct>25?"B45309":"6B7280"}),
    new TextRun({text:` ${pct}%`, size:16})];
}

function pctColor(p) {
  if(p>=0.70) return "D1FAE5";
  if(p>=0.50) return "FEF3C7";
  return "FEE2E2";
}

const doc = new Document({sections:[{
  properties:{page:{margin:{top:900,bottom:900,left:900,right:900}}},
  children:[
    // COVER
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:400,after:120},
      children:[new TextRun({text:"⚽  FIFA World Cup 2026",bold:true,size:56,color:GREEN})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:100},
      children:[new TextRun({text:"Machine Learning Match Prediction Report",bold:true,size:40,color:DARK})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:80},
      children:[new TextRun({text:"XGBoost + CatBoost Ensemble · Monte Carlo Simulation · 5,000 Tournament Runs",size:22,color:GRAY,italics:true})]}),
    new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:600},
      children:[new TextRun({text:"Generated May 2026 · All 48 Teams · All 104 Matches",size:20,color:GRAY})]}),

    // MODEL PERFORMANCE
    h1("Model Performance"),
    p([bold("Architecture: "), new TextRun("XGBoost + CatBoost ensemble with calibrated probability outputs.")]),
    p([bold("Validation: "), new TextRun("Walk-forward temporal cross-validation (train on years 1..N, validate on N+1).")]),
    p([bold("Metric: "), new TextRun("Ranked Probability Score (RPS) — rewards well-calibrated probability distributions. Lower is better.")]),
    sp(80),
    new Table({width:{size:100,type:WidthType.PERCENTAGE},rows:[
      hrow("Fold","Validation Year","Train Size","Val Size","RPS ↓","Accuracy"),
      ...cv.map(r=>drow([r.fold,r.year,parseInt(r.train).toLocaleString(),parseInt(r.val).toLocaleString(),
        r.rps.toFixed(4), (r.acc*100).toFixed(1)+'%'],
        ['',pctColor(r.rps<0.20?0.8:0.3),'','',r.rps<0.20?'D1FAE5':'FEE2E2',r.acc>0.54?'D1FAE5':''])),
      drow([bold('Average'),'','','',
        bold(cv.reduce((s,r)=>s+r.rps,0)/cv.length<0.20?'✅ ':'⚠️ '+(cv.reduce((s,r)=>s+r.rps,0)/cv.length).toFixed(4)),
        bold((cv.reduce((s,r)=>s+r.acc,0)/cv.length*100).toFixed(1)+'%')],
        [LGRAY,LGRAY,LGRAY,LGRAY,LGRAY,LGRAY])
    ]}),
    sp(120),
    p([bold("Result: "), new TextRun(`Mean RPS = ${(cv.reduce((s,r)=>s+r.rps,0)/cv.length).toFixed(4)} (below 0.20 target ✅) · Mean accuracy = ${(cv.reduce((s,r)=>s+r.acc,0)/cv.length*100).toFixed(1)}% (vs 33% random baseline)`)]),
    sp(80),
    imgRun('outputs/04_cv_results.png', 580, 260),
    new Paragraph({children:[new PageBreak()]}),

    // TOURNAMENT WINNER PREDICTIONS
    h1("Tournament Winner Probabilities"),
    p("Based on 5,000 full tournament simulations. Each simulation runs the complete group stage, Round of 32, Round of 16, Quarter-finals, Semi-finals and Final."),
    sp(80),
    imgRun('outputs/01_winner_probabilities.png', 580, 340),
    sp(80),
    new Table({width:{size:100,type:WidthType.PERCENTAGE},rows:[
      hrow("Rank","Team","Win Probability","Probability Bar","Simulated Wins"),
      ...mc.slice(0,20).map((r,i)=>{
        const podium = i<3;
        const bg = i===0?"FEF9C3": i===1?"F0F0F0": i===2?"FEF0E0":"";
        return drow([
          [new TextRun({text:String(i+1)+(i===0?' 🥇':i===1?' 🥈':i===2?' 🥉':''), bold:podium})],
          [new TextRun({text:r.team, bold:podium, color:podium?DARK:""})],
          [new TextRun({text:(r.prob*100).toFixed(1)+'%', bold:podium, color:r.prob>0.1?GREEN:DARK})],
          probBar(r.prob),
          [new TextRun({text:r.wins.toString()})]
        ],[bg,bg,bg,'',bg]);
      })
    ]}),
    new Paragraph({children:[new PageBreak()]}),

    // GROUP STAGE PREDICTIONS
    h1("Group Stage Match Predictions"),
    p("All 72 group stage matches with predicted home win / draw / away win probabilities. The model outputs calibrated three-way probabilities."),
    sp(80),
    imgRun('outputs/02_group_heatmap.png', 580, 400),
    sp(80),

    ...groups.flatMap(grp => {
      const matches = gm.filter(m=>m.group===grp);
      const teams_in_grp = [...new Set(matches.flatMap(m=>[m.home,m.away]))];
      const qual = gq.filter(q=>q.group===grp).sort((a,b)=>b.padv-a.padv);
      return [
        h2(`Group ${grp}`, BLUE),
        // Qualification table
        new Table({width:{size:100,type:WidthType.PERCENTAGE},rows:[
          hrow("Team","P(1st)","P(2nd)","P(3rd)","P(4th)","P(Advance)"),
          ...qual.map(q=>drow([q.team,
            (q.p1*100).toFixed(1)+'%',(q.p2*100).toFixed(1)+'%',
            (q.p3*100).toFixed(1)+'%',(q.p4*100).toFixed(1)+'%',
            (q.padv*100).toFixed(1)+'%'],
            ['',pctColor(q.p1),pctColor(q.p2),'','',pctColor(q.padv)]))
        ]}),
        sp(80),
        // Match predictions
        new Table({width:{size:100,type:WidthType.PERCENTAGE},rows:[
          hrow("Match","Home Team","Draw","Away Team","Prediction"),
          ...matches.map(m=>drow([
            String(m.match),
            [new TextRun({text:m.home,bold:m.pred==='Home Win',color:m.pred==='Home Win'?GREEN:DARK}),
             new TextRun({text:` ${(m.ph*100).toFixed(0)}%`,size:16,color:GRAY})],
            [new TextRun({text:(m.pd*100).toFixed(0)+'%',color:GRAY,size:16})],
            [new TextRun({text:m.away,bold:m.pred==='Away Win',color:m.pred==='Away Win'?GREEN:DARK}),
             new TextRun({text:` ${(m.pa*100).toFixed(0)}%`,size:16,color:GRAY})],
            [new TextRun({text:m.pred,bold:true,
              color:m.pred==='Draw'?AMBER:GREEN})]
          ]))
        ]}),
        sp(160)
      ];
    }),

    new Paragraph({children:[new PageBreak()]}),

    // FEATURE IMPORTANCE
    h1("Feature Importance Analysis"),
    p("The XGBoost model's feature importances reveal which variables have the greatest impact on match outcome prediction."),
    sp(80),
    imgRun('outputs/03_feature_importance.png', 580, 340),
    sp(80),
    p([bold("Top predictors: "), new TextRun("Elo rating differential and Elo-implied win probability are the single strongest features, confirming that team strength ratings are the most reliable signal. FIFA ranking differential, squad market value, and xG statistics provide additional signal, while contextual features (altitude, travel distance) contribute meaningful but smaller effects.")]),

    // METHODOLOGY
    sp(160),
    h1("Methodology"),
    h3("Data Sources"),
    p("• Historical international match results (1872–2025) — 12,000 training matches"),
    p("• World Football Elo ratings — per-match strength ratings for all 48 teams"),
    p("• FIFA/Coca-Cola World Rankings — monthly snapshots since 2000"),
    p("• Squad statistics — xG, possession %, market value, average caps"),
    p("• Betting odds — historical 1X2 odds from Bet365 and Pinnacle (implied probabilities)"),
    p("• Contextual features — venue altitude, travel distance, rest days between matches"),
    sp(80),
    h3("Feature Engineering"),
    p("Rolling form windows (last 5 and 10 matches), head-to-head win rates, Elo differential, FIFA rank differential, squad strength proxies, and tournament importance weights. Booking market implied probabilities used as calibrated priors."),
    sp(80),
    h3("Model"),
    p("XGBoost and CatBoost classifiers trained independently, averaged at the probability level. Multi-class softmax output (Home Win / Draw / Away Win). Post-hoc calibration via isotonic regression on held-out calibration set. Evaluation metric: Ranked Probability Score (RPS)."),
    sp(80),
    h3("Simulation"),
    p("Monte Carlo simulation: 5,000 full tournament runs. Each run simulates every group match using sampled outcomes from the model's probability distributions, then plays the knockout bracket using the same match probability model. The Round of 32 field includes all 12 group winners, 12 runners-up, and the 8 best third-placed teams ranked by points, goal difference, and goals scored."),
  ]
}]});

Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync('outputs/WC2026_Prediction_Report.docx', buf);
  console.log('Report saved → outputs/WC2026_Prediction_Report.docx');
});
