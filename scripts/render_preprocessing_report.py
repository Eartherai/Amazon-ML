"""Build standalone measured figures, exact distribution-shift metrics and gallery."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'white','axes.titleweight':'bold','savefig.facecolor':'#f6f8fb'})
COLORS=['#2463aa','#ec8944','#2b936f','#9a56a2','#d15761','#72818d']


def distribution_metrics(x:dict,y:dict)->dict:
    """Exact histogram JSD divergence (base 2), PSI with stated floor, Wasserstein."""
    keys=sorted(x.keys()|y.keys());xx=np.array([x.get(k,0) for k in keys],dtype=float);yy=np.array([y.get(k,0) for k in keys],dtype=float)
    xx/=xx.sum();yy/=yy.sum()
    p=np.maximum(xx,1e-6);q=np.maximum(yy,1e-6);p/=p.sum();q/=q.sum()
    return {'js_divergence_bits':float(jensenshannon(xx,yy,base=2)**2),
            'psi_floor_1e_6':float(np.sum((q-p)*np.log(q/p))),
            'wasserstein':float(wasserstein_distance(keys,keys,xx,yy))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,default=Path('artifacts/data_profile/PROFILE-001'))
    p.add_argument('--pairs',type=Path,default=Path('artifacts/pair_analysis/MORPH-002'));p.add_argument('--figures',type=Path,default=Path('docs/figures'))
    a=p.parse_args();a.figures.mkdir(parents=True,exist_ok=True)
    fields={}
    for path in a.profile.glob('*.business_*.json'):
        source,field=path.stem.split('.');fields[(source,field)]=json.loads(path.read_text())
    if len(fields)!=12:raise ValueError(f'Require complete 12 field checkpoints, got {len(fields)}')
    sources=[f'{split}_source{i}' for split in ['train','test'] for i in [1,2,3]]
    gallery=[]
    def save(fig,key,title,note):
        fig.suptitle(title,fontsize=17,x=.04,ha='left',y=.995)
        fig.text(.04,.01,note,fontsize=8,color='#46536a',wrap=True)
        fig.tight_layout(rect=[0,.055,1,.95]);path=a.figures/('phase2_'+key+'.png');fig.savefig(path,dpi=150);plt.close(fig)
        gallery.append({'file':path.name,'title':title,'note':note})
    # Counts / composition exact full corpus.
    fig,axes=plt.subplots(1,2,figsize=(13,5));countries=['India','US','France']
    for j,metric in enumerate(['count','share']):
        bottom=np.zeros(6)
        for k,country in enumerate(countries):
            vals=np.array([sum(c['rows'] for c in fields[(s,'business_name')]['by_country'] if c['country']==country) for s in sources],float)
            if metric=='share':vals=vals/np.array([sum(c['rows'] for c in fields[(s,'business_name')]['by_country']) for s in sources])*100
            else:vals/=1e6
            axes[j].bar(np.arange(6),vals,bottom=bottom,label=country,color=COLORS[k]);bottom+=vals
        axes[j].set_xticks(range(6),[s.replace('_source',' S') for s in sources],rotation=25);axes[j].set_ylabel('Millions of records' if metric=='count' else '% of source');axes[j].legend()
    save(fig,'volume','24.23 million source records: country composition changes','Exact full-source counts. France exists only in test; no France labels are available.')
    shifts=[]
    for field in ['business_name','business_address']:
        pretty=field.replace('business_','').title()
        for kind in ['length','tokens','ascii_numeric_tokens']:
            fig,axes=plt.subplots(1,3,figsize=(15,4.6))
            for j,country in enumerate(countries):
                for i,s in enumerate(sources):
                    hist={r['value']:r['n'] for r in fields[(s,field)]['histograms'] if r['country']==country and r['kind']==kind}
                    if not hist:continue
                    total=sum(hist.values());x=sorted(hist);y=np.cumsum([hist[v]/total for v in x]);axes[j].plot(x,y*100,label=s.replace('_source',' S'),color=COLORS[i],ls='-' if i<3 else '--')
                axes[j].set_title(country);axes[j].set_ylim(0,101);axes[j].set_xlabel(kind.replace('_',' '));axes[j].grid(alpha=.15);axes[j].legend(fontsize=7)
                if kind=='length':axes[j].set_xlim(0,180 if field=='business_address' else 100)
            axes[0].set_ylabel('Cumulative % of records')
            save(fig,field+'_'+kind,pretty+': exact distribution by source and country','All records; curves show the central range where useful. Full tails and quantiles are retained in PROFILE-001.')
            for source in [1,2,3]:
                for country in ['India','US']:
                    x={r['value']:r['n'] for r in fields[(f'train_source{source}',field)]['histograms'] if r['country']==country and r['kind']==kind}
                    y={r['value']:r['n'] for r in fields[(f'test_source{source}',field)]['histograms'] if r['country']==country and r['kind']==kind}
                    shifts.append({'source':source,'country':country,'field':field,'kind':kind,**distribution_metrics(x,y)})
        labels=[];rates=[];char=[];vocab=[];dup=[]
        for s in sources:
            f=fields[(s,field)]
            for c in f['by_country']:
                labels.append(s.replace('_source',' S')+' '+c['country']);rates.append([100*c[k]/c['rows'] for k in ['blank_after_trim','nonascii_rows','combining_mark_rows','repeated_whitespace_rows','ascii_digit_rows']]);char.append(list(c['character_category_ratios'].values()))
                v=next(v for v in f['vocabulary']['by_country'] if v['country']==c['country']);vocab.append([v['unique_tokens'],v['singleton_tokens']/v['unique_tokens']*100])
                raw=next(v for v in f['duplicates'] if v['country']==c['country'] and v['representation']=='raw');light=next(v for v in f['duplicates'] if v['country']==c['country'] and v['representation']!='raw')
                dup.append([100*raw['duplicate_excess']/c['rows'],100*(raw['distinct_values']-light['distinct_values'])/c['rows']])
        punctuation_chars=['&',"'",'-',',','.','(',')','/']
        punct=[];scripts=[]
        script_names=['latin','devanagari','bengali','gurmukhi','gujarati','oriya','tamil','telugu','kannada','malayalam']
        for source in sources:
            data=fields[(source,field)]
            for country_data in data['by_country']:
                punct.append([100*sum(r['records'] for r in data['punctuation'] if r['country']==country_data['country'] and r['character']==char)/country_data['rows'] for char in punctuation_chars])
                scripts.append([100*country_data['script_'+name+'_rows']/country_data['rows'] for name in script_names])
        for suffix,values,columns,title in [('punctuation',punct,punctuation_chars,'punctuation presence'),('scripts',scripts,script_names,'script indicators')]:
            fig,ax=plt.subplots(figsize=(13,7));im=ax.imshow(values,cmap='Blues',aspect='auto',vmin=0,vmax=100);ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(len(columns)),columns,rotation=25)
            for row in range(len(labels)):
                for col in range(len(columns)):ax.text(col,row,f'{values[row][col]:.1f}',ha='center',va='center',fontsize=7,color='white' if values[row][col]>55 else '#17263c')
            fig.colorbar(im,ax=ax,label='% of records');save(fig,field+'_'+suffix,pretty+': '+title,'Exact full-corpus document presence; indicators overlap. Script patterns are not inferred language labels.')
        fig,ax=plt.subplots(figsize=(12,7));im=ax.imshow(rates,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto');ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(5),['Blank','Non-ASCII','Combining marks','Repeated space','Any digit'])
        for row in range(len(labels)):
            for col in range(5):ax.text(col,row,f'{rates[row][col]:.1f}%',ha='center',va='center',color='white' if rates[row][col]>55 else '#122238',fontsize=8)
        fig.colorbar(im,ax=ax,label='% of records');save(fig,field+'_quality',pretty+': missingness, scripts and formatting','Exact full-corpus row rates. Indicators overlap; combining marks can be essential parts of a written word.')
        fig,ax=plt.subplots(figsize=(12,7));bottom=np.zeros(len(labels))
        for i,key in enumerate(['letters','marks','numbers','punctuation','symbols','separators','controls']):
            vals=np.array(char)[:,i]*100;ax.barh(range(len(labels)),vals,left=bottom,label=key);bottom+=vals
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('% of codepoints');ax.legend(ncol=4,fontsize=8,loc='lower right');save(fig,field+'_unicode',pretty+': Unicode categories','Character-weighted, exact full corpus. Categories partition codepoints; these are not record-level percentages.')
        fig,axes=plt.subplots(1,2,figsize=(14,7))
        for j in range(2):
            axes[j].barh(range(len(labels)),np.array(vocab)[:,j],color=COLORS[j]);axes[j].set_yticks(range(len(labels)),labels if j==0 else ['']*len(labels));axes[j].invert_yaxis();axes[j].set_xlabel('Distinct tokens' if j==0 else '% of vocabulary seen in only one record')
        save(fig,field+'_vocabulary',pretty+': a long tail of rare tokens','Exact within-record token document frequency. Source sizes differ; rarity is not automatically an error.')
        fig,ax=plt.subplots(figsize=(12,7));ax.barh(range(len(labels)),np.array(dup)[:,0],label='Raw duplicate excess',color=COLORS[0]);ax.barh(range(len(labels)),np.array(dup)[:,1],left=np.array(dup)[:,0],label='Additional normalized collapse',color=COLORS[1]);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel('% of records beyond the first value in each bucket');ax.legend()
        save(fig,field+'_duplicates',pretty+': collision pressure','Exact within-source/country field buckets, including blanks. Field equality does not imply business identity.')
        fig,axes=plt.subplots(1,3,figsize=(15,6))
        for j,country in enumerate(countries):
            s='test_source1' if country=='France' else 'train_source1';v=next(v for v in fields[(s,field)]['vocabulary']['by_country'] if v['country']==country);top=v['top_tokens'][:12]
            # Plot Unicode token spelling where available; tables remain the authoritative UTF-8 representation.
            axes[j].barh(range(len(top)),[100*t['df']/v['documents'] for t in top],color=COLORS[j]);axes[j].set_yticks(range(len(top)),[t['token'] for t in top]);axes[j].invert_yaxis();axes[j].set_title(country+(' (test)' if country=='France' else ' (train)'));axes[j].set_xlabel('% of S1 records containing token')
        save(fig,field+'_common_tokens',pretty+': common tokens offer weak identity evidence','Exact S1 document frequency. Test France is descriptive only; no learned mapping uses its labels.')
    pl.DataFrame(shifts).write_csv('docs/TRAIN_TEST_SHIFT.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for j,field in enumerate(['business_name','business_address']):
        grid=np.array([[next(r['js_divergence_bits'] for r in shifts if r['source']==s and r['country']==c and r['field']==field and r['kind']==k) for k in ['length','tokens','ascii_numeric_tokens']] for s in [1,2,3] for c in ['India','US']]);im=axes[j].imshow(grid,cmap='OrRd',vmin=0,vmax=max(.01,grid.max()),aspect='auto');axes[j].set_xticks(range(3),['Length','Tokens','Numeric tokens']);axes[j].set_yticks(range(6),[f'S{s} {c}' for s in [1,2,3] for c in ['India','US']]);axes[j].set_title(field.replace('business_','').title())
        for y in range(6):
            for x in range(3):axes[j].text(x,y,f'{grid[y,x]:.4f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=axes[j],label='Jensen–Shannon divergence (bits)')
    save(fig,'shift','Train–test shift within known countries','Exact full histograms. Compare countries separately to avoid hiding shifts behind changing country proportions.')
    summary=json.loads((a.pairs/'summary.json').read_text());frame=pl.read_parquet(a.pairs/'features.parquet');ab=pl.read_csv(a.pairs/'normalization_ablations.csv');diag=pl.read_csv(a.pairs/'feature_diagnostics.csv')
    populations=['positive','random','same_country','exact_name_full_pool','tfidf_name_sample_pool','tfidf_address_sample_pool']
    for field in ['name','address']:
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,feature in zip(axes.flat,['jw','levenshtein','token_set','char3_cosine','word_tfidf_cosine','numeric_jaccard']):
            for i,pop in enumerate(populations):
                vals=frame.filter(pl.col('population')==pop)[field+'_'+feature].to_numpy();hist,edges=np.histogram(vals,bins=np.linspace(0,1,21));ax.step(edges[1:],np.cumsum(hist)/max(1,len(vals)),label=pop.replace('_sample_pool','').replace('_full_pool',''),color=COLORS[i]);
            ax.set_title(feature.replace('_',' '));ax.set_xlabel('Similarity');ax.set_ylabel('Cumulative fraction');ax.legend(fontsize=7);ax.grid(alpha=.15)
        save(fig,field+'_pair_features',field.title()+': true matches versus different negative populations','Fold 0 sampled diagnostics. Exact-name negatives: full target pool. TF-IDF negatives: 100k pool. Curves are conditional, not deployment prevalence.')
        variants=['raw','case_only','nfc_only','light','compatible','accent','compact','sorted','deduped','number_format','aggressive']
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for j,pop in enumerate(['positive','tfidf_name_sample_pool']):
            for ci,country in enumerate(['India','US']):
                vals=[]
                for v in variants:
                    d=ab.filter((pl.col('field')==field)&(pl.col('population')==pop)&(pl.col('variant')==v)&(pl.col('country')==country));vals.append(100*d['exact'].sum()/max(1,d['pairs'].sum()))
                axes[j].plot(vals,range(len(variants)),marker='o',label=country,color=COLORS[ci]);
            axes[j].set_yticks(range(len(variants)),variants);axes[j].invert_yaxis();axes[j].set_xlabel('% exactly equal nonempty views');axes[j].set_title('True match agreement' if j==0 else 'TF-IDF name negative equality (100k pool)');axes[j].legend();axes[j].grid(alpha=.15)
        save(fig,field+'_ablations',field.title()+': normalization gain and collision risk','Independent optional views, evaluated on sampled fold 0 pairs. Equality is evidence, never an automatic merge rule.')
    fs=['name_jw','address_jw','name_char3_cosine','address_char3_cosine','name_word_tfidf_cosine','address_word_tfidf_cosine','name_token_set','address_numeric_jaccard','name_x_address_jw']
    for country in ['India','US']:
        pops=['random','same_country','exact_name_full_pool','first_token_sample_pool','numeric_sample_pool','tfidf_name_sample_pool','tfidf_address_sample_pool']
        grid=np.array([[diag.filter((pl.col('country')==country)&(pl.col('negative_population')==pop)&(pl.col('feature')==f))['roc_auc'][0] for pop in pops] for f in fs]);fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(grid,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto');ax.set_yticks(range(len(fs)),fs);ax.set_xticks(range(len(pops)),[x.replace('_sample_pool','').replace('_full_pool','') for x in pops],rotation=20,ha='right')
        for y in range(len(fs)):
            for x in range(len(pops)):ax.text(x,y,f'{grid[y,x]:.2f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,label='ROC AUC');save(fig,'auc_'+country,country+': easy negatives exaggerate feature quality','Descriptive single-feature AUC; sampling changes the task. This is not macro F0.5 or a fitted classifier.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for j,field in enumerate(['name','address']):
        grouped=frame.group_by(['source1_entity_id','n_matches']).agg(pl.col(field+'_jw').max().alias('max_score'))
        for singleton,label,color in [(True,'True singleton',COLORS[1]),(False,'Has true matches',COLORS[0])]:
            vals=grouped.filter((pl.col('n_matches')==0)==singleton)['max_score'];axes[j].hist(vals,bins=np.linspace(0,1,21),density=True,histtype='step',linewidth=2,label=label,color=color)
        axes[j].set_title(field.title());axes[j].set_xlabel('Maximum Jaro–Winkler across diagnostic pairs');axes[j].legend()
    save(fig,'singleton','Singleton confidence requires difficult candidate negatives','Diagnostic pair sets include sampled positives for non-singletons: optimistic separability, not a validated singleton classifier.')
    labels_audit=json.loads(Path('docs/audit_evidence/audit.json').read_text())['ground_truth']
    fig,axes=plt.subplots(1,2,figsize=(13,5));counts=labels_audit['match_counts'];axes[0].bar([r['n_matches'] for r in counts],[100*r['n']/labels_audit['integrity']['record_count'] for r in counts],color=COLORS[0]);axes[0].set_xlabel('True S2/S3 matches per S1');axes[0].set_ylabel('% of all train S1');axes[0].set_xticks(range(12))
    sd=labels_audit['source_distribution'];axes[1].bar(['Neither','S3 only','S2 only','Both'],[100*r['n']/labels_audit['integrity']['record_count'] for r in sd],color=COLORS[:4]);axes[1].set_ylabel('% of train S1');save(fig,'label_topology','Most reference entities have multiple target matches','All 2,206,821 training S1. Each labeled target has one S1 owner; S1 is not a one-to-one matching problem.')
    trans=json.loads((a.pairs/'transliteration.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for j,field in enumerate(['name','address']):
        for i,country in enumerate(['India','US']):
            rr=[r for r in trans['rates'] if r['field']==field and r['country']==country and r['population']=='positive' and r['nonascii']]
            n=sum(r['pairs'] for r in rr);vals=[100*sum(r[k] for r in rr)/max(n,1) for k in ['high_before','high_after']];axes[j].plot(['Original','ICU transliterated'],vals,marker='o',label=f'{country} (n={n})',color=COLORS[i])
        axes[j].set_title(field.title());axes[j].set_ylabel('% with Jaro–Winkler ≥ 0.90');axes[j].legend();axes[j].grid(alpha=.15)
    save(fig,'transliteration','Transliteration helps some cross-script pairs','Non-ASCII true pairs in the frozen fold-0 sample only. Negative collision rates are stored in transliteration.json; no recall or precision claim.')
    benchmark=json.loads(Path('artifacts/preprocessing/PREP-001/benchmark.json').read_text());res=[r for r in benchmark['native_results'] if len(r['operations'])==3]
    fig,ax=plt.subplots(figsize=(10,5));ax.bar([r['backend'] for r in res],[r['rows_per_second']/1000 for r in res],color=COLORS[:3]);ax.set_ylabel('Thousands of strings / second');save(fig,'throughput','Native normalization throughput on this Mac','100k strings; NFC + mark-preserving punctuation + whitespace only. Exact parity: zero mismatches. This excludes the full optional-view bundle.')
    supplementary=json.loads((a.profile/'additional_diagnostics.json').read_text());labels=[];postal=[]
    for source in supplementary['sources']:
        for row in source['by_country']:
            labels.append(source['source'].replace('_source',' S')+' '+row['country']);postal.append([100*row[key]/row['rows'] for key in ['postal_like5_rows','postal_like6_rows']])
    fig,ax=plt.subplots(figsize=(12,7));yy=np.arange(len(labels));ax.barh(yy-.18,np.array(postal)[:,0],height=.35,label='5-digit-shaped',color=COLORS[0]);ax.barh(yy+.18,np.array(postal)[:,1],height=.35,label='6-digit-shaped',color=COLORS[1]);ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlabel('% of addresses');ax.legend();save(fig,'postal_shapes','Postal-shaped numbers are too sparse for mandatory blocking','Exact full-source regex shape indicators; they do not identify or validate actual postal codes.')
    candidate=json.loads(Path('outputs/candidates/TOKEN-001/run-002/metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5));met=candidate['metrics'];keys=list(met)
    axes[0].plot([met[k]['average_candidates_per_query'] for k in keys],[100*met[k]['link_recall'] for k in keys],marker='o',color=COLORS[0]);axes[0].set_xlabel('Average candidates per S1');axes[0].set_ylabel('True-link recall (%)');axes[0].set_ylim(0,100)
    for k in keys:axes[0].annotate(k.replace('total_top_','K=').replace('route_capped_union','Union'),(met[k]['average_candidates_per_query'],100*met[k]['link_recall']),xytext=(5,5),textcoords='offset points',fontsize=8)
    routes=candidate['per_route'];axes[1].barh(list(routes),[100*v['link_recall'] for v in routes.values()],color=COLORS[1]);axes[1].set_xlabel('True-link recall (%)');axes[1].set_xlim(0,100)
    save(fig,'candidate_recall','Full-target lexical retrieval: more rescue routes are needed','1,000 balanced-country fold-0 queries; all 10.32M targets. Union recall 77.33%, not a trained-model score.')
    Path('docs/figures/phase2_manifest.json').write_text(json.dumps(gallery,indent=2))
    cards=''.join(f'<article><h2>{html.escape(g["title"])}</h2><a href="figures/{g["file"]}"><img loading="lazy" src="figures/{g["file"]}" alt="{html.escape(g["title"])}"></a><p>{html.escape(g["note"])}</p></article>' for g in gallery)
    Path('docs/PREPROCESSING_VISUAL_REPORT.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Preprocessing evidence | Amazon ML Challenge</title><style>body{font:16px system-ui;background:#edf2f8;color:#17263c;max-width:1400px;margin:35px auto;padding:0 24px}header{padding:24px;background:#132b47;color:white;border-radius:16px}h1{font-size:38px;line-height:1.1}h2{font-size:22px}article{background:white;margin:25px 0;padding:24px;border-radius:12px;border:1px solid #dbe2ec}img{width:100%;height:auto}p{line-height:1.5;color:#4f6179}header p{color:#d1e1f7}.meta{font-size:14px}input{padding:12px;width:90%;margin:20px 0;font-size:16px}</style></head><body><header><div class="meta">PHASE 2 · LOCAL COMPUTE · MEASURED EVIDENCE</div><h1>Understand the data.<br>Preserve what distinguishes businesses.</h1><p>Full profiles across 24,229,173 records, independent normalization views, difficult negatives, country shift and performance. No learned matcher or paid GPU run.</p></header><input id="filter" placeholder="Filter charts: India, name, shift, collision, Unicode…" aria-label="Filter charts"><main>'''+cards+'''</main><script>document.getElementById('filter').addEventListener('input',e=>{document.querySelectorAll('article').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))})</script></body></html>''')
    lines=['# Train–test distribution shift','','Exact full-source histograms; no hidden labels. Country-conditioned comparisons avoid composition confounding. France has no training counterpart, so no within-France train/test divergence is fabricated.','','JSD is divergence in bits (squared SciPy Jensen–Shannon distance); PSI floors each probability at 1e-6 then renormalizes; Wasserstein uses raw feature units. Binning for all three metrics is exact integer feature values. PSI depends on this choice and is descriptive, not a universal alarm threshold.','','| Source | Country | Field | Variable | JSD bits | PSI | Wasserstein |','|---|---|---|---|---:|---:|---:|']
    for r in sorted(shifts,key=lambda r:-r['js_divergence_bits']):lines.append(f"| S{r['source']} | {r['country']} | {r['field']} | {r['kind']} | {r['js_divergence_bits']:.6f} | {r['psi_floor_1e_6']:.6f} | {r['wasserstein']:.4f} |")
    lines += ['','See `TRAIN_TEST_SHIFT.csv` for machine-readable values and the visual report for country composition, script and missingness shifts. Test profiles are descriptive, not a supervised fit corpus.','', 'Reproduce: `.venv/bin/python scripts/render_preprocessing_report.py`']
    Path('docs/TRAIN_TEST_SHIFT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'figures':len(gallery),'shift_comparisons':len(shifts),'report':'docs/PREPROCESSING_VISUAL_REPORT.html'}))

if __name__=='__main__':main()
