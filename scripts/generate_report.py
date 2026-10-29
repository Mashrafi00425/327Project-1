#!/usr/bin/env python3
"""Create printable PDF report using existing measured benchmark_results.json.
Requires reportlab and matplotlib. Report is NOT an imaginary result template.
"""
from pathlib import Path
import json
from datetime import date
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak, Image, KeepTogether,
                                Preformatted, HRFlowable)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'
OUT = ROOT/'report'/'FP_Growth_CUDA_Project_Report.pdf'
B = json.loads((RESULTS/'benchmark_results.json').read_text())
records = B['experiments']

NAVY = colors.HexColor('#18304C')
TEAL = colors.HexColor('#007B88')
SLATE = colors.HexColor('#44556A')
LGRAY = colors.HexColor('#EEF2F6')
GRID = colors.HexColor('#D7E0E8')

plt.rcParams.update({'font.size': 10,'axes.spines.right':False,'axes.spines.top':False})
labels = ['WDBC\n40%','WDBC\n50%','Digits\n70%','Digits\n80%',
          'Digits x30\n70%','Digits x30\n80%', 'Retail 50k\n3%','Retail 50k\n5%','Retail 50k\n7%']
fig, ax = plt.subplots(figsize=(9.8,3.25),dpi=170)
bars=ax.bar(range(len(records)),[r['patterns'] for r in records],color=['#007B88']*2+['#2D5F91']*4+['#C47B37']*3,width=.67)
ax.set_yscale('log'); ax.set_ylabel('Frequent itemsets (log scale)'); ax.set_xticks(range(len(records)),labels)
ax.grid(axis='y',alpha=.2)
for b,r in zip(bars,records): ax.text(b.get_x()+b.get_width()/2,r['patterns']*1.12,f"{r['patterns']:,}",ha='center',fontsize=8)
fig.tight_layout(pad=1.4); fig.savefig(RESULTS/'figure_patterns.png',bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(9.8,3.25),dpi=170)
for typ, idx, color in [('UCI Wisconsin',[0,1],'#007B88'),('UCI digits',[2,3,4,5],'#2D5F91'),('Synthetic retail',[6,7,8],'#C47B37')]:
    ax.scatter([records[i]['runtime_median_s'] for i in idx],
               [records[i]['peak_rss_median_mib'] for i in idx],
               s=74,color=color,label=typ,alpha=.9)
for i,r in enumerate(records):
    ax.annotate(labels[i].replace('\n',' '),(r['runtime_median_s'],r['peak_rss_median_mib']),
                xytext=(5,5),textcoords='offset points',fontsize=7)
ax.set_xlabel('Median elapsed execution time (seconds)'); ax.set_ylabel('Median peak RAM (MiB)')
ax.grid(alpha=.18); ax.legend(loc='upper left',fontsize=8)
ax.set_xlim(0,max(r['runtime_median_s'] for r in records)*1.28)
ax.set_ylim(0,max(r['peak_rss_median_mib'] for r in records)*1.18)
fig.tight_layout(pad=1.4); fig.savefig(RESULTS/'figure_runtime_memory.png',bbox_inches='tight'); plt.close(fig)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CoverTitleX',fontName='Helvetica-Bold',fontSize=25,leading=30,
                          textColor=NAVY,spaceAfter=16))
styles.add(ParagraphStyle(name='CoverSubX',fontName='Helvetica',fontSize=12,leading=18,
                          textColor=SLATE,spaceAfter=18))
styles.add(ParagraphStyle(name='SectionX',fontName='Helvetica-Bold',fontSize=15.0,leading=18,
                          textColor=NAVY,spaceBefore=10,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='SubX',fontName='Helvetica-Bold',fontSize=10.8,leading=14,
                          textColor=TEAL,spaceBefore=8,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.5,leading=14,
                          textColor=colors.HexColor('#1B2733'),spaceAfter=8))
styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=8.3,leading=11.2,
                          textColor=SLATE,spaceAfter=7))
styles.add(ParagraphStyle(name='CellX',fontName='Helvetica',fontSize=8.0,leading=11,
                          textColor=NAVY))
styles.add(ParagraphStyle(name='HeadCellX',fontName='Helvetica-Bold',fontSize=7.7,leading=10,
                          textColor=colors.white))
styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=8.15,leading=12.3,
                          leftIndent=7,rightIndent=4,backColor=LGRAY,spaceAfter=8,spaceBefore=5))
styles.add(ParagraphStyle(name='ReferenceX',fontName='Helvetica',fontSize=7.7,leading=11,
                          textColor=SLATE,spaceAfter=6))


def P(text, kind='BodyX'): return Paragraph(text,styles[kind])
def bullet(text): return P('<font color="#007B88"><b>•</b></font>  '+text)
def heading(s): return P(s,'SectionX')
def sub(s): return P(s,'SubX')
def make_table(rows,widths=None,header=True,small=False):
    stylestr = 'HeadCellX' if header else 'CellX'
    rendered=[]
    for ri,row in enumerate(rows):
        rendered.append([P(str(cell), 'HeadCellX' if ri==0 and header else 'CellX') for cell in row])
    tbl = Table(rendered,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7 if not small else 4),
              ('TOPPADDING',(0,0),(-1,-1),7 if not small else 4),
              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),6),
              ('LINEBELOW',(0,-1),(-1,-1),.4,GRID)]
    if header:
        commands.extend([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LGRAY])])
    else: commands.append(('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,LGRAY]))
    tbl.setStyle(TableStyle(commands))
    return tbl


def footer(canvas,doc):
    canvas.saveState()
    W,H=letter
    canvas.setStrokeColor(GRID)
    canvas.line(0.68*inch,H-0.52*inch,W-0.68*inch,H-0.52*inch)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(SLATE)
    canvas.drawString(.7*inch,H-.43*inch,'FP-GROWTH  /  CUDA-ACCELERATED FREQUENT ITEMSET MINING')
    canvas.line(0.68*inch,.52*inch,W-0.68*inch,.52*inch)
    canvas.drawString(.7*inch,.37*inch,'C++17 + CUDA | measured CPU experiments | 28 Oct 2026')
    canvas.drawRightString(W-.7*inch,.37*inch,f'Page {doc.page}')
    canvas.restoreState()

content=[]
content += [Spacer(1,.42*inch),P('CUDA-Accelerated FP-Growth<br/>Frequent Itemset Mining','CoverTitleX'),
            P('GPU weighted-support kernels in recursive FP-Growth: design, validation, and benchmarks','CoverSubX'),
            HRFlowable(width='100%',thickness=2,color=TEAL,spaceBefore=8,spaceAfter=18),
            P('<b>Student:</b> Mashrafi Rodan<br/><b>Course project:</b> Frequent Itemset Mining<br/>'
              '<b>Institution:</b> Stony Brook University<br/><b>Prepared:</b> October 28, 2026','BodyX'),
            Spacer(1,.2*inch),heading('Abstract'),
            P('This project implements CUDA-accelerated FP-Growth in C++17 to enumerate all frequent itemsets above a user-defined '
              'minimum support. Transactions are compressed into a prefix-sharing FP-tree, and patterns are generated '
              'recursively from conditional pattern bases rather than by exhaustive candidate generation. '
              'The implementation outputs support counts, offers memory-conscious summary-only operation, '
              'and exports machine-readable CSV/JSON summaries. The CUDA variant computes weighted support histograms on the GPU for root and all recursive conditional FP-trees. C++ constructs and traverses tree structures.'),
            P('Correctness is checked with a hand-worked five-basket dataset and an independent exhaustive '
              'enumerator on 100 randomized cases. Experiments use two locally available copies of UCI datasets '
              '(handwritten digits and Wisconsin Diagnostic Breast Cancer), a clearly labeled replicated scaling '
              'version of the digits sample, and separately generated synthetic retail transactions. '
              'The CPU execution times and memory usages in this report are measured Linux values, not estimates. CUDA host integration passed 34 simulator comparisons; real nvcc and GPU performance tests remain pending.'),
            Spacer(1,.1*inch),
            make_table([
                ['Implementation','Validation','Experimental scale'],
                ['C++17 + recursive GPU histograms','7 CPU tests, 34 sim checks','569-53,910 transactions'],
                ['CSV/JSON exact support output','100 CPU exhaustive oracle checks','Measured CPU 5-run medians'],
            ],[165,170,174]),
            Spacer(1,.22*inch),P('<b>Validation status.</b> The CPU baseline and simulated CUDA '
              'host interface have been exercised, but this environment has no NVIDIA GPU or nvcc. '
              'The device kernel and GPU timings must be tested on lab hardware. '
              'This is a CUDA/C++ <b>hybrid</b>, not a fully GPU-resident FP-tree.','SmallX'),PageBreak()]

content += [heading('1. Problem definition and support'),
            P('Given a transaction database <i>D</i> with <i>N</i> rows, each transaction is a set of items. '
              'An itemset <i>X</i> is frequent when <b>support(X) = |{T in D : X is a subset of T}| '
              '&gt;= min_count</b>. Fractional support <i>s</i> is converted to an absolute threshold '
              '<b>ceil(s N)</b>. The application accepts an absolute count such as <font name="Courier">3</font>, '
              'a fraction such as <font name="Courier">0.7</font>, or <font name="Courier">70%</font>.'),
            P('The output is the <i>complete</i> collection of qualifying non-empty itemsets, each with its exact '
              'transaction support. An item repeated twice within one row still counts only once. Dataset class labels '
              'are deliberately excluded, because this project is unsupervised pattern mining, not classification.'),
            heading('2. Implementation method'),
            sub('2.1 Input normalization and frequent 1-items'),
            P('The program reads whitespace-separated item tokens, maps each distinct token to an integer ID, '
              'sorts and deduplicates every transaction, and counts singleton frequencies. Items below the '
              'minimum threshold are removed. Frequent items are then ordered by descending singleton support '
              '(with a deterministic ID-based tie rule).'),
            sub('2.2 Construction of the FP-tree'),
            P('A root node represents the empty prefix. For each filtered, frequency-sorted transaction, the '
              'program walks existing child links where possible and increments their counters; when no child exists, '
              'it creates a new node. A header table stores each item’s overall support and pointers to all '
              'tree nodes bearing that item. Identical prefixes share nodes and counters.'),
            sub('2.3 Recursive conditional mining'),
            P('Items in each tree are visited from lowest support to highest support. Each visited item is '
              'appended to the current suffix and emitted with its header support. The conditional pattern base '
              'is built by tracing the ancestors of every header-linked node; each ancestor path is weighted '
              'by that node’s count. A new conditional FP-tree removes items falling below support, and '
              'the procedure recurses until no frequent extension remains. This ordering prevents duplicate patterns.'),
            P('<b>Core pseudocode</b>','SubX'),
            Preformatted('''mine(tree, suffix):
    for item in reverse(tree.frequent_items_by_support):
        pattern = suffix + [item]
        emit(pattern, tree.support[item])
        paths = []
        for node in tree.header[item]:
            paths.append((ancestors(node), node.count))
        conditional = build_fp_tree(paths, minimum_support)
        if conditional contains frequent items:
            mine(conditional, pattern)''',styles['CodeX']),
            P('The tree recursively decomposes the dataset into smaller projected problems. As in FP-Growth '
              'generally, the number of output patterns can still be exponential in the number of distinct items; '
              'no algorithm can avoid the cost of explicitly reporting exponentially many frequent subsets.','SmallX'),
            PageBreak()]

baskets=[['T1','bread, milk'],['T2','bread, diaper, beer, eggs'],['T3','milk, diaper, beer, cola'],
         ['T4','bread, milk, diaper, beer'],['T5','bread, milk, diaper, cola']]
itemsets=[['bread','4'],['milk','4'],['diaper','4'],['beer','3'],['bread, milk','3'],
          ['bread, diaper','3'],['milk, diaper','3'],['beer, diaper','3']]
content += [heading('3. Small illustrative example: exact validation'),
            P('The five-transaction market-basket example below was encoded directly as '
              '<font name="Courier">data/toy_transactions.txt</font>. With minimum support count 3 '
              '(60% of the five transactions), the four singleton patterns and four pairs shown '
              'are the <b>only</b> frequent non-empty patterns.'),
            make_table([['ID','Items in transaction']]+baskets,[58,451],small=True),
            Spacer(1,8),sub('3.1 The complete expected result'),
            make_table([['Frequent itemset','Support count']]+itemsets,[315,194],small=True),
            Spacer(1,8),sub('3.2 Worked conditional-pattern-base trace'),
            P('For <b>beer</b> (support 3), the paths preceding its occurrences in the global FP-tree are '
              '<font name="Courier">[bread, diaper] (1)</font>, '
              '<font name="Courier">[milk, diaper] (1)</font>, and '
              '<font name="Courier">[bread, milk, diaper] (1)</font>. '
              'Within these weighted prefix paths, <b>diaper</b> occurs 3 times, while bread and milk '
              'each occur only 2 times. At threshold 3, the conditional tree contains diaper only. '
              'FP-Growth therefore emits {beer, diaper} with support 3, and no other pair containing beer.'),
            P('Independent Python combination-counting produces the same eight itemsets and the same exact '
              'support values. A three-item combination does not reach support 3 in these five baskets.'),
            PageBreak()]

content += [heading('4. Testing and validation'),
            P('The test suite runs the compiled C++ binary and independently computes every subset and its '
              'count in Python. The comparison is between complete itemset-to-support dictionaries, not just '
              'a few manually inspected outputs.'),
            make_table([
                ['Automated check','What was verified'],
                ['Five-basket example','All eight patterns and exact supports at count >= 3'],
                ['Duplicate items in a row','Repeated occurrences do not inflate support'],
                ['Disjoint / no valid pattern','Correct singleton-only or empty outputs'],
                ['Absolute, fractional, %','Equivalent thresholds produce identical results'],
                ['Dense / identical rows','All qualifying subsets emitted once'],
                ['100 randomized data sets','C++ results exactly match independent enumeration'],
                ['Repeated dataset','All support counts scale by factor 3; itemsets unchanged'],
            ],[187,322],small=True),
            Spacer(1,12),
            P('<b>Observed outcome:</b> all seven test groups passed, including the 100 random generated '
              'instances (random seed 217). The random tests varied 2–8 distinct items, 3–18 transactions, '
              'and the minimum support; each compared complete dictionaries and rejected duplicate emissions.'),
            sub('4.1 Additional real-data consistency check'),
            P('The 1,797-row UCI-derived handwritten-digits copy was replicated exactly 30 times to form '
              '53,910 transactions. At 70% support, both datasets produced 3,457 distinct patterns; '
              'at 80%, both produced 481. The corresponding absolute minimum supports were '
              '1,258 and 37,737 (70%) and 1,438 and 43,128 (80%), respectively. '
              'Replication is a scaling exercise and not a claim of 53,910 independent UCI observations.'),
            sub('4.2 Reproducibility'),
            Preformatted('''make
make test
python3 scripts/benchmark.py --repeats 5
./fp_growth data/toy_transactions.txt 3 \\
    --output results/toy_patterns.csv \\
    --summary results/toy_summary.json''',styles['CodeX']),
            P('Also completed: 34 full itemset/support comparisons between the CPU binary and the '
              'CUDA-enabled host control path using a CPU histogram simulator '
              '(four bundled datasets plus 30 randomized datasets). The simulator does not compile or '
              'execute the actual CUDA device kernel; real-device verification is still required. '
              'The Python scripts are only test/benchmark drivers, not mining implementations.','SmallX'),PageBreak()]

rows=[['Dataset','Rows','Support','Patterns','Time (s)','Peak RAM (MiB)']]
for r in records:
    name=r['dataset'].replace(' (median bins)','').replace(' (nonzero)','').replace(' (replicated)','')
    rows.append([name,f'{r["transactions"]:,}',r['support'],f'{r["patterns"]:,}',
                 f'{r["runtime_median_s"]:.4f}',f'{r["peak_rss_median_mib"]:.2f}'])
content += [heading('5. Benchmark results: executed CPU experiments'),
            P('All rows below are <b>measured</b>, with support thresholds and outputs taken from the included '
              '<font name="Courier">results/benchmark_results.csv</font>. '
              'Runtime is the median over five fresh processes, including input loading and mining. '
              'Peak RAM is the median of Linux process VmHWM readings, converted from KiB to MiB. '
              'Frequent-pattern counts and supports are exact. CPU times may vary on the lab computers.'),
            make_table(rows,[175,62,55,66,73,78],small=True),
            Spacer(1,8),Image(str(RESULTS/'figure_patterns.png'),width=7.05*inch,height=2.35*inch),
            P('<b>Figure 1.</b> Pattern count versus support for each independent experiment '
              '(log-scaled y-axis; repeated digits rows are explicitly labeled).','SmallX'),
            PageBreak()]

content += [heading('6. Performance discussion'),
            Image(str(RESULTS/'figure_runtime_memory.png'),width=7.05*inch,height=2.35*inch),
            P('<b>Figure 2.</b> CPU time and peak resident memory for the nine measured scenarios. '
              'The same hardware was used for all nine experiments.','SmallX'),
            sub('6.1 Effect of minimum support'),
            P('The digits test-set copy yields 3,457 patterns at 70% but 481 at 80%; the number of output '
              'patterns therefore decreases as the support threshold rises. The breast-cancer copy drops '
              'from 283 patterns at 40% to 30 at 50%. On the synthetic retail stress test, increasing '
              'support from 3% to 7% reduces output patterns from 265 to 60 and drops median peak resident '
              'memory from approximately 68 MiB to 30 MiB in these observed runs.'),
            sub('6.2 Scaling and dataset density'),
            P('The exact 30x repeated digits experiment retains the same frequent itemset identities '
              'at a fixed relative support, but the 53,910-row input increases parsing and tree-building '
              'work. The 50,000-row independent synthetic retail set has a different, sparsely selected '
              'item universe and uses more memory at low support because many conditional trees are constructed. '
              'Comparing runtimes between different datasets is not a controlled speed-up claim: row lengths, '
              'support levels, repeated prefixes, and item distributions all influence runtime.'),
            sub('6.3 Memory management and complexity'),
            P('Conditional trees are built recursively and automatically released on return via C++ '
              '<font name="Courier">unique_ptr</font>. The program does not retain every mined pattern in '
              'memory: full CSV results are streamed, and only a fixed-size top-K preview is retained. '
              'The reported RAM nevertheless includes input transactions, FP-tree nodes, header vectors, '
              'temporary weighted conditional paths, and runtime allocator overhead.'),
            P('Initial singleton counting is linear in the number of item occurrences; per-row sorting adds '
              'sorting cost. Building and mining conditional FP-trees is data-dependent, and enumerating all '
              'frequent subsets requires exponential output size in the worst case. '
              'Peak RAM is not the sum of nodes allocated over the entire run; the summary reports '
              'both cumulative allocation and actual high-water memory.','SmallX'),PageBreak()]

content += [heading('7. CUDA design: GPU counting at every tree level'),
            P('The CUDA build performs <b>weighted 64-bit support histogramming on the GPU at '
              'every recursive FP-tree level</b>. This is a substantial change from a root-only '
              'histogram prototype. A <font name="Courier">SupportCounter</font> shared by the '
              'root tree and conditional-tree constructors flattens each group of weighted '
              'pattern-base paths into two arrays: item IDs and the path multiplicities. '
              'The device kernel uses one thread per flattened occurrence and '
              '<font name="Courier">atomicAdd</font> to increment the item bucket by its weight.'),
            sub('7.1 Division of computation'),
            make_table([
                ['CUDA device (GPU)','C++ host (CPU)'],
                ['Root item supports','Parse input and remove duplicate row tokens'],
                ['All conditional-tree item supports','Construct FP-tree nodes and headers'],
                ['Parallel weighted integer atomic adds','Build weighted conditional pattern bases'],
                ['64-bit histogram bucket accumulation','Recursively mine and emit unique patterns'],
                ['Device-to-host histogram transfers','Write CSV/JSON and measure overall elapsed time'],
            ],[250,259],small=True),
            Spacer(1,9),
            P('For an item i in a conditional pattern base P, '
              '<b>support(i) = sum of w(p), over all paths p in P containing i</b>. '
              'Using the original FP-tree node count as each path weight preserves exact conditional '
              'supports. The histogram result feeds standard FP-Growth filtering and tree construction. '
              'The host controls recursion because tree topologies are pointer-heavy and irregular; '
              'the GPU accelerates the regular, data-parallel support-count operation.'),
            sub('7.2 GPU dataflow (repeated at EVERY conditional level)'),
            Preformatted('''flatten(weighted_pattern_paths) -> ids[], weights[]
copy ids[] and weights[] from CPU to CUDA device
zero the device histogram
CUDA kernel: atomicAdd(histogram[ids[j]], weights[j])
copy support histogram back to host
CPU: filter frequent items, build conditional FP-tree
CPU: recursively mine extensions''',styles['CodeX']),
            P('GPU arrays are reallocated only if larger capacity is needed. '
              'The summary stores the GPU name, number of histogram kernel calls, number of '
              'weighted entries processed, and the maximum CUDA buffer allocation, which is '
              '<b>not</b> the total device memory usage including CUDA context.'),
            PageBreak()]

content += [heading('8. Validation status and lab CUDA benchmarks'),
            sub('8.1 Exact CUDA verification: next lab session'),
            Preformatted('''make && make cuda
./fp_growth_cuda data/toy_transactions.txt 3 --no-output
python3 scripts/verify_cuda.py --random-cases 20
python3 scripts/benchmark_cuda.py --repeats 3''',styles['CodeX']),
            P('The CUDA verification script independently runs CPU and GPU builds, compares '
              'all emitted itemset-to-support dictionaries, checks GPU kernel-call counters, '
              'and independently exhaustively checks 20 generated small datasets. '
              'The benchmark compares CPU and CUDA on the same hardware with alternating '
              'process order, three timing replicates, and CPU/GPU peak memory reporting. '
              'It saves real measurements in <font name="Courier">results/cuda_benchmark_results.csv</font>. '
              '<b>CUDA device measurements have not yet been run.</b>'),
            make_table([
                ['Already tested (host-only)','Pending verification on NVIDIA GPU'],
                ['All 7 CPU test groups passed','Compile the .cu file with nvcc'],
                ['CPU: 100 random exhaustive cases passed','Compare every CUDA itemset and support'],
                ['34 CPU vs GPU-path emulator runs matched','Run 20+ random GPU equality checks'],
                ['9 datasets/thresholds benchmarked on CPU','Record GPU times and CUDA buffer memory'],
            ],[254,255],small=True),
            Spacer(1,10),
            sub('8.2 Tradeoffs, limitations, and future improvements'),
            P('Each conditional histogram triggers host-to-device transfers and a GPU kernel launch. '
              'When conditional databases are small, transfer/launch overhead can dominate, so '
              'CUDA may be <i>slower</i> than the CPU despite parallel atomic additions. '
              'The hybrid miner still constructs and walks irregular tree structures on the CPU; '
              'it is not a completely GPU-resident implementation. A possible optimization is to '
              'batch multiple conditional supports into fewer launches.'),
            P('The optional UCI Mushroom (8,124 rows), Chess (3,196), and Connect-4 (67,557) '
              'datasets can be downloaded with <font name="Courier">scripts/get_uci.py</font>; '
              'these originals have not been benchmarked here. Existing measured inputs are '
              'UCI-derived Digits and Wisconsin cancer copies, replicated Digits for scaling, '
              'and a separately labeled synthetic retail dataset. Dense data at low support '
              'can generate exponentially many itemsets regardless of GPU throughput.'),
            heading('9. Conclusion'),
            P('The project includes an exact C++ FP-Growth reference miner plus a CUDA hybrid '
              'that uses the GPU for weighted singleton histograms throughout root and recursive '
              'conditional FP-tree mining, not only at preprocessing. CPU and host-simulated '
              'CUDA paths passed tests; lab CUDA device compilation, correctness, and performance '
              'measurement remain necessary before claiming hardware-specific results.'),
            PageBreak(),
            heading('10. Lab GPU measurement worksheet'),
            P('Complete this worksheet from the real lab GPU benchmark output. The numbers below are '
              'deliberately blank because there was no CUDA device during project preparation. '
              'Copy actual data from <font name="Courier">results/cuda_benchmark_results.csv</font> '
              'before final printing. A CPU/GPU speedup less than 1 means CUDA was slower.'),
            make_table([
                ['Dataset', 'Min support', 'CPU (s)', 'CUDA (s)', 'CPU/CUDA'],
                ['Toy baskets', '3', '______', '______', '______'],
                ['WDBC', '50%', '______', '______', '______'],
                ['Digits', '80%', '______', '______', '______'],
                ['Digits', '70%', '______', '______', '______'],
                ['Synthetic retail', '7%', '______', '______', '______'],
            ],[154,89,89,89,88],small=True),
            Spacer(1,9),
            P('<b>GPU model:</b> __________________________  '
              '<b>CUDA toolkit:</b> __________________________'),
            P('<b>CPU peak RAM:</b> __________ MiB   <b>CUDA peak RAM:</b> __________ MiB   '
              '<b>CUDA buffer allocation:</b> __________ MiB'),
            P('<b>Verification:</b> All real CPU-GPU itemset/support comparisons: '
              'PASS / FAIL (circle after testing).'),
            Spacer(1,10),
            heading('References and dataset attribution'),
            P('1. Han, J., Pei, J., &amp; Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation. '
              '<i>ACM SIGMOD</i>. https://doi.org/10.1145/342009.335372','ReferenceX'),
            P('2. UCI Machine Learning Repository, Optical Recognition of Handwritten Digits, dataset 80. '
              'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','ReferenceX'),
            P('3. UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), dataset 17. '
              'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic','ReferenceX'),
            P('4. scikit-learn documentation: load_digits and load_breast_cancer (bundled UCI-derived data copies). '
              'https://scikit-learn.org/stable/modules/datasets/toy_dataset.html','ReferenceX'),
            P('5. UCI Machine Learning Repository: Mushroom (#73), Chess King-Rook vs King-Pawn (#22), '
              'Connect-4 (#26). https://archive.ics.uci.edu/','ReferenceX'),
            Spacer(1,8),P('<b>Measured CPU hardware:</b> '+B['environment']['cpu_model']+'; '
                 +B['environment']['platform']+'; '+B['environment']['compiler']+'. '
                 'Times and RAM values reflect this execution environment and are not fixed algorithm constants.','SmallX')]

OUT.parent.mkdir(parents=True,exist_ok=True)
doc = SimpleDocTemplate(str(OUT),pagesize=letter,leftMargin=.73*inch,rightMargin=.73*inch,
                        topMargin=.79*inch,bottomMargin=.74*inch,title='CUDA-accelerated FP-Growth for Frequent Itemset Mining',
                        author='Mashrafi Rodan',subject='C++17 CUDA-hybrid FP-Growth project report')
doc.build(content,onFirstPage=footer,onLaterPages=footer)
print(OUT)
