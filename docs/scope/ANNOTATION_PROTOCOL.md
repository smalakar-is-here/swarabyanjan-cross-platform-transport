# E27 source-guideline audit: annotation protocol

Approved by the researcher on 2026-11-28. This audit labels a probability sample of BanglaClick items under the **BanglaBait** definition of clickbait, so that source-construct performance can be estimated on social platforms (and on non-anchor news) without assuming that the news link transports.

## 1. Source of the guideline

The guideline below reproduces what the BanglaBait paper states, and nothing more:

> Mahtab, Haque, Hasan and Sadeque. *BanglaBait: Semi-Supervised Adversarial Approach for Clickbait Detection on Bangla Clickbait Dataset.* RANLP 2023. arXiv:2311.06204. A local copy and its SHA-256 are in the private store under `raw_external/BanglaBait/paper/`.

What the paper says about its own annotation (§3.2, Table 1):

- **Definition:** "Intentionally luring readers to click on a particular content by exploiting their curiosity defines a title as clickbait."
- **Annotators:** three annotators with an MA in Bangla Linguistics. Before labelling, they studied English clickbait datasets (Webis 2016/2017, Agrawal 2016).
- **Categories:** eight clickbait categories, adapted from Biyani et al. (2016), with "wrong" replaced by "question".
- **Blinding:** publisher and source were hidden from the annotators.
- **Label and aggregation:** binary label (1 = clickbait, 0 = not), decided by majority of three.
- **Agreement:** reported Fleiss κ = 0.62.

The paper does not publish a longer written guideline. Annotators must use only the definition and the categories below. They must not use the BanglaClick rubric (FAST criteria, lure triggers).

| Category | Definition (BanglaBait Table 1) | Example (the paper's English translation) |
|---|---|---|
| Questions | Titles pose a query that compels the reader to click to get the answer. | "Salman's actress Rombha mysteriously disappeared, what is she up to?" |
| Inflammatory | Titles evoke strong emotion. | "Lost his cool, kicked, then busted out the stamps before abusing the umpire!" |
| Curiosity gap / Teasing | Titles leave the reader in the dark, which tempts them to click. | "Explore how to preserve meat without a refrigerator!" |
| Ambiguous | Imprecise or unclear titles that pique interest. | "Not even Mir spared Nusrat" |
| Exaggerate | Titles overstating what is written on the landing page. | "Three foods when combined with blackberries, could kill you!" |
| Graphic | Salacious, unsettling, or implausible subject matter. | (the paper's example is a violent crime headline) |
| Formatting | Excessive use of punctuation or other symbols. | "Faria 'out' Porimoni 'in'!" |
| Bait & Switch | Overpromising titles with under-delivering content; requires additional clicks. | (see paper) |

**Decision rule:** label **1** if the text intentionally lures the reader to click by exploiting curiosity, as in any category above. Otherwise label **0**. A title that is merely low in quality (unattributed, vague about its source, informal in tone) without a curiosity lure is **0**. That is the BanglaBait criterion, not the BanglaClick one.

**Information access:** annotators see the text only, as BanglaBait's model input does. "Exaggerate" and "Bait & Switch" refer to the landing page, which is not available here; judge them from the text alone and record "no landing page" in the notes if it matters.

## 2. Sample (scripts/scope_audit_sample.py, seed 20261128)

| Block | Items | Design |
|---|---|---|
| Facebook | 100 | stratified by BanglaClick level L0–L3, proportional, ≥2 per stratum |
| Twitter | 100 | same |
| YouTube (BanglaClick) | 100 | same |
| News, non-anchor | 100 | same, from the 400 news items whose title is not an anchor |
| Gold (anchors) | 50 | stratified by level; BanglaBait label known; hidden among the others |

The sample has 450 rows and 449 distinct titles. Inclusion probabilities π = n_h/N_h are in `results/scope_v1/E27/audit_design.csv`. The gold labels and the mapping to rows are in the private key only.

## 3. Procedure

1. **Annotators:** three people. They should be fluent native Bengali readers, ideally with a linguistics or journalism background, and must not be involved in the modelling. They must not see model outputs, BanglaClick levels, platform names or the key.
2. **Training:** each annotator reads §1 and then labels a practice set of 20 BanglaBait **training** headlines with known labels, chosen by the coordinator and outside the audit sample. Discuss disagreements with the published labels. Repeat once if agreement on the practice set is below 70%.
3. **Labelling:** independently, in the order of the annotator's own sheet (`sheet_annotator_k.csv`; each sheet has a different random order). Fill in:
   - `label_0_1` (required);
   - `category` (one of the eight, or `none`);
   - `confidence_1_3` (1 low, 3 high);
   - `notes` (optional).
   Do not consult the other annotators, the internet or the source of the item.
4. **Aggregation:** the label is the majority of three, as in BanglaBait. All three individual labels are kept.
5. **Quality control (computed by E27, not by annotators):**
   - Fleiss κ among the three annotators;
   - Cohen's κ of the majority label with the BanglaBait label on the 50 gold items. This is the key check that the new team reproduces the source construct.
   - Results are reported whatever their values. If gold κ is far below the BanglaBait team's own agreement, the audit measures a third reading of the construct, and the paper must say so.
6. **Distribution and return:**
   - `python scripts/scope_audit_packs.py build` writes one zip per annotator to `private_data/scope/annotation/distribution/` (SHA-256 values in `MANIFEST.json`). Each zip holds an annotator-facing guideline (§1 and §3 only, with no sample design, block names or mention of gold items), the practice workbook and the annotator's own main workbook. The workbooks have dropdown validation and lock the order, ID and text columns.
   - Returned workbooks go in `private_data/scope/annotation/returned/`, unchanged in name. `python scripts/scope_audit_packs.py intake` scores the practice sheets against the practice key and converts each main workbook to `sheet_annotator_{k}.csv`. It refuses a workbook whose rows were re-ordered, whose text was changed, or that has missing or invalid labels.
   - Do not edit `AUDIT_KEY_PRIVATE.csv`.

## 4. Analysis (E27, `python -m scope.run E27`)

- **Per block and per anchor-blind scorer** (TF-IDF, BanglaBERT, XLM-R): source-construct AUROC, log loss and Brier score by the difference estimator (paper Eq. 8). The working link is the score-conditional link fitted on all anchors. Design-based 95% intervals use stratified-SRS variance.
- **Comparisons:**
  - against each scorer's held-out source AUROC;
  - between platforms;
  - against the identified sets and the γ-sensitivity range.
- **Measurement-error check:** on the gold items, E27 reports Cohen's κ and the confusion table of the new majority label against the BanglaBait label. The paper interprets the audit estimates in the light of this agreement; no automatic correction is applied.

## 5. Data handling and ethics

- **Restricted text:** item text comes from third-party corpora (BanglaBait CC BY-NC 4.0; BanglaClick per its release). The sheets stay in `private_data/` and go to the annotators for this research labelling only. If any annotator is outside the research team, confirm the data terms with the researcher before sending.
- **What is published:** no text is committed or published. The repository holds the text-free design and the aggregate results.
- **Ethics:** annotators are adult volunteers or paid assistants labelling public headlines. No personal data about annotators is stored beyond an anonymous ID. If the institution requires ethics review for annotation work, obtain it before labelling starts.
