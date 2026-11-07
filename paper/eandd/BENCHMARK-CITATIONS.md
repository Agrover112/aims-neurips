# Citations for the paper?s 17 benchmark groups

Verified on 7 November 2026 using the exact benchmark metadata used by our pipeline, original papers, author repositories and official citation exports. This covers the 17 groups analyzed in the paper, not the entire evolving Measurement Data Bank catalog.

Ready-to-use records: benchmark-citations.bib. Machine-readable ID mapping: benchmark-citation-map.json. The current paper bibliography has not been rewritten.

**Availability:** verified source links were found for all 17 groups. No separate paper was identified for the exact HELM African suite: cite the shared translation paper and the suite release. The alignment-faking group has a primary technical-blog study. Some OpenReview pages require browser verification; author-linked arXiv alternatives are provided.

| Dataset ID | Original paper / study | Response-data source | BibTeX keys |
|---|---|---|---|
| `afrieval` | [MasakhaNER 1.0 and 2.0: released NER predictions](https://doi.org/10.1162/tacl_a_00416) | [Recorded release](https://github.com/masakhane-io/masakhane-ner) | `adelani-etal-2021-masakhaner`, `adelani-etal-2022-masakhaner` |
| `afrimedqa` | [AfriMed-QA](https://aclanthology.org/2025.acl-long.96/) | [Recorded release](https://github.com/intron-innovation/AfriMed-QA) | `nimo-etal-2025-afrimed` |
| `agi_elo` | [AGI-Elo](https://arxiv.org/abs/2505.12844) | [Recorded release](https://github.com/SS47816/AGI-Elo) | `sun2025agielofarmasteringtask`, `hendrycks2021measuringmassivemultitasklanguage` |
| `ai2d_test` | [AI2D_TEST](https://arxiv.org/abs/1603.07396) | [Recorded release](https://huggingface.co/datasets/VLMEval/OpenVLMRecords) | `kembhavi2016diagramworthdozenimages` |
| `algotune` | [AlgoTune](https://arxiv.org/abs/2507.15887) | [Recorded release](https://github.com/oripress/AlgoTune) | `press2025algotunelanguagemodelsspeed` |
| `alignment_faking` | [Alignment Faking (RL)](https://alignment.anthropic.com/2025/alignment-faking-mitigations/) | [Recorded release](https://huggingface.co/datasets/Anthropic/alignment-faking-rl) | `gasteiger2025alignmentfakingmitigations` |
| `appworld` | [AppWorld](https://arxiv.org/abs/2407.18901) | [Recorded release](https://github.com/StonyBrookNLP/appworld-leaderboard) | `trivedi2024appworldcontrollableworldapps` |
| `bird_sql` | [BIRD (text-to-SQL)](https://arxiv.org/abs/2305.03111) | [Recorded release](https://github.com/petavue/NL2SQL-Benchmark/tree/0637e2593739f82961d38a14d0d80df2b20e3060) | `li2023llmservedatabaseinterface` |
| `brace` | [BRACE (CAF-Score decisions)](https://arxiv.org/abs/2512.10403) | [Recorded release](https://github.com/inseong00/CAF-Score) | `guo2025bracebenchmarkrobustaudio`, `lee2026cafscorecalibratingclaplalms` |
| `bridging_gap` | [Bridging-the-Gap African Languages](https://arxiv.org/abs/2412.12417) | [Recorded release](https://github.com/InstituteforDiseaseModeling/Bridging-the-Gap-Low-Resource-African-Languages) | `alhanai2024bridginggapenhancingllm`, `hendrycks2021measuringmassivemultitasklanguage`, `sakaguchi2019winograndeadversarialwinogradschema`, `bandarkar2024belebelebenchmarkparallelreading` |
| `cmmlu` | [CMMLU](https://arxiv.org/abs/2306.09212) | [Recorded release](https://huggingface.co/datasets/opencompass/compass_academic_predictions) | `li2024cmmlumeasuringmassivemultitask` |
| `cruxeval` | [CRUXEval](https://arxiv.org/abs/2401.03065) | [Recorded release](https://github.com/facebookresearch/cruxeval) | `gu2024cruxevalbenchmarkcodereasoning` |
| `editbench` | [EDIT-Bench](https://arxiv.org/abs/2511.04486) | [Recorded release](https://github.com/waynchi/editbench/tree/b406cc5143feb91e9bda1edd01cbe37d84d185fd) | `chi2025editbenchevaluatingllmabilities` |
| `financebench` | [FinanceBench](https://arxiv.org/abs/2311.11944v1) | [Recorded release](https://github.com/patronus-ai/financebench/tree/22257c455f9f9ca9ca3dc67456d9a32e60c7ef08) | `islam2023financebenchnewbenchmarkfinancial` |
| `hallusionbench` | [HallusionBench](https://arxiv.org/abs/2310.14566) | [Recorded release](https://huggingface.co/datasets/VLMEval/OpenVLMRecords) | `guan2024hallusionbenchadvanceddiagnosticsuite` |
| `harmbench` | [HarmBench](https://arxiv.org/abs/2402.04249) | [Recorded release](https://github.com/justinphan3110cais/harmbench_website/tree/data/playground_data) | `mazeika2024harmbenchstandardizedevaluationframework` |
| `helm_afr` | [HELM African MMLU + Winogrande](https://arxiv.org/abs/2412.12417) | [Recorded release](https://crfm.stanford.edu/helm/mmlu-winogrande-afr/latest/) | `alhanai2024bridginggapenhancingllm`, `helmAfricanRelease`, `liang2023holisticevaluationlanguagemodels`, `hendrycks2021measuringmassivemultitasklanguage`, `sakaguchi2019winograndeadversarialwinogradschema` |

## Attribution details

- **afrieval:** MasakhaNER 1.0/2.0 released NER predictions, not a separate benchmark paper named AfriEval.
- **afrimedqa:** Use the published ACL record (Nimo et al., 2025); the preprint has a different author ordering.
- **agi_elo:** Cite AGI-Elo for the released MMLU predictions and MMLU for the question bank.
- **ai2d_test:** AI2D diagram questions; the released outputs come from OpenVLMRecords.
- **algotune:** The author-linked arXiv paper is provided as an accessible alternative to OpenReview.
- **alignment_faking:** This group uses the RL mitigation study and alignment-faking-rl data, not only the earlier Alignment Faking in Large Language Models paper.
- **appworld:** Retain the specific leaderboard response-release link as well as the AppWorld paper.
- **bird_sql:** BIRD task, with Petavue evaluations on a 360-question development subset.
- **brace:** Cite BRACE and CAF-Score: the stored observations are CAF-Score decisions, not solely the original BRACE author evaluation.
- **bridging_gap:** Translation/evaluation paper; the stored release contains MMLU, WinoGrande and Belebele components.
- **cmmlu:** The question bank is CMMLU; responses are the OpenCompass academic-predictions release.
- **cruxeval:** Author-released code reasoning/input-output prediction observations.
- **editbench:** EDIT-Bench for instructed code edits, not the image-inpainting EditBench.
- **financebench:** The 150-question human-evaluated open sample, not all 10,231 questions.
- **hallusionbench:** Distinguish the original question bank from the OpenVLMRecords response collection.
- **harmbench:** Retain the published playground response source as well as the HarmBench paper.
- **helm_afr:** No separate paper for this exact HELM suite identified. Cite its shared Bridging-the-Gap translation paper and the official HELM suite. The general HELM paper is supporting framework credit.

## Additional primary references

- [MasakhaNER 1.0](https://aclanthology.org/2021.tacl-1.66/) and [MasakhaNER 2.0](https://aclanthology.org/2022.emnlp-main.298/).
- [CAF-Score](https://arxiv.org/abs/2603.19615) supplies the observed BRACE decisions.
- [MMLU](https://arxiv.org/abs/2009.03300), [WinoGrande](https://arxiv.org/abs/1907.10641), [Belebele](https://arxiv.org/abs/2308.16884) and [HELM](https://arxiv.org/abs/2211.09110) are included for underlying-task/framework attribution.

## Bibliography provenance

- Author lists and bibliographic fields are official arXiv or ACL Anthology exports. The blog and exact HELM suite are explicitly identified web references.
- The AfriMed-QA ACL record is Nimo et al.; its older preprint has a different lead-author order.
- arXiv export years may follow the revised version (e.g. CMMLU and HallusionBench). These records preserve official export values. No unverified publication venue or release year was invented.
- For HELM African, retain both the shared translation-paper citation and the exact release URL; it is not an independent question bank from the related translation study.
