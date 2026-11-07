# NeurIPS 2026 paper checklist

`checklist.tex` contains the 16 official NeurIPS 2026 questions with completed answers and justifications for this paper. The instruction block has been removed, while the questions and guidelines are retained. Both `eandd.tex` and `eandd_reviewing.tex` include it after their appendices.

## Official sources

- [2026 formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
- [Checklist guidance](https://neurips.cc/public/guides/PaperChecklist)
- [2026 main-track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [2026 Evaluations & Datasets FAQ](https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ)

SHA-256 of the original blank `checklist.tex` downloaded on 2026-11-07, before answers were added: `780ba13c480f652dcc42e69ed61a752ce0ea270f15d332d4a45b059dabad84f6`.

## Maintaining the answers

1. Recheck each answer when the manuscript or released artifacts change. Answers use `\answerYes{}`, `\answerNo{}`, or `\answerNA{}`, followed by a short evidence-based justification.
2. Keep the official section heading, questions, and guidelines unchanged.
3. Both manuscripts now contain the following after their appendices and before `\end{document}`:

   ```latex
   \clearpage
   \input{checklist}
   ```

4. Rebuild the PDF and check that no TODO answers remain. The existing `neurips_2026.sty` already defines the answer macros.

Check any competition-specific submission instructions in addition to the general NeurIPS guidance.

## Current reporting gaps

The checklist answers **No** to questions 5, 8, 10, and 12:

- **Open access:** code and derived artifacts are released, but full reruns require separate authorised access to restricted competition inputs.
- **Compute reporting:** a complete per-experiment hardware, memory, runtime, and total-development-compute inventory is not reported.
- **Broader impacts:** the paper discusses transfer limitations but does not explicitly analyse both positive and negative societal impacts.
- **Asset licenses:** credits and data restrictions are documented, but a complete explicit third-party license-and-terms inventory is not provided.

These are reporting limitations, not claims that these requirements have already been satisfied. The LLM-usage answer refers to the existing Disclosure and author responsibility; it does not add an account of text editing.

## Questions

1. Claims
2. Limitations
3. Theory assumptions and proofs
4. Experimental result reproducibility
5. Open access to data and code
6. Experimental setting/details
7. Experiment statistical significance
8. Experiments compute resources
9. Code of ethics
10. Broader impacts
11. Safeguards
12. Licenses for existing assets
13. New assets
14. Crowdsourcing and research with human subjects
15. Institutional review board (IRB) approvals or equivalent for research with human subjects
16. Declaration of LLM usage
