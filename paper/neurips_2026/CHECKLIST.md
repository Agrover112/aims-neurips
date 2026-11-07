# NeurIPS 2026 paper checklist

`checklist.tex` is the unmodified checklist from the official NeurIPS 2026 formatting archive, downloaded on 2026-11-07. It contains 16 questions. Answers and justifications are still TODO; this is a template, not a completed assessment of our paper.

## Official sources

- [2026 formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
- [Checklist guidance](https://neurips.cc/public/guides/PaperChecklist)
- [2026 main-track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [2026 Evaluations & Datasets FAQ](https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ)

SHA-256 of the downloaded `checklist.tex`: `780ba13c480f652dcc42e69ed61a752ce0ea270f15d332d4a45b059dabad84f6`.

## Completing and including it

1. Review each question against the actual manuscript and released artifacts. Replace `\answerTODO{}` with `\answerYes{}`, `\answerNo{}`, or `\answerNA{}`, and replace each `\justificationTODO{}` with a short justification and relevant section references.
2. Remove the block between `%%% BEGIN INSTRUCTIONS %%%` and `%%% END INSTRUCTIONS %%%`. Keep the section heading, questions, and guidelines as directed by the template.
3. Add the following after all appendices and before `\end{document}` in the manuscript:

   ```latex
   \clearpage
   \input{checklist}
   ```

4. Rebuild the PDF and check that no TODO answers remain. The existing `neurips_2026.sty` already defines the answer macros.

The checklist has not been inserted into our manuscript automatically, so unfinished answers do not appear in the current paper PDF. Check any competition-specific submission instructions in addition to the general NeurIPS guidance.

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
