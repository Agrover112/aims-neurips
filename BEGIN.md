You are to design a robust, extensible starter kit for the AIMS competition, beginning with the official starting_kit.zip (present locally), the BLE baseline code, and the paiec_baseline.zip containing PAIEC Predictor Baselines. This starter kit will be submitted for review by the NeurIPS AIMS Group, whose reviewers are likely experts in statistics, experimental design, controlled randomized trials, statistical inference, and Bayesian inference. They expect scientific rigor, clarity, modularity, and adherence to competition and scientific guidelines.

Your objectives and guidelines:

1. Carefully Read Official Guidelines and Starting Kits:
  * Before any coding, thoroughly read and understand the instructions on the AIMS Competition website.
  * Pay special attention to how code must be bundled and submitted, and how the pipeline is expected to operate.
  * Use the official starting_kit.zip (present locally) as your foundation, ensuring that your starter kit is fully compatible with the provided structure and requirements.
  * Thoroughly review the PAIEC Predictor Baselines in paiec_baseline.zip (present locally). These predictors provide reference implementations for estimating the probability that an AI system will answer an item correctly, using supplied attributes and any labels acquired during evaluation. Use these as starting points for developing your own predictors, and for comparing the value of test-time adaptation.
  * Carefully study the official AIMS Foundations GitHub repository for additional resources, code structure conventions, and relevant implementations. Integrate best practices and relevant modules from this repo to ensure your starter kit is compatible and aligns with the broader AIMS community standards.
  * Follow all instructions on the CodaLab competition page for submission and evaluation protocols.
  * Keep an eye on community discussions and participants’ experiences—especially those sharing about their submissions to the AIMS Workshop at COLM this year (workshop page). These can provide useful clarifications, inspiration, or practical examples.
  * Note: A small preliminary literature review has been performed by AI and is included in the literature folder, which contains 1 or 2 relevant documents to inform and inspire further development.
2. Deeply Understand the Evaluation Methodology:
  * Carefully study the Evaluation section on the competition website, which provides an interesting and detailed description of the evaluation process, including an illustrative example. Use this as a key reference to implement the evaluation pipeline in a way that faithfully matches the competition’s intent.
  * Faithfully reproduce the competition’s evaluation strategy for offline use, since competitors cannot access the leaderboard.
  * Ensure all evaluation code is aligned with the competition’s requirements and is transparent, reproducible, and scientifically sound.
  * Clearly document and highlight any core assumptions made in implementing the evaluation, including any choices or simplifications.
3. Code Structure, Scientific Standards, and Experimentation:
  * Mirror the structure and organizational principles of the provided starting kits, baseline code, and the AIMS Foundations GitHub repository.
  * Write modular, well-documented code, avoiding convoluted or deeply nested logic. The kit does not need to be a Python package—scripts, notebooks, and modular code files are sufficient as long as they are clearly organized and easy to use or extend.
  * The kit must be extensible to allow rapid, repeated experimentation ("spamming experiments") with minimal friction.
  * Include robust support for hyperparameter tuning (e.g., configuration files or parameter sweep scripts).
  * Set up the code so that experiments can be run easily on Modal (as in your TopoBench workflow), enabling scalable and reproducible compute.
  * Integrate Weights & Biases (WandB) for experiment tracking, logging, and plotting of all relevant results. Ensure that all key metrics, parameters, and plots are automatically logged to WandB, and that plots/statistics are reproducible and exportable for reporting.
  * Experiment results—including metrics, configurations, and model artifacts—should be saved in an organized, queryable format for easy retrieval, visualization, and reporting.
4. Visualization Suite and Reporting:
  * Include a visualization module for dataset exploration and evaluation results, following best practices in statistical visualization.
  * The suite should allow for granular inspection and comparisons across experiments or hyperparameter settings.
  * Plots and summary tables should be easily exportable for inclusion in the NeurIPS report.
  * Emulate the clarity and statistical correctness of figures4papers.
5. Testing:
  * Write basic unit tests or smoke tests for each module to ensure correctness and robustness.
  * These tests should provide safety but must not violate any competition rules or guidelines.
6. Troubleshooting:
  * Include a dedicated troubleshooting section or module in the starter kit documentation.
  * Provide practical guidance for common issues (e.g., installation errors, data mismatches, evaluation failures), debugging tips, and clear error messaging.
  * Suggest strategies for diagnosing and resolving unexpected behavior in the pipeline.
7. Test-Time Adaptation:
  * Include a modular, clearly documented section or set of functions for test-time adaptation, if allowed by competition rules.
  * This should outline how models or pipeline components can be adapted or fine-tuned specifically at test time, with clear explanations and justified use cases.
  * Clearly denote any assumptions, limitations, or constraints on test-time adaptation in the documentation.
8. No Shortcuts, Full Transparency:
  * Do not take shortcuts in implementation or documentation.
  * Always highlight, in comments or documentation, any core assumptions, simplifications, or limitations in your approach.
  * Succinctly state claims and avoid overstatements.

Deliverable: A succinct, well-organized, and extensible starter kit (as a code repository or collection of scripts and notebooks) that is fully documented, modular, and scientifically rigorous. The kit must strictly follow the structure and requirements outlined in both the official starting_kit.zip and paiec_baseline.zip (present locally) and integrate best practices from the AIMS Foundations GitHub repository and the competition guidelines, making it easy to launch, track, and compare experiments—including running at scale on Modal and tracking/plotting with WandB. Troubleshooting and test-time adaptation guidance should be clearly included. The included literature folder provides a starting point for further study. All code and documentation should enable a statistically-minded reviewer to easily understand, trust, and extend your work, and should stand up to scrutiny from experts in both machine learning and statistics.

---

Let me know if you want further refinements or help with specific folder organization for scripts/notebooks!

Add that. Please note. You will not get biased by the literature search okay.

Absolutely! Here’s your revised prompt with a clear note that the AI should not be biased by the preliminary literature review—the literature folder is only for reference and context, not for restricting or biasing scientific exploration or solution design.

---

You are to design a robust, extensible starter kit for the AIMS competition, beginning with the official starting_kit.zip (present locally), the BLE baseline code, and the paiec_baseline.zip containing PAIEC Predictor Baselines. This starter kit will be submitted for review by the NeurIPS AIMS Group, whose reviewers are likely experts in statistics, experimental design, controlled randomized trials, statistical inference, and Bayesian inference. They expect scientific rigor, clarity, modularity, and adherence to competition and scientific guidelines.

Your objectives and guidelines:

1. Carefully Read Official Guidelines and Starting Kits:
  * Before any coding, thoroughly read and understand the instructions on the AIMS Competition website.
  * Pay special attention to how code must be bundled and submitted, and how the pipeline is expected to operate.
  * Use the official starting_kit.zip (present locally) as your foundation, ensuring that your starter kit is fully compatible with the provided structure and requirements.
  * Thoroughly review the PAIEC Predictor Baselines in paiec_baseline.zip (present locally). These predictors provide reference implementations for estimating the probability that an AI system will answer an item correctly, using supplied attributes and any labels acquired during evaluation. Use these as starting points for developing your own predictors, and for comparing the value of test-time adaptation.
  * Carefully study the official AIMS Foundations GitHub repository for additional resources, code structure conventions, and relevant implementations. Integrate best practices and relevant modules from this repo to ensure your starter kit is compatible and aligns with the broader AIMS community standards.
  * Follow all instructions on the CodaLab competition page for submission and evaluation protocols.
  * Keep an eye on community discussions and participants’ experiences—especially those sharing about their submissions to the AIMS Workshop at COLM this year (workshop page). These can provide useful clarifications, inspiration, or practical examples.
  * Note: A small preliminary literature review has been performed by AI and is included in the literature folder, which contains 1 or 2 relevant documents to inform and inspire further development. Please note: You will not get biased by the literature search. The literature folder is for reference and inspiration only—not for restricting or steering the scientific exploration or solution design.
2. Deeply Understand the Evaluation Methodology:
  * Carefully study the Evaluation section on the competition website, which provides an interesting and detailed description of the evaluation process, including an illustrative example. Use this as a key reference to implement the evaluation pipeline in a way that faithfully matches the competition’s intent.
  * Faithfully reproduce the competition’s evaluation strategy for offline use, since competitors cannot access the leaderboard.
  * Ensure all evaluation code is aligned with the competition’s requirements and is transparent, reproducible, and scientifically sound.
  * Clearly document and highlight any core assumptions made in implementing the evaluation, including any choices or simplifications.
3. Code Structure, Scientific Standards, and Experimentation:
  * Mirror the structure and organizational principles of the provided starting kits, baseline code, and the AIMS Foundations GitHub repository.
  * Write modular, well-documented code, avoiding convoluted or deeply nested logic. The kit does not need to be a Python package—scripts, notebooks, and modular code files are sufficient as long as they are clearly organized and easy to use or extend.
  * The kit must be extensible to allow rapid, repeated experimentation ("spamming experiments") with minimal friction.
  * Include robust support for hyperparameter tuning (e.g., configuration files or parameter sweep scripts).
  * Set up the code so that experiments can be run easily on Modal (as in your TopoBench workflow), enabling scalable and reproducible compute.
  * Integrate Weights & Biases (WandB) for experiment tracking, logging, and plotting of all relevant results. Ensure that all key metrics, parameters, and plots are automatically logged to WandB, and that plots/statistics are reproducible and exportable for reporting.
  * Experiment results—including metrics, configurations, and model artifacts—should be saved in an organized, queryable format for easy retrieval, visualization, and reporting.
4. Visualization Suite and Reporting:
  * Include a visualization module for dataset exploration and evaluation results, following best practices in statistical visualization.
  * The suite should allow for granular inspection and comparisons across experiments or hyperparameter settings.
  * Plots and summary tables should be easily exportable for inclusion in the NeurIPS report.
  * Emulate the clarity and statistical correctness of figures4papers.
5. Testing:
  * Write basic unit tests or smoke tests for each module to ensure correctness and robustness.
  * These tests should provide safety but must not violate any competition rules or guidelines.
6. Troubleshooting:
  * Include a dedicated troubleshooting section or module in the starter kit documentation.
  * Provide practical guidance for common issues (e.g., installation errors, data mismatches, evaluation failures), debugging tips, and clear error messaging.
  * Suggest strategies for diagnosing and resolving unexpected behavior in the pipeline.
7. Test-Time Adaptation:
  * Include a modular, clearly documented section or set of functions for test-time adaptation, if allowed by competition rules.
  * This should outline how models or pipeline components can be adapted or fine-tuned specifically at test time, with clear explanations and justified use cases.
  * Clearly denote any assumptions, limitations, or constraints on test-time adaptation in the documentation.
8. No Shortcuts, Full Transparency:
  * Do not take shortcuts in implementation or documentation.
  * Always highlight, in comments or documentation, any core assumptions, simplifications, or limitations in your approach.
  * Succinctly state claims and avoid overstatements.

Deliverable: A succinct, well-organized, and extensible starter kit (as a code repository or collection of scripts and notebooks) that is fully documented, modular, and scientifically rigorous. The kit must strictly follow the structure and requirements outlined in both the official starting_kit.zip and paiec_baseline.zip (present locally) and integrate best practices from the AIMS Foundations GitHub repository and the competition guidelines, making it easy to launch, track, and compare experiments—including running at scale on Modal and tracking/plotting with WandB. Troubleshooting and test-time adaptation guidance should be clearly included. The included literature folder provides a starting point for further study, but you must not let it bias your scientific exploration or design. All code and documentation should enable a statistically-minded reviewer to easily understand, trust, and extend your work, and should stand up to scrutiny from experts in both machine learning and statistics.