# Midpoint Paper Outline and Rubric Map

**Working title:** *Terrain-Aware Semantic Segmentation for Autonomous Mars Rover Driving: CNNs vs. Transformers vs. Foundation Models*

**Research question:** For four-class Mars terrain segmentation, how do convolutional, transformer, and foundation-model systems compare; how much does pretrained initialization help; and how well does a Curiosity-trained system generalize to Opportunity/Spirit imagery?

This outline follows the course rubric in `RESEARCH.MD`. The midpoint paper intentionally separates completed design and prototype work from confirmatory evidence that still requires the preregistered three-seed experiment set.

## Paper structure

1. **Abstract — near complete**
   - State the autonomous-driving motivation and the controlled architecture/transfer/generalization gap.
   - Identify AI4Mars, the four terrain classes, the model families, and the primary metric.
   - Summarize the available descriptive results while clearly deferring confirmatory claims.

2. **Introduction — near complete**
   - Explain why terrain perception constrains autonomous rover mobility and science return.
   - Define the applied machine-learning task and the practical importance of per-class performance.
   - State the research question, project goals, and intended contributions.

3. **Related Work — near complete**
   - Cover AI4Mars, SPOC, and Mars-specific semantic segmentation.
   - Contrast U-Net and DeepLabV3+ convolutional inductive biases with SegFormer attention.
   - Position SAM and DINOv3 as foundation-model references, explicitly distinguishing peer-reviewed work from the DINOv3 research preprint.

4. **Research Project Problem — near complete**
   - Formalize four-class dense semantic segmentation and the ignore label.
   - Define the dataset/camera scope and cross-rover domain shift.
   - State H0--H5 and their preregistered decision rules.

5. **Method — near complete**
   - Describe dataset indexing, by-image splits, preprocessing, augmentation, and leakage controls.
   - Describe majority, Tiny U-Net, U-Net, DeepLabV3+, SegFormer, DINOv3-SAT, and the SAM region-oracle upper bound.
   - Define weighted cross-entropy plus Dice loss, mIoU, per-class IoU, pixel accuracy, and boundary-F1.
   - Document reproducibility manifests and the paired seed/image bootstrap with Holm correction.

6. **Experimental Progress and Design — partially populated**
   - Inventory the 17 completed full-GPU runs and the missing seed/configuration work.
   - Show representative validation learning curves.
   - Show an expert-test rover image, label, and predictions from five model families.

7. **Statistical Analysis and Results — descriptive results plus confirmatory placeholders**
   - Report available-seed mIoU means, seed variation, and representative per-class IoU.
   - Show every completed seed-level result without treating ranges as confidence intervals.
   - Interpret the rare big-rock error pattern and mixed preliminary pretraining directions.
   - Reserve a table for effect sizes, 90% confidence intervals, raw and Holm-adjusted p-values, and H0--H5 verdicts.
   - Defer formal decisions until seeds 1414, 1415, and 1416 are complete under the sealed protocol.

8. **Discussion and Limitations — preliminary interpretation plus final placeholder**
   - Interpret current architecture, transfer, and rare-class patterns cautiously.
   - Reserve cross-rover and confirmatory interpretation for the completed protocol.
   - Discuss label noise, camera/domain scope, compute constraints, and the non-deployable nature of SAM's oracle assignment.

9. **Conclusion — structured midpoint placeholder**
   - Answer the research question only after the confirmatory evidence is available.
   - Connect supported findings to autonomous rover drivability and future work.

10. **References — active and sufficient at midpoint**
    - Use peer-reviewed conference/journal papers and official NASA/Zenodo dataset sources.
    - Label research preprints accurately.

## Rubric mapping

| Course rubric category | Paper evidence | Midpoint evidence/status |
|---|---|---|
| **INTRO — topic, goals, approach, hypotheses, ML problem, analysis** | Abstract; Introduction; Research Project Problem | Near-complete prose defines the applied segmentation problem, scope, goals, H0--H5, and planned analysis. |
| **HYPOTHESES & METHOD — sound algorithm/math/code, demonstrated understanding, block diagram, measures** | Research Project Problem; Method; pipeline diagram; prototype notebook | Near-complete mathematical task/loss/metric definitions plus a runnable dataset/model/loss/metric demonstration. |
| **RESEARCH — related work and merit** | Related Work; References | Near-complete positioning against Mars perception, CNN, transformer, and foundation-model literature. |
| **APPLICATION — empirical runs, metrics, statistics, problem/solution association** | Method; Experimental Progress; Statistical Analysis and Results | Seventeen full-GPU runs, seed-level plots, learning curves, qualitative predictions, and descriptive metrics are included; confirmatory tests remain reserved for complete three-seed evidence. |
| **WHAT IS LEARNED? — supported conclusions and motivation** | Discussion; Limitations; Conclusion | Preliminary observations identify strong broad-terrain performance and persistent big-rock difficulty, while formal conclusions remain deferred. |

## Midpoint submission checklist

- [x] Paper outline follows the research project and rubric.
- [x] Abstract, Introduction, Related Work, Research Project Problem, and Method are drafted close to completion.
- [x] A bounded Jupyter prototype runs on staged AI4Mars data and existing project code.
- [x] Available results and scientific figures are included and explicitly labeled descriptive.
- [x] Remaining cross-rover and confirmatory statistical evidence locations are explicit placeholders.
- [x] References include multiple reputable, relevant sources.
- [x] The dataset and model weights are excluded; the public release and download procedure are cited.
