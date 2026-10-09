<h1 align="center">Trident</h1>

<p align="center">
  <b>Mitigating Modality Preference in Mixed-Modality Retrieval</b><br>
  Official repository for <i>"Chaos in the Text: Revealing the Modality Preference in Mixed-Modality Retrievers"</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/OpenBMB/Trident"><img src="https://img.shields.io/badge/code-coming%20soon-blue.svg" alt="Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-TBD-lightgrey.svg" alt="License"></a>
</p>

> 🚧 **This repository is under construction.** Code, data, and model checkpoints will be released soon. Star / watch the repo to get notified.

## Overview

Dense retrievers perform well on text and image corpora separately, but real-world corpora often mix text-only, image-only, and fused text-image documents. We find that retrieval performance is highly sensitive to modality composition:

- **Chaos in the Text.** Irrelevant text hurts more than an equal number of irrelevant images, and retrievers often rank irrelevant text above relevant images.
- **Modality preference.** Text representations receive systematically higher similarity scores than image representations.

To mitigate this bias, we propose **Trident**, which builds text, image, and fused text-image views of each document as co-equal positives, and jointly optimizes relevance discrimination and positive-view balance via Multi-Positive View InfoNCE. Trident improves mixed-modality retrieval on both CLIP-based and VLM-based architectures, reduces sensitivity to modality composition and text distractors, and improves average single-modality retrieval performance.

## Release Plan

- [ ] Paper on arXiv
- [ ] Training code
- [ ] Evaluation code and mixed-modality benchmark construction scripts
- [ ] Model checkpoints
- [ ] Usage examples

## Repository Structure

```
Trident/
├── README.md
└── (to be added)
```

## Citation

If you find this work useful, please cite:

```bibtex
coming soon
```

## Contact

For questions, please open an issue or contact the corresponding authors: Yukun Yan and Zhenghao Liu.