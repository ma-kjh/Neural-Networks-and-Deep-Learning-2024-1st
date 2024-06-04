# [Re] Negative Label Guided OOD Detection with Pretrained Vision-Language Models

## Reproducibility Summary

Unlike traditional visual out-of-distribution (OoD) detection methods, Vision-Language Models (VLMs) have the capability to leverage multimodal information. However, the potential of such multimodal information in VLM-based OoD detection remains largely untapped. In this report, we aim to reproduce the results reported by NegLabel, which propose a method to more effectively leverage the knowledge embedded in VLMs for OoD detection. Our objective is to evaluate whether this methodology, which leverages VLMs knowledge, can exhibit generalizability beyond the benchmark datasets utilized in their study, extending to more challenging benchmarks. To achieve this, we employ the MOS benchmark, one of the most widely used in recent studies, as well as the more challenging OpenOOD v1.5 benchmark for our reproduction efforts. We observed the generalizability of in leveraging the capabilities of VLMs knowledge through reproducing experiments and conducting extended benchmark dataset experiments.

### MOS 
<img width="884" alt="Screenshot 2024-06-05 at 00 46 34" src="https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/28b2aabd-97b9-491e-8ba7-092a3cc3fa40">
<img width="884" alt="Screenshot 2024-06-05 at 00 46 45" src="https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/a78564f6-60a6-4f39-8b58-8de075b23845">

### OpenOOD v1.5
<img width="884" alt="Screenshot 2024-06-05 at 00 47 12" src="https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/18126681-82db-41f5-aabd-8f34a482b861">
<img width="884" alt="Screenshot 2024-06-05 at 00 47 28" src="https://github.com/ma-kjh/Neural-Networks-and-Deep-Learning-2024-1st/assets/132344612/71e9bd5c-a327-4bdf-903a-1b9853c517ed">
