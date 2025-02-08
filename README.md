# Regional-Speech: A Bangla Speech Recognition Dataset for Benchmarking Models under Dialectal Variation


## 📌 Overview

**Regional-Speech** is an extensive **Bangla Speech Recognition Dataset** that enables benchmarking **Automatic Speech Recognition (ASR)** models under regional dialectal variations. This dataset captures spontaneous speech data from various regions across **Bangladesh**, ensuring diversity in phonetics, lexicon, and prosody. This repo has all the codes and relevant files/resources behind the development of the regional speech dataset 

## 🎯 Project Goal

- Develop a **high-quality, annotated speech corpus** covering multiple Bangla dialects.
- Benchmark **ASR models** to enhance speech technology in regional Bangla.
- Create an **open-source dataset** for researchers and developers.

---

## 📂 Repo Structure

The repository is organized as follows:


### 🔍 File Descriptions:
- **`data/`** - Contains raw and processed speech data categorized into train, test, and validation.
- **`transcripts/`** - CSV files mapping speech samples to text transcripts.
- **`models/`** - Baseline ASR models trained using the dataset.
- **`notebooks/`** - Jupyter Notebooks for data exploration, visualization, and benchmarking.
- **`scripts/`** - Python scripts for preprocessing, training, and evaluation.
- **`docs/`** - Detailed documentation, guidelines, and analysis reports.

---

## 📊 Dataset Statistics

### 🗂 Data Collection & Validation Summary

| District      | Collected Hours | Validated Hours | Speech Samples |
|--------------|----------------|----------------|----------------|
| Rangpur      | 6:06:14         | 6:00:57        | 1320           |
| Kishoreganj  | 10:08:36        | 9:36:51        | 2087           |
| Narail       | 8:41:20         | 8:36:52        | 1880           |
| Chittagong   | 8:16:34         | 8:11:47        | 1774           |
| Narsingdi    | 6:22:10         | 6:20:25        | 1385           |
| Sylhet       | 39:50:10        | 36:16:29       | 7695           |
| Noakhali     | 20:19:41        | 1:16:25        | 4297           |
| **Total**    | **145:17:16**   | **101:52:18**  | **30063**      |

(*More details in the full documentation.*)

---
### 🛠 Steps to Use:

1️⃣ **Clone the repository**
```bash
git clone https://github.com/BengaliAI/RegSpeech.git
cd RegSpeech
