# Regional-Speech: A Bangla Speech Recognition Dataset for Benchmarking Models under Dialectal Variation


## 📌 Overview

**Regional-Speech** is an extensive **Bangla Speech Recognition Dataset** that enables benchmarking **Automatic Speech Recognition (ASR)** models under regional dialectal variations. This dataset captures spontaneous speech data from various regions across **Bangladesh**, ensuring diversity in phonetics, lexicon, and prosody. This repo has all the codes and relevant files/resources behind the development of the regional speech dataset

<img src="https://github.com/user-attachments/assets/2f64606d-ebae-4dfc-beb0-740bb2d32b82" alt="District Coverage Map" width="700"/>
<p><i>Map showing the districts covered in the Regional-Speech dataset</i></p>

## 🎯 Project Goals

- Develop a **high-quality, annotated speech corpus** covering multiple Bangla dialects.
- Benchmark **ASR models** to enhance speech technology in regional Bangla.
- Create an **open-source dataset** for researchers and developers.

  ## 🗣️ Dataset Features

- **100+ hours** of validated transcribed spontaneous speech
- Natural conversations covering **12 distinct dialectal regions**
- Balanced gender representation across regions
- Diverse acoustic environments (indoor/outdoor recordings)
- Transcriptions validated by native dialect speakers and linguistic experts
- Rich metadata including regional statistics and linguistic features

<table>
  <tr>
    <td width="50%"><img src="https://github.com/user-attachments/assets/ef286d4f-7dff-4c94-86cd-0476d2d674e9" alt="Topic Distribution" width="100%"/></td>
    <td width="50%"><img src="https://github.com/user-attachments/assets/cdf2b9df-1537-45ba-b2ae-b3b450ca3e2a" alt="Gender Distribution" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><i>Topic distribution of audio recordings</i></td>
    <td align="center"><i>Gender distribution across the dataset</i></td>
  </tr>
</table>

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

## 📊 Regional Speech Corpus Statistics

| District     | Total Count | Total Duration | Avg Rec. Length | WPM   | Unique Words | OOV%  | Train Count | Train Duration | Train OOV% | Test Count | Test Duration | Test OOV% | Val Count | Val Duration | Val OOV% |
|-------------|------------|----------------|----------------|-------|--------------|-------|------------|----------------|------------|------------|----------------|------------|----------|--------------|----------|
| Rangpur     | 1,298      | 6:00:57        | 16.66          | 138.99| 10,487       | 44.21 | 1,038      | 4:48:49        | 50.81      | 130        | 0:35:50        | 45.25      | 130      | 0:36:18      | 36.11    |
| Kishoreganj | 2,049      | 9:36:52        | 16.80          | 116.74| 14,770       | 55.71 | 1,639      | 7:42:31        | 63.25      | 205        | 0:58:28        | 48.31      | 205      | 0:55:53      | 55.56    |
| Narail      | 1,859      | 8:36:51        | 16.80          | 116.90| 13,251       | 48.38 | 1,487      | 6:52:13        | 56.64      | 186        | 0:51:19        | 44.97      | 186      | 0:53:19      | 43.52    |
| Chittagong  | 1,757      | 8:11:47        | 16.56          | 129.82| 16,353       | 61.43 | 1,405      | 6:35:38        | 62.13      | 176        | 0:47:11        | 64.68      | 176      | 0:48:58      | 57.49    |
| Narsingdi   | 1,373      | 6:20:24        | 16.71          | 150.32| 12,931       | 44.93 | 1,099      | 5:03:50        | 53.95      | 137        | 0:37:34        | 39.77      | 137      | 0:39:00      | 41.08    |
| Tangail     | 1,271      | 6:13:13        | 17.00          | 146.52| 11,253       | 32.37 | 1,017      | 5:03:09        | 45.26      | 127        | 0:34:40        | 24.81      | 127      | 0:35:24      | 27.02    |
| Habiganj    | 1,170      | 5:26:21        | 16.99          | 123.96| 9,826        | 58.06 | 936        | 4:19:25        | 58.06      | 117        | 0:34:46        | 56.69      | 117      | 0:32:10      | 59.38    |
| Barishal    | 1,006      | 4:46:05        | 17.22          | 140.93| 9,056        | 48.76 | 804        | 3:47:42        | 62.02      | 101        | 0:29:16        | 48.33      | 101      | 0:29:07      | 47.24    |
| Sylhet      | 7,624      | 36:16:29       | 17.35          | 124.39| 38,359       | 61.81 | 6,100      | 28:51:42       | 60.62      | 762        | 3:42:32        | 58.21      | 762      | 3:42:15      | 60.50    |
| Sandwip     | 1,310      | 6:03:06        | 16.61          | 144.87| 10,349       | 61.09 | 1,048      | 4:48:22        | 62.72      | 131        | 0:37:14        | 54.02      | 131      | 0:37:30      | 53.48    |
| Cumilla     | 318        | 1:27:26        | 16.12          | 160.44| 4,211        | 133.53| 254        | 1:09:48        | 55.23      | 32         | 0:08:51        | 21.90      | 32       | 0:08:48      | 39.02    |
| Noakhali    | 278        | 1:16:25        | 16.88          | 106.58| 3,337        | 44.00 | 222        | 1:00:24        | 48.39      | 28         | 0:08:02        | 43.24      | 28       | 0:07:59      | 40.36    |
| **TOTAL**   | **21,313** | **100:15:57**  | -              | -     | **-**        | **-** | **17,049** | **80:13:45**   | **-**      | **2,132**  | **10:00:56**   | **-**      | **2,132** | **10:01:16** | **-**    |

**Notes:**
- **WPM (Words per Minute):** Average speech rate.
- **OOV% (Out-of-Vocabulary Percentage):** Words not in the predefined vocabulary.
- **Train/Test/Validation splits** indicate dataset partitions.


## 📊 Model Benchmarking on the Test Segments of the Dataset

### Word Error Rate (WER) & Character Error Rate (CER) Across Different Regions

| Models                | Rangpur (WER / CER) | Kishoreganj (WER / CER) | Narail (WER / CER) | Chittagong (WER / CER) | Narsingdi (WER / CER) | Tangail (WER / CER) | Habiganj (WER / CER) |
|----------------------|--------------------|----------------------|------------------|----------------------|------------------|------------------|------------------|
| **Google ASR**      | 0.872 / 0.698      | 0.525 / 0.291        | 0.946 / 0.892    | 0.913 / 0.881        | 0.936 / 0.874    | 0.920 / 0.845    | 0.933 / 0.843    |
| **YellowKing**      | 0.942 / 0.723      | 0.961 / 0.806        | 0.946 / 0.963    | 0.984 / 0.808        | 0.937 / 0.727    | 0.722 / 0.401    | 0.943 / 0.704    |
| **Hishab Conformer**| 0.827 / 0.827      | 0.900 / 0.631        | 0.829 / 0.494    | 0.974 / 0.660        | 1.276 / 0.766    | 0.931 / 0.588    | 0.901 / 0.547    |
| **Tugstugi**        | 0.917 / 0.847      | 0.977 / 0.954        | 0.958 / 0.847    | 0.987 / 0.969        | 0.979 / 0.943    | 0.482 / 0.220    | 0.860 / 0.784    |
| **Wav2Vec 2 Large** | 0.964 / 0.626      | 1.031 / 0.744        | 0.948 / 0.562    | 0.989 / 0.659        | 0.953 / 0.561    | 0.805 / 0.332    | 1.052 / 0.668    |
| **W2V2LM**         | 0.915 / 0.450      | 1.050 / 0.690        | 0.953 / 0.516    | 0.971 / 0.577        | 0.996 / 0.545    | 0.923 / 0.408    | 0.974 / 0.473    |
| **Whisper (Medium)**| 1.002 / 0.877      | 1.006 / 0.891        | 1.000 / 0.890    | 1.000 / 0.885        | 0.998 / 0.883    | 1.028 / 0.879    | -                |
| **PX12**           | 0.746 / 0.404      | 1.005 / 0.745        | 0.621 / 0.303    | 0.833 / 0.439        | 0.628 / 0.286    | 0.377 / 0.129    | 0.691 / 0.380    |
| **Wav2Vec 2 Large** | 1.000 / 0.999      | 1.000 / 0.998        | 1.000 / 0.998    | 1.000 / 0.999        | 1.000 / 0.998    | 1.000 / 0.999    | 1.000 / 0.995    |

---

### 

| Models                | Barishal (WER / CER) | Sylhet (WER / CER) | Sandwip (WER / CER) | Cumilla (WER / CER) | Noakhali (WER / CER) | **Overall Avg (WER / CER)** |
|----------------------|--------------------|------------------|------------------|------------------|------------------|----------------------|
| **Google ASR**      | 0.977 / 0.954      | 0.889 / 0.797    | 0.874 / 0.771    | 0.964 / 0.927    | 0.528 / 0.277    | 0.856 / 0.754        |
| **YellowKing**      | 0.930 / 0.697      | 0.950 / 0.703    | 0.973 / 0.780    | 0.717 / 0.400    | 0.982 / 0.773    | 0.913 / 0.682        |
| **Hishab Conformer**| 1.038 / 0.890      | 0.900 / 0.890    | 0.890 / 0.797    | 0.958 / 0.527    | 1.010 / 0.773    | 0.938 / 0.611        |
| **Tugstugi**        | 0.548 / 0.314      | 0.916 / 0.820    | 0.938 / 0.927    | 0.887 / 0.876    | 0.816 / 0.760    | 0.866 / 0.760        |
| **Wav2Vec 2 Large** | 0.960 / 0.612      | 1.034 / 0.751    | 0.995 / 0.616    | 0.784 / 0.339    | 1.020 / 0.780    | 0.968 / 0.543        |
| **W2V2LM**         | 0.974 / 0.574      | 1.101 / 0.951    | 0.970 / 0.644    | 0.910 / 0.810    | 0.980 / 0.869    | 0.968 / 0.613        |
| **Whisper (Medium)**| 1.005 / 0.911      | 1.005 / 0.898    | 1.000 / 0.998    | 1.143 / 0.846    | -                | 1.020 / 0.884        |
| **PX12**           | 0.792 / 0.444      | 1.000 / 0.529    | 0.602 / 0.489    | 0.958 / 0.390    | -                | 0.725 / 0.417        |
| **Wav2Vec 2 Large** | 1.000 / 0.999      | 1.000 / 0.999    | 1.000 / 0.998    | 1.000 / 0.999    | 1.000 / 0.999    | 1.000 / 0.998        |

---

### 📌 Notes:
- **WER (Word Error Rate)** measures the percentage of words incorrectly transcribed.
- **CER (Character Error Rate)** measures the percentage of characters incorrectly transcribed.
- **Lower WER/CER values indicate better ASR model performance.**

---



1️⃣ **Clone the repository**
```bash
git clone https://github.com/BengaliAI/RegSpeech.git
cd RegSpeech
.........


....
