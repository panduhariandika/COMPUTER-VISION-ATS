# COMPUTER-VISION-ATS
ATS
# EMNIST Character Classification
## HOG + SVM + LOOCV

> Computer Vision — ATS Midterm Project

---

## 🎬 Video Penjelasan

[![YouTube](https://img.shields.io/badge/YouTube-Video%20Penjelasan-red?style=for-the-badge&logo=youtube)](LINK_YOUTUBE_DI_SINI)

**Link Video:** `[https://youtu.be/ECFl7MbbPLo ]`
> Contoh: https://www.youtube.com/watch?v=XXXXXXXXXXX

---

## 👤 Identitas

| Field | Detail |
|-------|--------|
| **Nama** | Pandu Hariandika Pratama |
| **NIM** | 4222301076 |
| **Mata Kuliah** | Computer Vision |
| **Dataset** | EMNIST Letters (A–Z, 26 kelas) |

---

## 📌 Deskripsi Project

Project ini mengimplementasikan pipeline klasifikasi karakter tulisan tangan menggunakan dataset **EMNIST Letters**. Pipeline terdiri dari tiga tahap utama:

1. **Feature Extraction** — HOG (Histogram of Oriented Gradients)
2. **Classification** — SVM (Support Vector Machine) + Grid Search
3. **Evaluation** — LOOCV (Leave-One-Out Cross Validation)

---

## 📁 Struktur Folder

```
emnist-hog-svm/
├── emist/
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── emnist-letters-mapping.txt
├── ATS_Rapi.ipynb       ← Notebook utama
├── README.md            ← File ini
└── requirements.txt     ← Daftar library
```

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

| Library | Fungsi |
|---------|--------|
| numpy | Operasi array & matriks |
| pandas | Load & manipulasi CSV |
| scikit-learn | SVM, Grid Search, LOOCV, metrics |
| scikit-image | Ekstraksi fitur HOG |
| matplotlib | Visualisasi gambar & grafik |
| seaborn | Confusion matrix heatmap |
| mlxtend | Tambahan tools ML |

---

## 📊 Dataset

| Parameter | Nilai |
|-----------|-------|
| Nama | EMNIST Letters |
| Jumlah Kelas | 26 (A–Z) |
| Sampel per Kelas | 100 (balanced) |
| **Total Sampel Digunakan** | **2.600** |
| Ukuran Gambar | 28 × 28 pixel |
| Format | CSV (785 kolom) |
| Sumber | [Kaggle EMNIST](https://www.kaggle.com/datasets/crawford/emnist) |

> **Catatan:** Dataset asli ±145.600 sampel. Digunakan 2600 sampel agar LOOCV berjalan dalam waktu wajar.

---

## 🔄 Pipeline Program

### Step 1 — Dataset Preparation
- Load CSV → kolom 0 = label, kolom 1–784 = pixel
- Adjust label: 1–26 → 0-based (0–25)
- **Shuffle** dataset sebelum sampling
- Ambil **100 sampel/kelas** × 26 kelas = 2600 total
- Split: **80% Train** (2080) | **20% Test** (520) — stratified

```python
data_labels = data_labels - 1                    # adjust label
shuffle_idx = np.random.permutation(len(data))   # shuffle
X_train, X_test, y_train, y_test = train_test_split(
    X_hog, y, test_size=0.2, stratify=y)
```

### Step 2 — Feature Extraction (HOG)

| Parameter | Default | **Dipakai** | Alasan |
|-----------|---------|-------------|--------|
| orientations | 9 | **8** | Lebih general |
| pixels_per_cell | (8,8) | **(4,4)** | Detail lebih tinggi |
| cells_per_block | (3,3) | **(2,2)** | Normalisasi lebih lokal |
| block_norm | L2-Hys | **L2** | Sederhana & efektif |

```python
HOG_PARAMS = dict(orientations=8, pixels_per_cell=(4,4),
                  cells_per_block=(2,2), block_norm='L2')
X_hog = extract_hog_features(X, **HOG_PARAMS)
# X_hog.shape = (2600, 288)
```

### Step 3 — SVM + Grid Search

| Parameter | Nilai Dicoba | **Terbaik** |
|-----------|-------------|-------------|
| kernel | linear, rbf, poly | **rbf** |
| C | 0.1, 1, 10 | **10** |
| gamma | scale, auto | **scale** |

```python
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf', 'poly'],
              'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### Step 4 — Evaluasi LOOCV

> **Leave-One-Out Cross Validation:** 1 sampel = test, 2599 sisanya = train. Diulang 2600×.

```python
loo        = LeaveOneOut()
y_loo_pred = cross_val_predict(SVC(**best_params),
                               X_hog, y, cv=loo, n_jobs=-1)
```

---

## 📈 Hasil Evaluasi

| Metrik | Train (80%) | Test (20%) | LOOCV |
|--------|------------|-----------|-------|
| Accuracy | ~99%+ | ~85% | ~84% |
| Precision | ~99%+ | ~85% | ~84% |
| Recall | ~99%+ | ~85% | ~84% |
| F1-Score | ~99%+ | ~85% | ~84% |

**Interpretasi:**
- Train tinggi → model berhasil belajar dari data
- Train >> Test → overfitting ringan, wajar untuk SVM+RBF
- **LOOCV ≈ Test (~84%)** → model generalisasi dengan baik

---

## ▶️ Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/[USERNAME]/emnist-hog-svm.git
cd emnist-hog-svm

# 2. Download dataset dari Kaggle dan letakkan di folder emist/
# https://www.kaggle.com/datasets/crawford/emnist

# 3. Install dependency
pip install -r requirements.txt

# 4. Jalankan notebook
jupyter notebook ATS_Rapi.ipynb
# Klik 'Run All' untuk menjalankan semua cell
```

---

## ⚠️ Catatan Penting

- LOOCV membutuhkan waktu sekitar **±2 jam** (melatih model 2600×)
- File CSV harus berada di folder **`emist/`** (subfolder)
- Gunakan **Python 3.8+**
- `n_jobs=-1` memanfaatkan semua core CPU

---

*Pandu Hariandika Pratama • NIM: 4222301076 • Computer Vision • Politeknik Negeri Batam • 2025*
