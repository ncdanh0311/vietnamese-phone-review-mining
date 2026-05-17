# Phân cụm chủ đề và Phân tích Cảm xúc Bình luận Điện thoại Tiếng Việt

## 1. Thông tin sinh viên

| MSSV | Họ và tên | Lớp | Email |
|---|---|---|---|
|  |  |  |  |

## 2. Tổng quan dự án

Dự án áp dụng khai phá dữ liệu văn bản tiếng Việt trên bình luận điện thoại. Pipeline chính gồm tiền xử lý văn bản, biểu diễn TF-IDF, phân cụm chủ đề bằng K-Means và phân loại cảm xúc bằng LinearSVC.

## 3. Đặt vấn đề

1. K-Means có thể tự phân nhóm bình luận theo chủ đề tương đồng với nhãn aspect thực tế không?
2. SVM đạt F1 bao nhiêu khi phân loại cảm xúc Positive, Negative, Neutral?
3. Chủ đề nào nhận nhiều phản hồi tiêu cực nhất từ khách hàng?

## 4. Mô tả dataset

Dữ liệu gồm `Train.csv` và `Test.csv`, chứa bình luận điện thoại tiếng Việt. Mỗi dòng có 5 cột:

| Cột | Ý nghĩa |
|---|---|
| `index` | Mã dòng trong dữ liệu gốc |
| `comment` | Nội dung bình luận |
| `n_star` | Số sao đánh giá từ 1 đến 5 |
| `date_time` | Thời điểm bình luận |
| `label` | Nhãn aspect-based, ví dụ `{CAMERA#Positive};{BATTERY#Negative};` |

Các aspect thường gặp gồm `CAMERA`, `BATTERY`, `PRICE`, `FEATURES`, `PERFORMANCE`, `GENERAL`, `SER&ACC`, `OTHERS`. Notebook 01 và 02 sẽ thống kê số dòng, thiếu dữ liệu, phân phối cảm xúc và phân phối aspect thực tế từ dataset.

## 5. Cấu trúc project

```text
vietnamese-phone-sentiment-mining/
├── configs/config.yaml
├── datasets/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── stopwords/
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_text.ipynb
│   ├── 03_tfidf_kmeans_clustering.ipynb
│   ├── 04_svm_classification.ipynb
│   └── 05_evaluation_visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── visualization.py
├── results/
│   ├── models/
│   ├── csv/
│   ├── figures/
│   └── logs/
├── main.py
├── requirements.txt
└── README_VI.md
```

## 6. Pipeline tổng thể

```text
Raw CSV
  -> EDA
  -> Lowercase, bỏ dấu câu, chuẩn hóa số, bỏ stopwords, ViTokenizer, bỏ từ lặp
  -> Gán sentiment tổng thể và main_aspect
  -> TF-IDF
  -> K-Means clustering và so sánh với aspect
  -> LinearSVC sentiment classification
  -> Tổng hợp kết quả và insight kinh doanh
```

## 7. Luồng notebook

| Notebook | Mục đích | Output chính |
|---|---|---|
| `01_explore_dataset.ipynb` | Khám phá dữ liệu thô | Biểu đồ EDA trong `results/figures/` |
| `02_preprocessing_text.ipynb` | Làm sạch văn bản, gán sentiment và aspect | `train_clean.csv`, `test_clean.csv` |
| `03_tfidf_kmeans_clustering.ipynb` | TF-IDF, chọn K, phân cụm chủ đề | Model K-Means, vectorizer, `kmeans_clustered.csv` |
| `04_svm_classification.ipynb` | Train và đánh giá LinearSVC | `svm_model.pkl`, `svm_results.csv`, confusion matrix |
| `05_evaluation_visualization.ipynb` | Tổng hợp kết quả, insight, hạn chế | Biểu đồ tổng hợp và nhận xét cuối |

## 8. Tiền xử lý

Pipeline tiền xử lý dùng `pyvi.ViTokenizer`:

1. Lowercase để giảm trùng lặp từ do viết hoa.
2. Bỏ dấu câu và emoji để giảm nhiễu.
3. Chuẩn hóa số thành `number` để gom các biểu thức số.
4. Bỏ stopwords tiếng Việt để giữ lại từ mang nhiều thông tin.
5. Tokenize tiếng Việt bằng PyVi để nối các cụm từ như `màn_hình`.
6. Bỏ từ lặp liên tiếp để giảm nhiễu từ cách viết kéo dài.

## 9. TF-IDF

TF-IDF biểu diễn bình luận thành vector dựa trên độ quan trọng của từ trong một bình luận so với toàn bộ tập dữ liệu. Cấu hình dùng unigram và bigram, `max_features=3000`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True` để giảm ảnh hưởng của các từ xuất hiện quá nhiều lần trong cùng văn bản.

## 10. K-Means

K-Means được chạy với nhiều giá trị K từ 2 đến 10. Notebook 03 dùng Elbow Method và Silhouette Score để chọn K hợp lý. Sau khi phân cụm, mỗi cụm được gán tên theo `main_aspect` phổ biến nhất và được so sánh bằng crosstab giữa `cluster` và `main_aspect`.

## 11. SVM

Mô hình phân loại dùng `LinearSVC(C=1.0, class_weight='balanced')`. `class_weight='balanced'` giúp giảm thiên lệch khi số mẫu Positive, Negative, Neutral không đều nhau. Notebook 04 có baseline `DummyClassifier`, cross-validation 5-fold, confusion matrix và phân tích lỗi.

## 12. Kết quả kỳ vọng

| Hạng mục | Mục tiêu |
|---|---:|
| Silhouette K-Means | ≥ 0.08 |
| F1 Macro SVM | ≥ 0.75 |
| SVM so với baseline | Tốt hơn ≥ 10% |
| Cụm khớp aspect tốt | Match ratio ≥ 50% cho các cụm chính |

## 13. Hướng dẫn chạy

```bash
pip install -r requirements.txt
jupyter notebook
```

Chạy notebook theo thứ tự:

```text
01 -> 02 -> 03 -> 04 -> 05
```

Có thể chạy nhanh pipeline chính bằng:

```bash
python main.py
```

## 14. Hạn chế và hướng phát triển

Hạn chế:

- TF-IDF không nắm đầy đủ ngữ nghĩa và phủ định như `không tốt`.
- K-Means nhạy cảm với khởi tạo ngẫu nhiên và giả định cụm dạng cầu.
- Stopwords có thể thiếu từ đặc thù lĩnh vực điện thoại.
- Neutral thường khó phân loại vì ranh giới mờ với Positive và Negative.

Hướng phát triển:

- Thay TF-IDF bằng PhoBERT embedding.
- Thử LDA hoặc BERTopic cho topic modeling.
- Làm aspect-level sentiment thay vì gán một nhãn tổng thể cho cả bình luận.
- Thu thập thêm dữ liệu để cân bằng nhãn.

## 15. Kết luận

Kết luận cuối cùng cần điền sau khi chạy đủ 5 notebook. Notebook 05 sẽ tổng hợp số liệu thực tế để trả lời ba câu hỏi nghiên cứu: mức độ khớp của K-Means với aspect, F1 Macro của SVM và chủ đề có tỷ lệ phản hồi tiêu cực cao nhất.
