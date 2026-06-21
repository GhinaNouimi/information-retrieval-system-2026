# مشروع نظم استرجاع المعلومات (Information Retrieval System) - 2026

## وصف المشروع

يهدف هذا المشروع إلى بناء نظام استرجاع معلومات متكامل (Information Retrieval System) باستخدام لغة Python اعتماداً على مجموعة بيانات Quora من مشروع BEIR.

تم تصميم النظام وفق مبدأ الخدمات المستقلة (Service-Oriented Architecture) بحيث يتم فصل مراحل المعالجة والفهرسة والاسترجاع والتخزين إلى خدمات مستقلة قابلة للتطوير والصيانة.

يعتمد النظام على عدة نماذج للاسترجاع والمقارنة بينها، بما في ذلك النماذج المعجمية (Lexical Retrieval)، والنماذج الدلالية (Semantic Retrieval)، والنماذج الهجينة (Hybrid Retrieval).

---

# مجموعة البيانات المستخدمة

المجموعة المستخدمة:

```text
beir/quora/test
```

إحصائيات المجموعة:

* عدد الوثائق: 522,931 وثيقة
* عدد الاستعلامات: 10,000 استعلام
* عدد علاقات الصلة (Qrels): 15,675

---

# البنية العامة للنظام

تم تقسيم النظام إلى الخدمات التالية:

```text
services/
│
├── dataset_service
├── preprocessing_service
├── indexing_service
├── retrieval_service
├── embedding_service
├── document_store_service
├── clustering_service
├── evaluation_service
└── api_service
```

---

# الخدمات المنفذة

## 1. خدمة المعالجة المسبقة (Preprocessing Service)

تقوم هذه الخدمة بتجهيز الوثائق والاستعلامات قبل استخدامها داخل نماذج البحث.

الخطوات المطبقة:

* Text Normalization
* Tokenization
* Stopword Removal
* Lemmatization

تم استخدام نفس خط المعالجة لكل من:

* الوثائق
* الاستعلامات

لضمان التوافق أثناء عملية المطابقة.

---

## 2. خدمة الفهرسة (Indexing Service)

تم تنفيذ عدة أنواع من الفهارس:

### Inverted Index

تم تنفيذه لأغراض تعليمية وفهم بنية أنظمة الاسترجاع.

### TF-IDF Index

تم إنشاء:

* TF-IDF Sample
* TF-IDF Full Dataset

باستخدام:

```python
TfidfVectorizer
```

### BM25 Index

تم إنشاء:

* BM25 Sample
* BM25 Full Dataset

باستخدام:

```python
BM25Okapi
```

كما تم توفير نسخة قابلة لضبط معاملات:

* k1
* b

لإجراء التجارب والمقارنات.

---

## 3. خدمة التضمين الدلالي (Embedding Service)

تم إنشاء تمثيلات متجهية للوثائق باستخدام نماذج Sentence Transformers.

تستخدم هذه الخدمة من أجل:

* Semantic Search
* Embedding Retrieval
* Hybrid Retrieval

---

## 4. خدمة الاسترجاع (Retrieval Service)

تحتوي الخدمة على عدة استراتيجيات استرجاع مستقلة.

### الاسترجاع باستخدام TF-IDF

* TfidfRetrievalStrategy
* TfidfFullV2RetrievalStrategy

### الاسترجاع باستخدام BM25

* BM25RetrievalStrategy
* BM25FullRetrievalStrategy
* BM25FullTunableStrategy

### الاسترجاع باستخدام Embeddings

* EmbeddingRetrievalStrategy
* EmbeddingFullRetrievalStrategy

### الاسترجاع الهجين (Hybrid Retrieval)

تم تنفيذ طريقتين:

#### Hybrid Serial Retrieval

دمج النتائج بشكل تسلسلي بين:

* BM25
* Embeddings

#### Hybrid Parallel Retrieval

دمج النتائج بشكل متوازي مع إعادة ترتيب النتائج النهائية.

---

## 5. خدمة تخزين الوثائق (Document Store Service)

تم تخزين الوثائق الأصلية داخل قاعدة بيانات:

```text
SQLite
```

بحيث يتم استرجاع النص الكامل باستخدام:

```text
doc_id
```

بعد انتهاء عملية البحث.

---

## 6. خدمة التجميع (Clustering Service)

تم تنفيذ متطلب إضافي يتمثل في تجميع نتائج البحث ضمن مجموعات موضوعية.

وظائف الخدمة:

* تحديد العنقود (Cluster) الخاص بكل وثيقة.
* إعطاء تسمية وصفية لكل عنقود باستخدام أهم الكلمات.
* تجميع نتائج البحث بحسب الموضوع.
* عرض النتائج للمستخدم على شكل مجموعات مترابطة.

مثال:

```text
Cluster 1:
stock, investment, market, trading

Cluster 2:
education, school, university
```

وبذلك تصبح نتائج البحث أكثر تنظيماً وأسهل للتصفح.

---

## 7. خدمة التقييم (Evaluation Service)

تم استخدام مقاييس التقييم القياسية لمقارنة أداء نماذج الاسترجاع المختلفة.

المقاييس المستخدمة:

* Precision@K
* Recall@K
* MAP
* MRR
* NDCG

---

# دورة عمل النظام

## مرحلة البناء (Offline Phase)

```text
تحميل البيانات

↓

معالجة الوثائق

↓

بناء الفهارس

↓

بناء الـ Embeddings

↓

حفظ النماذج والفهارس

↓

بناء بيانات Clustering

↓

تخزين الوثائق في قاعدة البيانات
```

---

## مرحلة البحث (Online Phase)

```text
استعلام المستخدم

↓

معالجة الاستعلام

↓

اختيار استراتيجية الاسترجاع

↓

تنفيذ البحث

↓

ترتيب النتائج

↓

تجميع النتائج (اختياري)

↓

استرجاع النص الأصلي

↓

عرض النتائج
```

---

# النماذج المدعومة

يدعم النظام حالياً:

* TF-IDF Retrieval
* BM25 Retrieval
* Embedding Retrieval
* Hybrid Serial Retrieval
* Hybrid Parallel Retrieval

---

# تشغيل المشروع

تفعيل البيئة:

```bash
conda activate ir_env
```

تثبيت المتطلبات:

```bash
pip install -r requirements.txt
```

تحميل موارد NLTK:

```bash
python scripts/download_nltk_resources.py
```

---

# أمثلة التشغيل

البحث باستخدام TF-IDF:

```bash
python services/retrieval_service/search_tfidf_full.py
```

البحث باستخدام BM25:

```bash
python services/retrieval_service/search_bm25_sample.py
```

البحث باستخدام Embeddings:

```bash
python services/retrieval_service/search_embedding_full.py
```

البحث باستخدام Hybrid Serial:

```bash
python services/retrieval_service/search_hybrid_serial_full.py
```

البحث باستخدام Hybrid Parallel:

```bash
python services/retrieval_service/search_hybrid_parallel_full.py
```

---

# التقنيات والمكتبات المستخدمة

* Python
* NLTK
* Scikit-Learn
* Rank-BM25
* Sentence Transformers
* NumPy
* SciPy
* SQLite
* Pickle

---



