# دليل الملفات المنشأة والمعدّلة

هذا الملف يشرح كل ملف تم إنشاؤه أو تعديله خلال مراحل تطوير المشروع.

---

## Evaluation Service

### services/evaluation_service/metrics.py

**نوع:** ملف جديد

**الموقع:**
```
services/evaluation_service/metrics.py
```

**ماذا يفعل:**
يحتوي على دوال حساب مقاييس التقييم الأربعة بشكل مستقل.

**الدوال الموجودة:**

`precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)`
- تحسب من أول K نتيجة كم منها صحيحة
- مثال: أرجع 10 نتائج، 3 منها صحيحة → 3/10 = 0.30

`recall(retrieved_doc_ids, relevant_doc_ids)`
- تحسب من كل الوثائق الصحيحة الموجودة كم وجد النظام
- مثال: يوجد 5 وثائق صحيحة، وجدنا 2 → 2/5 = 0.40

`average_precision(retrieved_doc_ids, relevant_doc_ids)`
- تُستخدم لحساب MAP لاحقاً
- تحسب دقة النظام عند كل نقطة يجد فيها وثيقة صحيحة

`ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)`
- تقيس جودة الترتيب
- الوثيقة الصحيحة في المرتبة الأولى تُحسب أكثر من نفس الوثيقة في المرتبة العاشرة

---

### services/evaluation_service/evaluator.py

**نوع:** ملف جديد

**الموقع:**
```
services/evaluation_service/evaluator.py
```

**ماذا يفعل:**
يربط كل شيء معاً — يقرأ الـ Queries والـ Qrels من الـ Dataset ويشغّل التقييم على أي Strategy.

**الدوال الموجودة:**

`load_qrels()`
- يقرأ الـ Qrels من Dataset
- يُرجع Dictionary: `{query_id: {doc_id_1, doc_id_2, ...}}`

`load_queries()`
- يقرأ الـ Queries من Dataset
- يُرجع Dictionary: `{query_id: "نص الاستعلام"}`

`evaluate(strategy, top_k)`
- يأخذ أي Strategy تملك دالة `search()`
- يشغّل البحث على كل Query
- يحسب المقاييس الأربعة
- يُرجع Dictionary بالمتوسطات النهائية

---

## BM25 Large

### scripts/build_bm25_index_large.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/build_bm25_index_large.py
```

**ماذا يفعل:**
يبني فهرس BM25 على أول 50,000 وثيقة من الـ Dataset.

**الخطوات التي ينفذها:**
1. يحمّل الـ Dataset
2. يقرأ أول 50,000 وثيقة
3. يطبّق `preprocess_document()` على كل وثيقة
4. يبني BM25 index
5. يحفظ الملفات في `artifacts/bm25_large/`

**الملفات الناتجة:**
```
artifacts/bm25_large/
├── bm25_index.pkl
└── bm25_doc_ids.pkl
```

**طريقة التشغيل:**
```bash
python -m scripts.build_bm25_index_large
```

---


## Document Database Large

### scripts/build_document_database_large.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/build_document_database_large.py
```

**ماذا يفعل:**
يوسّع قاعدة البيانات لتشمل أول 50,000 وثيقة بدلاً من 10,000.

**لماذا أنشأناه:**
عند تشغيل Embedding على 50,000 وثيقة كانت النتائج تُظهر `Text: None` لأن قاعدة البيانات تحتوي فقط على 10,000 وثيقة.

**طريقة التشغيل:**
```bash
python -m scripts.build_document_database_large
```

---

## Embedding

### scripts/build_embedding_index_large.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/build_embedding_index_large.py
```

**ماذا يفعل:**
يبني Embedding index على أول 50,000 وثيقة باستخدام نموذج `all-MiniLM-L6-v2`.

**الخطوات التي ينفذها:**
1. يحمّل نموذج Sentence-BERT
2. يقرأ أول 50,000 وثيقة من الـ Dataset
3. يحوّل كل وثيقة إلى متجه (vector) بحجم 384
4. يحفظ المتجهات في `artifacts/embedding_large/`

**الملفات الناتجة:**
```
artifacts/embedding_large/
├── embeddings_matrix.npy    ← مصفوفة شكلها (50000, 384)
└── embedding_doc_ids.pkl
```

**طريقة التشغيل:**
```bash
python -m scripts.build_embedding_index_large
```

**ملاحظة:**
في المرة الأولى سيحمّل النموذج من الإنترنت (~90MB).
في المرات التالية يستخدم النسخة المحفوظة محلياً.

---






## Evaluation Scripts

### scripts/run_evaluation.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/run_evaluation.py
```

**ماذا يفعل:**
يقيّم TF-IDF وBM25 (على 10,000 وثيقة) ويقارن بينهما.

**طريقة التشغيل:**
```bash
python -m scripts.run_evaluation
```

---

### scripts/run_evaluation_large.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/run_evaluation_large.py
```

**ماذا يفعل:**
يقارن BM25 على 10,000 وثيقة مقابل BM25 على 50,000 وثيقة ويُظهر نسبة التحسن.

**طريقة التشغيل:**
```bash
python -m scripts.run_evaluation_large
```

---

### scripts/run_evaluation_embedding.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/run_evaluation_embedding.py
```

**ماذا يفعل:**
يقارن BM25 Large مع Embedding ويُظهر أيهما أفضل في كل مقياس.

**طريقة التشغيل:**
```bash
python -m scripts.run_evaluation_embedding
```

---

### scripts/run_evaluation_hybrid.py

**نوع:** ملف جديد

**الموقع:**
```
scripts/run_evaluation_hybrid.py
```

**ماذا يفعل:**
يقيّم جميع النماذج الأربعة معاً ويعرض جدول مقارنة نهائي.

**النماذج التي يقيّمها:**
1. BM25 Large
2. Embedding
3. Hybrid Serial
4. Hybrid Parallel

**طريقة التشغيل:**
```bash
python -m scripts.run_evaluation_hybrid
```

---

## ترتيب التشغيل الكامل عند البدء من الصفر

```bash
# 1. تفعيل البيئة
conda activate ir_env

# 2. بناء قاعدة البيانات الكبيرة
python -m scripts.build_document_database_large

# 3. بناء فهرس BM25 الكبير
python -m scripts.build_bm25_index_large

# 4. بناء فهرس Embedding
python -m scripts.build_embedding_index_large

# 5. تجربة البحث
python -m services.retrieval_service.search_embedding_sample

# 6. تقييم كل النماذج
python -m scripts.run_evaluation_hybrid
```

---

## النتائج النهائية للتقييم

```
Metric           TF-IDF    BM25    Embedding   Serial   Parallel
Precision@10     0.0049    0.0203    0.0230     0.0225   0.0220
Recall           0.0199    0.0977    0.1033     0.1023   0.1018
MAP              0.0181    0.0871    0.0971     0.0965   0.0930
nDCG@10          0.0214    0.0982    0.1083     0.1075   0.1045
```

الترتيب من الأفضل للأضعف:
```
Embedding > Hybrid Serial > Hybrid Parallel > BM25 > TF-IDF
```

ملاحظة: TF-IDF مبني على 10,000 وثيقة بينما البقية على 50,000 وثيقة.