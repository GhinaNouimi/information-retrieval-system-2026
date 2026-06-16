
### services/retrieval_service/strategies/bm25_large_strategy.py

**نوع:** ملف جديد

**الموقع:**
```
services/retrieval_service/strategies/bm25_large_strategy.py
```

**ماذا يفعل:**
Strategy للبحث باستخدام BM25 المبني على 50,000 وثيقة.

مطابق لـ `bm25_strategy.py` الموجود لكنه يقرأ من `BM25_LARGE_DIR` بدلاً من `BM25_SAMPLE_DIR`.

---

### services/retrieval_service/strategies/embedding_strategy.py

**نوع:** ملف جديد

**الموقع:**
```
services/retrieval_service/strategies/embedding_strategy.py
```

**ماذا يفعل:**
Strategy للبحث باستخدام Embedding.

**طريقة عملها:**
1. يحوّل الـ Query إلى متجه باستخدام نفس النموذج
2. يحسب Cosine Similarity بين متجه الـ Query وكل متجهات الوثائق
3. يُرجع أعلى top_k نتيجة

**لماذا Cosine Similarity:**
لأن المتجهات تمثّل اتجاهاً في فضاء المعنى — التشابه في الاتجاه يعني التشابه في المعنى.

---

### services/retrieval_service/search_embedding_sample.py

**نوع:** ملف جديد

**الموقع:**
```
services/retrieval_service/search_embedding_sample.py
```

**ماذا يفعل:**
Demo runner لتجربة البحث بالـ Embedding بسرعة.

**طريقة التشغيل:**
```bash
python -m services.retrieval_service.search_embedding_sample
```

---

## Hybrid Retrieval

### services/retrieval_service/strategies/hybrid_serial_strategy.py

**نوع:** ملف جديد

**الموقع:**
```
services/retrieval_service/strategies/hybrid_serial_strategy.py
```

**ماذا يفعل:**
يطبّق Hybrid Retrieval بالطريقة التسلسلية (Serial).

**طريقة العمل:**
```
الخطوة 1: BM25 يسترجع أفضل 100 وثيقة مرشحة
الخطوة 2: Embedding يعيد ترتيب هذه الـ 100 وثيقة فقط
الخطوة 3: نأخذ أفضل 10 من النتائج المعاد ترتيبها
```

**ميزتها:**
أسرع من Embedding الكامل لأنه لا يحسب التشابه مع 50,000 وثيقة بل مع 100 فقط.

---
### services/retrieval_service/strategies/hybrid_parallel_strategy.py

**نوع:** ملف جديد

**الموقع:**
```
services/retrieval_service/strategies/hybrid_parallel_strategy.py
```

**ماذا يفعل:**
يطبّق Hybrid Retrieval بالطريقة المتوازية (Parallel) باستخدام Reciprocal Rank Fusion.

**طريقة العمل:**
```
BM25 يبحث ←→ Embedding يبحث  (في نفس الوقت)
      ↓
دمج النتائج باستخدام RRF
      ↓
النتائج النهائية
```

**ما هو RRF:**
لكل وثيقة يحسب:
```
score = 1/(60 + rank_bm25) + 1/(60 + rank_embedding)
```
الوثيقة التي تظهر في مراتب عالية في كلا النموذجين تحصل على أعلى درجة.

---