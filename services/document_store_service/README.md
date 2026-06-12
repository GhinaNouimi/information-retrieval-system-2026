# Document Store Service

هذه الخدمة مسؤولة عن تخزين واسترجاع الوثائق الأصلية raw documents.

حسب تحديث المعيدة، وقت وصول Query من المستخدم يجب ألا نقرأ النص الأصلي من ملفات نصية أو من ir_datasets مباشرة، بل يجب قراءة الوثيقة الأصلية من Database حسب doc_id.

لذلك نستخدم قاعدة بيانات محلية SQLite لتخزين:

- doc_id
- raw text

الفهارس مثل TF-IDF و BM25 و Embeddings يمكن أن تبقى محفوظة كملفات artifacts.

لكن عند عرض النتائج النهائية للمستخدم، نستخدم doc_id الناتج من عملية البحث ونقرأ النص الأصلي من قاعدة البيانات.

مثال:

Search Model returns:

doc_id = 1746

ثم نقرأ من قاعدة البيانات:

SELECT text FROM documents WHERE doc_id = 1746

ثم نعرض النص الأصلي للمستخدم.