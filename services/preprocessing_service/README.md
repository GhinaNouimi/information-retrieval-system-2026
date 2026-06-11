# Preprocessing Service

هذه الخدمة مسؤولة عن معالجة النصوص قبل استخدامها في الفهرسة أو البحث.

الهدف من هذه المرحلة هو تحويل النصوص إلى شكل موحد يسهل على النظام مقارنتها واسترجاع الوثائق المناسبة.

تطبق هذه الخدمة نفس خطوات المعالجة على Documents و Queries لضمان التوافق بينهما.

الخطوات المستخدمة حالياً:

1. Lowercasing
تحويل جميع الأحرف إلى أحرف صغيرة.

مثال:

Stock → stock

2. Punctuation Removal
إزالة الرموز غير المهمة مثل:

? ! , . ; :

3. Extra Spaces Removal
إزالة المسافات الزائدة وتوحيد شكل النص.

4. Tokenization
تقسيم النص إلى كلمات منفصلة.

مثال:

best way to invest

↓

["best", "way", "to", "invest"]

5. Stopwords Removal
إزالة الكلمات الشائعة التي لا تضيف معنى مهماً للبحث.

مثل:

the
is
are
what
how

6. Lemmatization
تحويل الكلمات إلى شكلها الأساسي مع الحفاظ على معناها.

مثال:

investing → invest

questions → question

سبب اختيار Lemmatization:

تم اختيار Lemmatization بدلاً من Stemming لأن Dataset Quora تحتوي على أسئلة قصيرة، ونرغب بالحفاظ على المعنى اللغوي للكلمات قدر الإمكان.

مثال:

historical

قد تصبح مع Stemming:

histor

بينما مع Lemmatization تبقى أقرب للشكل الصحيح لغوياً.

الخرج النهائي للخدمة:

Raw Text
↓

Processed Tokens

مثال:

What is the best way to invest in stocks?

↓

["best", "way", "invest", "stock"]