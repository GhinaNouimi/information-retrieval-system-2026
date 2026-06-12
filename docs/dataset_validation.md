# Dataset Validation
في بداية المشروع كان علينا اختيار 
 Dataset
 مناسبة تحقق شروط المشروع.


بعد البحث وتجربة أكثر من 
Dataset
 قمنا باختيار:

beir/quora/test

ثم قمنا بالتأكد من أنها تحقق جميع الشروط المطلوبة.

نتائج الفحص:

* عدد الوثائق (Documents): 522,931
* عدد الاستعلامات (Queries): 10,000
* عدد الـ Qrels: 15,675

كما تأكدنا أن:

* تحتوي على Documents
* تحتوي على Queries
* تحتوي على Qrels
* عدد الوثائق أكبر من 200 ألف
* ليست Antique Dataset

مثال على Document:

What is the step by step guide to invest in share market in india?

مثال على Query:

Which question should I ask on Quora?

مثال على Qrel:

query_id = 46

doc_id = 134031

relevance = 1

اخترنا هذه 
Dataset
 لأنها تحقق جميع شروط المشروع وتحتوي على عدد جيد من الاستعلامات والـ 
 Qrels 
 مما يساعدنا على تقييم النظام باستخدام:

* MAP
* Recall
* Precision@10
* nDCG

لذلك تم اعتماد:

beir/quora/test

كداتاسيت أساسية للمشروع.
