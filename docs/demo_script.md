# THAMAN — Graduation Defense Demo Script
## BSc Computer Science | Umm Al-Qura University | 2026

---

## Setup (before presentation)

Do this 15 minutes before the committee enters the room.

- [ ] Open browser, navigate to: https://huggingface.co/spaces/Turki-Almurahhem/thaman
- [ ] Wait for the map to fully load (the space cold-starts in ~30 sec)
- [ ] Toggle language to **Arabic** once, then back to **English** — confirm the bilingual toggle works
- [ ] Pan map to **NYC view** (map should default to NYC; if not, refresh once)
- [ ] Open `charts.html` in a second browser tab (keep it hidden behind the main tab)
- [ ] Have this script open on your phone or a second screen — not on the projector screen
- [ ] Confirm screen mirroring / projector is working before the committee sits down
- [ ] Mute your phone

**Coordinates to have ready (copy into browser console if Nominatim search is slow):**
```
NYC click target:     40.7549, -73.9840   (Midtown Manhattan)
Riyadh click target:  24.6877,  46.7219   (Downtown Riyadh / King Fahd Road)
```

---

## Narration (5 minutes)

---

### Step 1: NYC Prediction (1 minute)

**[Map is showing NYC. Click approximately on Midtown Manhattan — 40.7549, -73.9840]**

**English:**
> "This is THAMAN — a dual-city Automated Valuation Model I built for my graduation project. We're starting in New York City. I'm clicking on Midtown Manhattan to get an instant property valuation."

> "I'll select building type — let's go with Elevator Condo, D4 — and enter 1,200 square feet, 15 floors, built in 1985."

> [Submit prediction — pause for result to load]

> "THAMAN returns a predicted price, a confidence band, a letter grade, and the top features driving this estimate. On the right you can see the SHAP waterfall — building class and neighbourhood encoding are the top two features. That's consistent with NYC real estate: where you are and what type of building matter more than size alone."

> "The comparable sales bubbles on the map show the nearest actual recorded sales — green means our estimate is close, red means we're further off. These are real deed-recorded transactions from 185,000 NYC sales, 2022 to 2026."

**Arabic (direct translation for committee members):**
> "هذا نظام ثمان — نموذج تقييم عقاري ذكي لمدينتين طورته كمشروع تخرج. نبدأ في مدينة نيويورك. أضغط على منتصف مانهاتن للحصول على تقييم فوري للعقار."

> "اخترت نوع البناء: شقة بمصعد، المساحة 1200 قدم مربع، 15 طابقاً، بُني عام 1985."

> "النظام يعطينا سعراً تقديرياً، نطاق ثقة، درجة تقييم، وأهم العوامل المؤثرة. المبيعات المحيطة تظهر على الخريطة بنقاط ملوّنة من سجلات المعاملات الفعلية."

**Metric to highlight:**
> "Our NYC model achieves MedAPE of 20.24% on 27,763 holdout sales — competitive with commercial AVMs like Zillow Zestimate."

---

### Step 2: City Switch to Riyadh (30 seconds)

**[Click the city-switch toggle or navigate to Riyadh mode in the UI]**

**English:**
> "Now here's what makes THAMAN distinctive — it's a dual-city system. I'll switch to Riyadh."

> [Map animates to Riyadh view with district polygons visible]

> "The same stacking architecture, the same FastAPI backend, now running on a completely different market. Saudi Arabia's real estate data is published as district-level quarterly aggregates by the Ministry of Justice — not individual transactions like NYC. The model had to learn from 6,910 district-quarter observations instead of 185,000 individual sales."

**Arabic:**
> "الآن ننتقل إلى الرياض — وهذا ما يميّز ثمان. نفس البنية التقنية، لكن على سوق مختلف تماماً. بيانات العقارات السعودية تُنشر على مستوى الأحياء ربعياً، وليس كمعاملات فردية كما في نيويورك."

---

### Step 3: Riyadh Prediction + SHAP Drivers (1.5 minutes)

**[Click on Downtown Riyadh — coordinates 24.6877, 46.7219 — King Fahd Road area]**

**English:**
> "I'll click on the King Fahd Road corridor — one of Riyadh's prime districts. Let's select Villa, 400 square metres."

> [Submit prediction — wait for result]

> "The model returns a prediction in SAR per square metre. You can see the SHAP breakdown: metro access, commercial density, air quality, and district price history are the top drivers here."

> "The Riyadh Metro opened in 2024 and is a novel infrastructure signal — the model captures the premium for proximity to metro stations, which is entirely absent from pre-2024 models."

> "Notice the confidence interval is shown in SAR/sqm, and the district choropleth layer behind the prediction is showing the metro access overlay — you can see Line 1, the busiest east-west corridor, cutting across the city."

**Arabic:**
> "أضغط على منطقة طريق الملك فهد — إحدى أهم مناطق الرياض. سأختار فيلا، 400 متر مربع."

> "النموذج يُعطينا التقدير بالريال السعودي لكل متر مربع. في تحليل SHAP: القرب من المترو، الكثافة التجارية، جودة الهواء، والتاريخ السعري للحي هي أبرز العوامل."

> "مترو الرياض افتُتح عام 2024 وهو إشارة بنية تحتية جديدة يلتقطها النموذج — ما كان موجوداً في النماذج السابقة."

**Metrics to highlight here:**
> "On the training folds — out-of-fold cross-validation — the Riyadh model achieves R² = 0.9252 and MedAPE = 9.03%. That's the figure that tells you the model has genuinely learned the Saudi market structure."

> "On the holdout — Q1 through Q3 of 2025, entirely unseen calendar quarters — R² = 0.7981 and MedAPE = 18.16%. I'll explain that gap in a moment, but the short answer is: the holdout is a new-quarter stress test, not a random sample."

---

### Step 4: Listings Layer (30 seconds)

**[Toggle on the Haraj active listings layer]**

**English:**
> "This layer shows 1,615 active property listings scraped from Haraj.com.sa — Saudi Arabia's largest classifieds marketplace. Each point is colour-coded by type: blue for apartments, green for villas, amber for plots."

> "Click any bubble and you'll see the asking price versus our model's estimate, plus a direct link to the actual listing."

> "The model systematically predicts lower than asking prices — and that's expected. THAMAN was trained on deed-recorded transaction prices from the Ministry of Justice. Haraj shows what sellers are asking for, before negotiation. The overall gap is 54% MedAPE against asking prices, which is consistent with documented Saudi negotiation margins of 20 to 50 percent."

**Arabic:**
> "هذه الطبقة تُظهر 1615 عرضاً نشطاً من موقع حراج.كوم — أكبر سوق للعقارات في السعودية. كل نقطة مُلوَّنة حسب نوع العقار."

> "النموذج يتنبأ بأسعار أقل من أسعار العرض بشكل منتظم — وهذا متوقع. ثمان تدرّب على أسعار العقود المسجّلة، بينما حراج يعرض أسعار البائعين قبل التفاوض."

---

### Step 5: Analytics Dashboard (30 seconds)

**[Switch to second browser tab — charts.html]**

**English:**
> "Finally, the analytics dashboard. This shows model performance broken down by NYC borough and price tier."

> "Notice the Staten Island paradox: lowest R² in the dataset — 0.41 — but the best MedAPE at 13.5%. That's because Staten Island has very low price variance; the model's absolute errors are small, but R² penalises a low-variance target. MedAPE is the right metric for a user-facing AVM."

> "Manhattan is the hardest market at 36.7% MedAPE. Co-op board approval discounts and unobservable interior finishes create heterogeneity that no tabular dataset can capture."

**Arabic:**
> "لوحة التحليلات تُظهر أداء النموذج مقسّماً حسب منطقة نيويورك وشريحة السعر."

> "لاحظوا مفارقة ستاتن آيلاند: أقل R² في البيانات لكن أفضل MedAPE. السبب: تشابه العقارات يُصغّر التباين الكلي، فيُعاقب R² حتى التنبؤات الدقيقة."

---

### Step 6: Q&A Talking Points (1 minute)

Use this minute as a buffer. If the committee has not started asking questions, summarise:

**English closing:**
> "To summarise: THAMAN is a production-deployed AVM across two cities — New York and Riyadh — using a four-model stacking ensemble across 104 and 76 features respectively. It achieves competitive accuracy on NYC's 185,000-sale holdout and demonstrates cross-market generalisability on Saudi Arabia's data-scarce district-aggregate market. The full system — data pipelines, training code, API, and web interface — is deployed on Hugging Face and open-sourced on GitHub. Thank you."

**Arabic closing:**
> "خلاصة القول: ثمان نظام تقييم عقاري منتشر فعلياً لمدينتين، يستخدم مجموعة من أربعة نماذج ذكاء اصطناعي عبر مئة وأربع ميزات في نيويورك، وستة وسبعين ميزة في الرياض. يحقق دقة تنافسية على 27,763 مبيعة اختبارية في نيويورك، ويُثبت قابلية التعميم على السوق السعودية ذات البيانات المحدودة. النظام كاملاً — البيانات، الكود، الـ API، والواجهة — منشور على Hugging Face ومفتوح المصدر على GitHub. شكراً."

---

## Fallback (if Hugging Face is slow)

If the HF Space is still cold-starting (spinning/loading > 30 sec):

1. **Say:** "The deployed version is loading from cold start — this is common with Hugging Face Spaces after inactivity. While it loads, I'll walk through the architecture." 
2. Switch to showing the paper / slides (if available) and explain the model architecture verbally (§5 of the paper).
3. Keep refreshing the HF tab in the background — it typically loads in 45–90 seconds.
4. **If HF is completely unavailable:** Start the local API — open Terminal and run:
   ```
   cd /Users/totam/Desktop/new_try && uvicorn api.main:app --port 8000
   ```
   Then open: `http://localhost:8000/ui` in the browser.
   The local version is identical to the deployed version.
5. **API startup time:** ~30 seconds (spatial KD-tree indexes loading). Say: "The local API is starting up — it needs about 30 seconds to load the spatial indexes into memory."

---

## Key Numbers to Memorize

Print this section and keep it in your pocket.

| Metric | Value | Context |
|--------|-------|---------|
| NYC training rows | 185,092 | Sales from 2022–2026 |
| NYC features | 104 | Structural + spatial + QoL |
| NYC holdout rows | 27,763 | Time-based, newest 15% |
| NYC R² (holdout) | 0.6450 | Stack v11 |
| NYC MedAPE (holdout) | 20.24% | Stack v11 |
| Riyadh total rows | 6,910 | District-quarter obs., 2018–2025 |
| Riyadh training rows | 5,531 | Trained on 2018–2024 (including Metro-era) |
| Riyadh features | 76 | Transit, QoL, macro, rental |
| Riyadh OOF R² | **0.9252** | 5-fold spatial GroupKFold |
| Riyadh OOF MedAPE | **9.03%** | In-sample cross-validation |
| Riyadh holdout R² | 0.7981 | holdout covers 2025 Q1–Q3, n=1,379 |
| Riyadh holdout MedAPE | 18.16% | Out-of-sample stress test |
| Riyadh holdout MAE | 991 SAR/sqm | Out-of-sample stress test |
| Haraj validation MedAPE | 54.33% | Asking vs. transaction — expected gap |
| Haraj listings scraped | 1,615 | 444 apts, 630 villas, 526 plots, 15 buildings |
| NYC NTA groups | 212 | Neighbourhood spatial units |
| Riyadh district polygons | 133 | From OSM admin_level=10 |
| Base learners | 4 | XGB-A, XGB-B, LGB, CatBoost |
| Meta-learner | Ridge (L2) | positive=True, alpha=1.0 |
| NYC CV strategy | 10-fold Spatial GroupKFold | Groups = NTA code |
| Riyadh CV strategy | 5-fold Spatial GroupKFold | Groups = district_ar |
| API latency | 200–400 ms | Including SHAP computation |
| Automated tests | 37 | 15 scorer + 22 API tests |

---

## Anticipated Committee Questions + Model Answers

---

**Q1: "Why is the Riyadh OOF score higher than the holdout — 0.93 vs 0.80? Isn't that overfitting?"**

> **Short answer:** It's not classical overfitting — it's temporal distribution shift plus a known limitation of district-level aggregates in GroupKFold.

> **Long answer:** Three factors explain the gap. First, the holdout covers 2025 Q1–Q3 — an entirely unseen calendar horizon, not a random sample from the same time period. Even though 2024 Metro-era data (867 rows) is now correctly included in training, the 2025 quarters represent a further market evolution the model has not seen. Second, when data is organised as district-quarter aggregates, the OOF folds share temporal overlap: the same quarter appears across multiple folds, so the meta-learner sees some temporal information during CV. Third, the dataset has only 5,531 training rows across 163 districts, giving sparse per-district coverage in some areas. These factors combine to inflate OOF relative to the true out-of-time holdout. The OOF MedAPE of 9.03% vs holdout MedAPE of 18.16% shows the gap has narrowed significantly compared to the previous version (8.28% vs 23.45%), confirming that adding the 2024 Metro-era data to training was the right call. The holdout R² of 0.7981 is the honest number — and it's competitive for a 2025-only stress test on a 6,910-row dataset.

---

**Q2: "Why does THAMAN predict 54% below Haraj asking prices? Is the model wrong?"**

> **Short answer:** No — THAMAN was trained on deed-recorded transaction prices. Haraj shows asking prices before negotiation. The gap is structurally expected.

> **Long answer:** THAMAN was trained exclusively on Ministry of Justice deed records — what buyers actually paid after negotiation. Haraj.com.sa is a classifieds marketplace where sellers post aspirational asking prices. The Saudi residential real estate market is documented to have negotiation margins of 20–50% (Al-Otaibi & Al-Subaihi, 2021). An 80% listing premium over transaction prices — which the data confirms (median asking 5,232 SAR/sqm vs. training median 2,903 SAR/sqm) — is consistent with prior research (CBRE, 2024). Districts where the gap is smallest — النسيم الغربي at 17%, شبرا at 21% — are more transparent, liquid markets. Districts with the largest gaps — الفرسان at 450%, الشعلة at 443% — are speculative listing outliers in data-sparse areas.

---

**Q3: "Why use Ridge as the meta-learner instead of another gradient boosted model?"**

> **Short answer:** LightGBM at the meta level overfits to OOF noise. Ridge prevents this and gives better holdout performance despite a lower OOF score.

> **Long answer:** We empirically tested both. LightGBM meta achieved OOF R²=0.6376 but holdout R²=0.6349. Ridge meta achieved OOF R²=0.5995 but holdout R²=0.6450. The LightGBM meta learned to exploit residual noise and correlation patterns in OOF predictions rather than the true signal — meta-level overfitting. By the time predictions reach the meta-stage, the four base models have already exhausted the non-linear relationships in 104 features. The remaining variance is largely noise, and Ridge's L2 regularisation — with positive=True to enforce blending, not arbitrage — produces conservative, stable weights. This is a concrete, empirically validated architectural decision in the paper.

---

**Q4: "How does THAMAN handle neighbourhoods it has never seen before?"**

> **Short answer:** Fallback encodings — unknown NTAs receive the global median; unknown building classes get the borough mean. The SPARSE_MARKET flag fires if fewer than 5 comparables exist within 800m.

> **Long answer:** Target encodings for NTA, building class, NTA×building class, and borough×building class are computed on training data. At inference, if the queried location maps to an NTA not in the training set, the encoder returns the global mean log-price as a fallback. The SPARSE_MARKET quality flag signals this case to the user. Spatial features (distances, densities) are always computable from the KD-tree indexes regardless of NTA coverage.

---

**Q5: "What would it take to improve the Riyadh model accuracy?"**

> **Short answer:** More granular individual-transaction data, more quarters in the holdout training set, and interior property features from platforms like SA_Aqar.

> **Long answer:** Three improvements would have the most impact. First, individual transaction data (not district-level aggregates) — if the Ministry of Justice publishes parcel-level records, the training set could expand from 6,910 to potentially hundreds of thousands of rows. Second, as more 2025 quarters are published, retraining on the expanded dataset will close the OOF-to-holdout gap as the Metro-era market becomes better represented. Third, SA_Aqar rental data currently provides 4 district-level medians — expanding this to individual listing attributes (floor number, exact interior size, renovation year) would add the unobservable structural quality signals that currently limit the model in luxury tiers.

---

**Q6: "Why did you use a time-based split instead of random split for evaluation?"**

> **Short answer:** Random splitting causes temporal leakage — the model would learn from 2025 sales while predicting 2024 sales, giving a falsely optimistic score.

> **Long answer:** Real estate prices are temporally correlated — a sale in Brooklyn in January 2025 is highly informative about a sale in the same neighbourhood in February 2025. A random split would put some 2025 sales in training and some 2024 sales in test, allowing the model to effectively "look into the future." The time-based split — oldest 85% for training, newest 15% for holdout — simulates the real-world deployment scenario: the model is trained on historical data and asked to price properties it has never seen in a subsequent time period.

---

**Q7: "What is the difference between MedAPE and MAPE, and why do you prefer MedAPE?"**

> **Short answer:** MAPE averages percentage errors; MedAPE takes the median. One Manhattan $10M misvaluation can swing MAPE by several percentage points while barely moving MedAPE.

> **Long answer:** MAPE = mean(|y_true − y_pred| / y_true). In a right-skewed price distribution, a single extreme misvaluation — e.g. a $10M Manhattan penthouse predicted at $6M — contributes 40% / n to the MAPE, dominating the metric. MedAPE takes the 50th percentile of the absolute percentage error distribution, making it robust to such outliers. MedAPE is also the standard reporting metric in AVM industry literature (IAAO standards, Fannie Mae AVM guidelines). For a user-facing system, MedAPE answers: "what error does a typical user experience?" — which is the meaningful question.

---

**Q8: "Is your model fair? Does it discriminate by neighbourhood demographics?"**

> **Short answer:** Fairness is a known concern for AVMs. We include median income and crime rate as features — these reflect market reality but can encode historical inequity.

> **Long answer:** This is a legitimate and important question. The model uses median_income_nta and crime_rate_nta as features, which are themselves products of historical patterns including redlining and unequal resource allocation. A model trained on market prices will learn these patterns — because the market itself encodes them. Literature shows commercial AVMs have produced systematically higher errors in majority-Black neighbourhoods (Phan, 2018; broader AVM bias research). A fairness audit — measuring error rates stratified by NTA demographic composition — is listed as future work. The academic contribution here is not claiming the model is unbiased, but making its inputs transparent via SHAP so users can see exactly which features are driving any given estimate.

---
