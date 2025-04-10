# Predictive Engagement Analytics 
---

This self-directed project explores what drives social media engagement using a dataset of 731 posts from Kaggle. The goal: predict a custom Engagement Score defined as:

> Engagement Score = Likes + (2 × Retweets)

This weighting emphasizes the strategic value of shares over likes, focusing on visibility and amplification rather than surface-level popularity.

The project covers full-cycle analytics. From data cleaning and feature engineering, to predictive modeling and interactive data storytelling with Tableau, my aim was to see how accurate we can predict engagement on a modest dataset.

---

**Notebook**

🔗 [Quick Jump to File](https://github.com/AKapett/Social_Media_Engagement_Predictive_Model/blob/main/Cleaned%20SM%20Engagement%20Model.ipynb)



**Tableau Dashboard**

🔗 [Dashboard](https://public.tableau.com/views/SocialMediaEngagementRetweetWeighted/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


---

## Assumptions & Limitations

- **Engagement Score Weighting:** Likes + (2 × Retweets) was chosen to reflect shareability over vanity metrics

- **Limited user data:** 89% of users posted only once in this dataset, restricting behavioral pattern analysis. No metadata available for follower count, user activity, etc.

- **Assumed duplication**: Identical posts across platforms were treated as duplicates in the absence of clear cross-post indicators and improbability of exact like and share count

---

## Workflow

**Data Cleaning & Preparation**

- Removed misleading duplicates (same timestamp, content, but different platform or country)
- Validated time values and binned post length into categorical buckets
- Grouped rare countries and sentiments under "Other"


**Feature Engineering**

- Derived temporal features like Daypart, Time Block, and Day of Week
- Extracted content structure features (Char_Per_Hashtag, Post_Char_Count)
- Created interaction terms (e.g., Sentiment × Hashtag Structure) to capture nuanced behavioral patterns


**Modeling**

- Trained and tuned XGBoost and CatBoost regressors
- Used RepeatedKFold with GridSearchCV and RandomizedSearchCV
- Applied SHAP for feature explainability and importance ranking

> CatBoost ultimately delivered the best trade-off between accuracy and generalization.

--

**Model	Test R²	Train R² Notes**

CatBoost	0.3442	0.5479	Strongest overall performance

XGBoost	0.3302	0.7202	Higher variance, more overfit

--

**Model Performace w/o weighted Engagement Score provided similar results**

Despite switching to an unweighted engagement definition, the model achieved close to identical R² score compared to the weighted version. However,  as expected, the MSE was lower due to smaller absolute error values from the smaller target magnitudes.

---

## Tableau Dashboard

An interactive dashboard was built in Tableau to surface patterns in both total and per-post engagement. Users can filter by country, platform, sentiment, day, post length, and more.

**Key Features**

- Toggle between SUM and AVG engagement

- View engagement by daypart, platform, sentiment, post length, and month

- Visual breakdown of likes vs. retweets

- Dynamic tooltips and filters for strategic exploration


🔗 [Dashboard](https://public.tableau.com/views/SocialMediaEngagementRetweetWeighted/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Why It Matters

This project demonstrates that meaningful insight and predictive modeling are possible even on modest datasets. 

While this dataset lacks real-world volume and depth, the modeling pipeline is scalable to richer, real behavioral datasets.

With future access to user-level context (followers, history, media type), this framework could evolve into a robust decision-support system for content planning and campaign optimization.

