# Predictive Engagement Analytics — Project Overview
---

This self-directed project explores what drives social media engagement using a dataset of 731 posts from Kaggle. The goal: predict a custom Engagement Score defined as:

Engagement Score = Likes + (2 × Retweets)

This weighting emphasizes the strategic value of shares over likes, focusing on visibility and amplification rather than surface-level popularity.

The project covers full-cycle analytics — from data cleaning and feature engineering to predictive modeling and interactive data storytelling with Tableau.

---

## Assumptions & Limitations

- **Engagement Score Weighting:** Likes + (2 × Retweets) was chosen to reflect shareability over vanity metrics

- **Limited user data:** 89% of users posted only once, restricting behavioral pattern analysis

- **Assumed duplication**: Identical posts across platforms were treated as duplicates in the absence of clear cross-post indicators

---

## Workflow Summary

**1. Data Cleaning & Preparation**

- Removed misleading duplicates (same timestamp, content, but different platform or country)

- Validated time values and binned post length into categorical buckets

- Grouped rare countries and sentiments under "Other"


**2. Feature Engineering**

- Extracted temporal features (e.g., Day of Week, Daypart, Time Block)

- Calculated content metrics (e.g., Hashtag Count, Char_Per_Hashtag)

- Grouped infrequent sentiment labels for cleaner modeling

- Built interaction terms (e.g., Sentiment × Time) for behavioral nuance


**3. Modeling**

- Trained and tuned XGBoost and CatBoost regressors

- Used GridSearchCV, RandomizedSearchCV, and RepeatedKFold for validation

- Applied SHAP to guide feature selection and reduce noise

- Model	Test R²	Train R²	Notes

--

CatBoost (Final)	0.4008	0.5605	Best generalization + categorical support

XGBoost	0.3626	0.7730	Strong fit, prone to overfitting

CatBoost (Time Block only)	0.3609	0.4749	Simpler time abstraction, decent generalization

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

This project demonstrates that models can be built even on modest datasets. With a focus on feature clarity, explainability, and strategic alignment, this workflow shows how analytics can help marketers and content creators optimize their strategy to not only understand their metrics, but plan content campaigns with data-driven direction.

I believe with additional user-level data (e.g., followers, engagement history, posting frequency), this pipeline could scale into a powerful decision-support tool for content and campaign teams.

