 
## NY Apartment listing price prediction using Zillow API 🏡
 
![App Screenshot](https://cdn.vox-cdn.com/thumbor/n__W88RH2lLfwikcCFBISLOxreE=/0x0:2000x1333/1200x800/filters:focal(837x619:1157x939)/cdn.vox-cdn.com/uploads/chorus_image/image/65368722/171109_06_27_03_5DS_9686.0.jpg)

---
### Disclaimer 
* Please open all .ipynb notebooks in google.colab , because github has problems with showing visualization packages.
![disclaimer](https://snipboard.io/uHI1v5.jpg)
 
### Project Overview 
---
* Created fine-tuned machine learning models that predicts listing price of apartments in New York based on data from custom-made Zillow API.
* Engineered features using latitude and longitude columns to link geospatial provided data with model to improve model performance.
---
### How will this project help? 
* This project can help with determining fair value for apartments in NY. Later this information could be used as a way to make right decisions while purchasing, selling apartments or when looking for investment opportunities that are connected with Real Estate market.
---
### Resources used 
* *packages: pandas, numpy, matplotlib, seaborn, plotly, folium, squarify, missingno, datetime, sklearn, google.colab*
* *Data: Custom-made Zillow API* - [Link for API](https://rapidapi.com/apimaker/api/zillow-com1/)
---
### Details of project 
* This project was built using Google Colab. Also, Google Drive was used in order to save and transport dataframe. To hide API keys I also used a pre-made .csv file containing access to private API keys. For simplicity, project were divided into 2 notebooks. **GettingData.ipynb** - contains the process of extracting data from APi. **EDA&Building_model.ipynb** - contains processes of cleaning, EDA, building model, fine_tuning.
---
### Phases in EDA&Building_model notebook
1. Installation & Import of required libraries
2. Structure Investigation
3. Exploratory Data Analysis (EDA)
4. Correlationship Analysis
5. Geospatial Data Analysis
6. Feature Engineering
7. Preprocessing dataframe for building models
8. Baseline models
9. Fine tuning
10. Interpretability
---
### Feature Engineered features 
* `distance`- Feature that was created by using information from `longitude` and `latitude` features and then using coordinates of Empire State building. Finally, by using *Haversine Formula* distance between two points were calculated.
* Where distance to the Empire State Building was used?  - Because this resulted in a better score compared with other places used for this feature. You could see in the code block below which coordinates were also used and tested for `distance` feature.
``` df['latitude_empier_state_building'] =40.748500421093865
df['longitude_empier_state_building'] = -73.98556979674355
# 40.748500421093865, -73.98556979674355 Empire State Building
# 40.752640832162534, -73.97738783463491 - Main Railway station
# 40.64957748302697, -73.79345274856009 - John F. Kennedy International Airport
# 40.7671676305183, -73.97889716461052 - Entry to Central Park
```
---
### Models: 
* Dataframe containing results of models (sorted by RMSE) :
![plot with results](https://snipboard.io/ru8mR2.jpg)
* Plot containing results of models:
![plot with model results](https://snipboard.io/ec4rdV.jpg)
 
### Model choice and fine-tuning ✔️✔️✔️
* GradientBoostingRegressor for further exploration due to a big difference in scores compared with results of other models. First of all, I tried to manually determine the optimal range of values for given parameters, after that GridSearchCV with KFold were used for further fine-tuning.
```
params ={
   'max_depth':[5, 6, 7, 8, 9],
   'learning_rate': [0.25, 0.3, 0.4, 0.5, 0.6],
   'n_estimators': [300, 400, 500, 600, 700],
   'subsample': [0.3, 0.4, 0.5, 0.6, 0.7]
}
 
folds = 10
model = GradientBoostingRegressor()
kf = KFold(n_splits=folds, shuffle=True, random_state=1)
gridsearch = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", n_jobs=-1, cv=kf.split(X,y), verbose=3)
```
* *Results of fine-tuned GradientBoostingRegressor model:*
|Model Name| MSE | RMSE            |MAE |
| -----------|:----:|:-----:|-------------:|
|Gradient Boosting Regressor| 8.065473e+10| 283997.761376|204007.123004|
 
---
### Interpretability of model 
* For this purpose, SHAP package were used.
Mean SHAP value
![shap value](https://snipboard.io/LAe2Z6.jpg)
SHAP value (impact on model output):
![plot with model results](https://snipboard.io/i0rpNs.jpg)
SHAP values:
![plot with model results](https://snipboard.io/3oqnTW.jpg)
Plot shows how complex model arrive at their prediction for randomly choosen row 15:
![plot with model results](https://snipboard.io/V48zof.jpg)
Plot shows how complex model arrive at their prediction for randomly choosen random row 25:
![plot with model results](https://snipboard.io/FnyBDf.jpg)
---
### Reference

[1] https://www.kaggle.com/code/janiobachmann/melbourne-comprehensive-housing-market-analysis 

[2] https://rapidapi.com/apimaker/api/zillow-com1/

[3] https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values
