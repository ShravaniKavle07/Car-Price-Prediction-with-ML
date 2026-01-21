<p align="center">
  <img src="https://img.icons8.com/?size=256&id=X5JXCeWYbw3V&format=png" width="120" alt="Car ML Logo"/>
</p>

<h1 align="center">🚗 Car Price Prediction using Machine Learning</h1>
<p align="center">A complete end-to-end ML project that predicts used car prices with high accuracy.</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-success?logo=google"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>

</p>


# 📘 **Project Overview**

This project builds a **Machine Learning model** to accurately predict the **selling price of used cars** based on features like:

- Brand  
- Year  
- Fuel type  
- Transmission  
- Kms driven  
- Owner history  
- Present price  

The project includes:

✔ Data Preprocessing  
✔ Exploratory Data Analysis (EDA)  
✔ Feature Engineering  
✔ Model Training  
✔ Model Evaluation  
✔ Predictions  



#  **Project Structure**

 Car-Price-Prediction-ML
│── dataset.csv
│── Car_Price_Prediction.ipynb
│── README.md
│── requirements.txt



# 📊 **Dataset Description**

| Column           | Description                                |
|------------------|--------------------------------------------|
| Car_Name         | Name of the car                           |
| Year             | Manufacturing year                        |
| Present_Price    | Ex-showroom price                         |
| Kms_Driven       | Total km driven                           |
| Fuel_Type        | Petrol / Diesel / CNG                     |
| Seller_Type      | Dealer / Individual                       |
| Transmission     | Manual / Automatic                        |
| Owner            | Number of previous owners                 |
| Selling_Price    | Target variable                           |


#  **ML Workflow**

###  1. Data Cleaning  
- Removing duplicates  
- Handling missing values  
- Outlier detection  

###  2. Feature Engineering  
- Creating `Car_Age`  
- Label encoding categorical variables  
- Normalization (if required)  

###  3. Exploratory Data Analysis  
- Price distribution  
- Brand-wise price analysis  
- Pair plots  
- Correlation heatmap  

###  4. Model Training  
Models used:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- XGBoost Regressor (optional)

###  5. Model Evaluation  
Metrics used:

- **R² Score**  
- **MAE**  
- **MSE / RMSE**  


#  **Results**

- **Random Forest Regressor performed best**  
- High accuracy and stable performance  
- Great for both budget and premium cars
   
#  **Model Performance

- Mean Absolute Error (MAE): 0.60
- Root Mean Squared Error (RMSE): 0.90
- R² Score: 0.965
