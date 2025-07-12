## ReelFriend

This solo project presents a Neural hybrid deep learning recommendation model that combines matrix factorization with neural networks, making use of residual connections and feature engineering to improve accuracy. It is built using Python, TensorFlow, and JavaScript.

The approach was designed to handle common challenges like the cold start problem and data sparsity, which frequently limit real-world systems. 

To make the model’s predictions more transparent, post hoc SHAP-based explanations are integrated to offer users insight into why specific items were recommended.  
The model is trained and evaluated using the MovieLens ml-latest-small dataset, and results show that it predicts 20% more accurately than several standard baseline models across key metrics.

Alongside the model, a web application called ReelFriend was developed to demonstrate how the system can be applied in practice, with a focus on delivering both personalised and understandable recommendations.

---

## Exploratory Data Analysis

Before developing the recommendation model, an in-depth exploratory data analysis (EDA) was conducted on the MovieLens dataset to understand the data’s structure and patterns. The exploratory data analysis was performed in a dedicated Python script, `exploration.py`, using the `matplotlib` and `pandas` libraries. The script loads the MovieLens dataset, applies data transformations, computes summary statistics, and generates figures for various statistics. The use of `matplotlib` enabled quick generation of these figures, which were helpful in gaining a better grasp of the dataset.

<img width="641" height="374" alt="Screenshot 2025-07-11 at 23 06 22" src="https://github.com/user-attachments/assets/6c4d74bf-006b-4ada-9cde-e09cde34d6c2" />

---

## Evaluation
![Model Evaluation Results](https://github.com/user-attachments/assets/ceb3f993-8337-4f17-83e2-30120b4b0319)




The results highlight the strengths of the hybrid deep neural model, which achieved the lowest **RMSE** and **MAE** values among all evaluated approaches. This indicates a stronger ability to accurately predict user ratings.

These results show a ~20% accuracy improvement over baseline models! Meaning my model is very good at predicting what users will like on average.

In contrast, traditional collaborative filtering methods, both user-based and item-based, show higher prediction errors, likely due to their limited ability to model complex relationships in the data.

---

## Dataset

The dataset used for the project can be found here:  
[MovieLens latest dataset](https://grouplens.org/datasets/movielens/latest/)

---

## Demo
A Demo Video of the Web Application can be found here: 
[ReelFriend Demo Video](https://www.youtube.com/watch?v=Kk1nljuuW1w&t=7s)
-----------
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)


