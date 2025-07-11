## ReelFriend

This project presents a Neural hybrid deep learning recommendation model that combines matrix factorization with neural networks, making use of residual connections and feature engineering to improve accuracy. It is built using Python, TensorFlow, and JavaScript.

The approach was designed to handle common challenges like the cold start problem and data sparsity, which frequently limit real-world systems. To make the model’s predictions more transparent, post hoc SHAP-based explanations are integrated to offer users insight into why specific items were recommended.  
The model is trained and evaluated using the MovieLens ml-latest-small dataset, and results show that it performs better than several standard baseline models across key metrics.

Alongside the model, a web-based application called ReelFriend was developed as a proof-of-concept to demonstrate how the system can be applied in practice, with a focus on delivering both personalised and understandable recommendations.

---

## Evaluation

![ReelFriend Model Evaluation](https://github.com/user-attachments/assets/9e3fcb08-ed1f-4947-87fc-487e33c3bd12)

The results highlight the strengths of the hybrid deep neural model, which achieved the lowest **RMSE** and **MAE** values among all evaluated approaches. This indicates a stronger ability to accurately predict user ratings.

In contrast, traditional collaborative filtering methods, both user-based and item-based, show higher prediction errors, likely due to their limited ability to model complex relationships in the data.

---

## Dataset
The dataset used for the project can be found here:
https://grouplens.org/datasets/movielens/latest/


