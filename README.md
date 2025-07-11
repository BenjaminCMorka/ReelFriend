## ReelFriend

This project presents a Neural hybrid deep learning recommendation model that combines matrix factorization with neural networks, making use of residual connections and feature engineering to improve accuracy. It is built using Python, TensorFlow, and JavaScript.

The approach was designed to handle common challenges like the cold start problem and data sparsity, which frequently limit real-world systems. To make the model’s predictions more transparent, post hoc SHAP-based explanations are integrated to offer users insight into why specific items were recommended.  
The model is trained and evaluated using the MovieLens ml-latest-small dataset, and results show that it performs better than several standard baseline models across key metrics.

Alongside the model, a web application called ReelFriend was developed to demonstrate how the system can be applied in practice, with a focus on delivering both personalised and understandable recommendations.

---

## Exploratory Data Analysis

Before developing the recommendation model, an in-depth exploratory data analysis (EDA) was conducted on the MovieLens dataset to understand the data’s structure and patterns. The exploratory data analysis was performed in a dedicated Python script, `exploration.py`, using the `matplotlib` and `pandas` libraries. The script loads the MovieLens dataset, applies data transformations, computes summary statistics, and generates figures for various statistics. The use of `matplotlib` enabled quick generation of these figures, which were helpful in gaining a better grasp of the dataset.

<img width="641" height="374" alt="Screenshot 2025-07-11 at 23 06 22" src="https://github.com/user-attachments/assets/6c4d74bf-006b-4ada-9cde-e09cde34d6c2" />

---

## Data Processing Pipeline

Below is a diagram detailing the data processing pipeline for the neural hybrid recommendation model. It shows the process through which raw data is prepared into features used to train the neural network, enabling accurate movie recommendations

![Data Processing Pipeline](https://github.com/user-attachments/assets/b9bb4738-19f8-4fd9-8077-f06039d3b045)


---


## Evaluation

![ReelFriend Model Evaluation](https://github.com/user-attachments/assets/9e3fcb08-ed1f-4947-87fc-487e33c3bd12)

The results highlight the strengths of the hybrid deep neural model, which achieved the lowest **RMSE** and **MAE** values among all evaluated approaches. This indicates a stronger ability to accurately predict user ratings.

These results show a ~20% accuracy improvement over baseline models! Meaning my model is very good at predicting what users will like on average.

In contrast, traditional collaborative filtering methods, both user-based and item-based, show higher prediction errors, likely due to their limited ability to model complex relationships in the data.

---

## Dataset

The dataset used for the project can be found here:  
[MovieLens latest dataset](https://grouplens.org/datasets/movielens/latest/)


