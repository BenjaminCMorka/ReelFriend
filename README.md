ReelFriend

This project presents a Neural hybrid deep learning recommendation model that combines matrix factorization with neural networks, making use of residual connections and feature engineering to improve accuracy. It is built using Python, TensorFlow, and JavaScript.

The approach was designed to handle common challenges like the cold start problem and data sparsity, which frequently limit real-world systems. To make the model’s predictions more transparent, post hoc SHAP-based explanations are integrated to offer users insight into why specific items were recommended. 
The model is trained and evaluated using the MovieLens ml-latest-small dataset, and results show that it performs better than several standard baseline models across key metrics.

Alongside the model, a web-based application called ReelFriend was developed as a proof-of-concept to demonstrate how the system can be applied in practice, with a focus on delivering both personalised and understandable recommendations.
