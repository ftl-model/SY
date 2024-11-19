###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('RFC5.pkl')
scaler = joblib.load('scaler5.pkl') 

# Define feature options
Level_of_Education_options = {    
    0: 'Primary (0)',    
    1: 'Secondary (1)',    
    2: 'Certificate (2)',    
    3: 'Diploma (3)',
    4: 'Degree (4)'
}

Tumor_Grade_options = {       
    1: 'Grade1 (1)',    
    2: 'Grade2 (2)',    
    3: 'Grade3 (3)',
    4: 'Grade4 (4)'
}


# Define feature names
feature_names = ["Level_of_Education", "Tumor_Size_2_Years_after_Surgery", "Tumor_Grade", "Lymph_Node_Metastasis", "Numbe_of_Lymph_Nodes", "Marital_Status_Unmarried", "Marital_Status_Married", "Marital_Status_Divorced"]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")

# Level_of_Education
Level_of_Education = st.selectbox("Level of Education:", options=list(Level_of_Education_options.keys()), format_func=lambda x: Level_of_Education_options[x])

# Tumor_Size_2_Years_after_Surgery
Tumor_Size_2_Years_after_Surgery = st.number_input("Tumor Size 2 Years after Surgery(mm):", min_value=0, max_value=100, value=50)

# Tumor_Grade
Tumor_Grade = st.selectbox("Tumor Grade:", options=list(Tumor_Grade_options.keys()), format_func=lambda x: Tumor_Grade_options[x])

# Lymph_Node_Metastasis
Lymph_Node_Metastasis = st.selectbox("Lymph Node Metastasis:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Numbe_of_Lymph_Nodes
Numbe_of_Lymph_Nodes = st.number_input("Numbe of Lymph Nodes:", min_value=0, max_value=50, value=25)

# Marital_Status_Unmarried
Marital_Status_Unmarried = st.selectbox("Marital Status Unmarried:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Married
Marital_Status_Married = st.selectbox("Marital Status Married:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Marital_Status_Divorced
Marital_Status_Divorced = st.selectbox("Marital Status Divorced:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')


# 准备输入特征
feature_values = [Level_of_Education, Tumor_Size_2_Years_after_Surgery, Tumor_Grade, Lymph_Node_Metastasis, Numbe_of_Lymph_Nodes, Marital_Status_Unmarried, Marital_Status_Married, Marital_Status_Divorced]

# 分离连续变量和分类变量
continuous_features = [Tumor_Size_2_Years_after_Surgery, Numbe_of_Lymph_Nodes]
categorical_features=[Level_of_Education,Tumor_Grade,Lymph_Node_Metastasis,Marital_Status_Unmarried,Marital_Status_Married,Marital_Status_Divorced]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["Tumor_Size_2_Years_after_Surgery", "Numbe_of_Lymph_Nodes"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])


# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of breast cancer recurrence. "            
                f"The model predicts that your probability of having breast cancer recurrence is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of breast cancer recurrence. "            
                f"The model predicts that your probability of not having breast cancer recurrence is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

    # 创建SHAP解释器
    explainer_shap = shap.TreeExplainer(model)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame([final_features_df],columns=feature_names))
    
  
    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
         shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([final_features_df], columns=feature_names),matplotlib=True)
    else:
         shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,0], pd.DataFrame([final_features_df], columns=feature_names),matplotlib=True)
    
    # 保存图像并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
