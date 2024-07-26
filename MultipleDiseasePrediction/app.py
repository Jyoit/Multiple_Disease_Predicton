import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
diabetes_dataset = pd.read_csv("diabetes.csv")
heart_dataset = pd.read_csv("heart.csv")
hypertension_dataset = pd.read_csv("hypertension_data.csv")

# Train models
def model_training(dataset, target_column):
    x = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy for {target_column} is: {accuracy:.2f}")
    return model

st.title("Health Prediction Application")

# Sidebar for dataset display
st.sidebar.title("Dataset Display")
if st.sidebar.checkbox("Show Diabetes Dataset"):
    st.subheader("Diabetes Dataset")
    st.dataframe(diabetes_dataset)
if st.sidebar.checkbox("Show Heart Dataset"):
    st.subheader("Heart Dataset")
    st.dataframe(heart_dataset)
if st.sidebar.checkbox("Show Hypertension Dataset"):
    st.subheader("Hypertension Dataset")
    st.dataframe(hypertension_dataset)

# Train models and display accuracies
st.subheader("Training Models")
diabetes_model = model_training(diabetes_dataset, "Outcome")
heart_model = model_training(heart_dataset, "target")
hypertension_model = model_training(hypertension_dataset, "target")

# User input for prediction
st.subheader("Make Predictions")
st.write("Provide input data for predictions:")

with st.form(key='prediction_form'):
    st.write("Diabetes Input")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1, value=85)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, value=66)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, value=29)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1, value=26.6)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001, value=0.351)
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=31)

    st.write("Heart Disease Input")
    age_heart = st.number_input("Age", min_value=0, max_value=120, step=1, value=37)
    sex = st.selectbox("Sex", [0, 1], key='sex_heart')
    cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, step=1, value=2, key='cp_heart')
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, step=1, value=130, key='trestbps_heart')
    chol = st.number_input("Cholesterol", min_value=0, max_value=600, step=1, value=250, key='chol_heart')
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], key='fbs_heart')
    restecg = st.number_input("Resting ECG", min_value=0, max_value=2, step=1, value=1, key='restecg_heart')
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, step=1, value=187, key='thalach_heart')
    exang = st.selectbox("Exercise Induced Angina", [0, 1], key='exang_heart')
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, value=3.5, key='oldpeak_heart')
    slope = st.number_input("Slope of ST Segment", min_value=0, max_value=2, step=1, value=0, key='slope_heart')
    ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, step=1, value=0, key='ca_heart')
    thal = st.number_input("Thalassemia", min_value=0, max_value=3, step=1, value=2, key='thal_heart')

    st.write("Hypertension Input")
    age_hyp = st.number_input("Age", min_value=0, max_value=120, step=1, value=57, key='age_hyp')
    sex_hyp = st.selectbox("Sex", [0, 1], key='sex_hyp')
    cp_hyp = st.number_input("Chest Pain Type", min_value=0, max_value=3, step=1, value=3, key='cp_hyp')
    trestbps_hyp = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, step=1, value=145, key='trestbps_hyp')
    chol_hyp = st.number_input("Cholesterol", min_value=0, max_value=600, step=1, value=233, key='chol_hyp')
    fbs_hyp = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], key='fbs_hyp')
    restecg_hyp = st.number_input("Resting ECG", min_value=0, max_value=2, step=1, value=0, key='restecg_hyp')
    thalach_hyp = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, step=1, value=150, key='thalach_hyp')
    exang_hyp = st.selectbox("Exercise Induced Angina", [0, 1], key='exang_hyp')
    oldpeak_hyp = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, value=2.3, key='oldpeak_hyp')
    slope_hyp = st.number_input("Slope of ST Segment", min_value=0, max_value=2, step=1, value=0, key='slope_hyp')
    ca_hyp = st.number_input("Number of Major Vessels", min_value=0, max_value=4, step=1, value=0, key='ca_hyp')
    thal_hyp = st.number_input("Thalassemia", min_value=0, max_value=3, step=1, value=1, key='thal_hyp')

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    new_data_diabetes = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    new_data_heart = pd.DataFrame({
        'age': [age_heart],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    new_data_hypertension = pd.DataFrame({
        'age': [age_hyp],
        'sex': [sex_hyp],
        'cp': [cp_hyp],
        'trestbps': [trestbps_hyp],
        'chol': [chol_hyp],
        'fbs': [fbs_hyp],
        'restecg': [restecg_hyp],
        'thalach': [thalach_hyp],
        'exang': [exang_hyp],
        'oldpeak': [oldpeak_hyp],
        'slope': [slope_hyp],
        'ca': [ca_hyp],
        'thal': [thal_hyp]
    })

    def prediction(diabetes_model, heart_model, hypertension_model, newDataDiabetes, newDataHeart, newDataHypertension):
        prediction = {}

        heart_disease = heart_model.predict(newDataHeart)
        prediction["heart disease"] = "Yes" if heart_disease[0] == 1 else "No"

        diabetes_disease = diabetes_model.predict(newDataDiabetes)
        prediction["diabetes"] = "Yes" if diabetes_disease[0] == 1 else "No"

        hypertension_disease = hypertension_model.predict(newDataHypertension)
        prediction["hypertension"] = "Yes" if hypertension_disease[0] == 1 else "No"

        return prediction

    predictions = prediction(diabetes_model, heart_model, hypertension_model, new_data_diabetes, new_data_heart, new_data_hypertension)
    
    # Generate diagnosis message
    diagnosis_message = "You have been diagnosed with the following conditions: "
    diagnosed_conditions = []

    for disease, result in predictions.items():
        if result == "Yes":
            diagnosed_conditions.append(disease)

    if diagnosed_conditions:
        diagnosis_message += ", ".join(diagnosed_conditions) + "."
    else:
        diagnosis_message = "You have not been diagnosed with any of the conditions."

    # Display the diagnosis message in a larger font
    st.markdown(f"<h3 style='color: white;'>{diagnosis_message}</h3>", unsafe_allow_html=True)

# Visualize datasets
# st.subheader("Data Visualization")

# if st.checkbox("Show Pairplot for Diabetes Dataset"):
#     sns.pairplot(diabetes_dataset, hue='Outcome')
#     st.pyplot(plt.gcf())

# if st.checkbox("Show Pairplot for Heart Dataset"):
#     sns.pairplot(heart_dataset, hue='target')
#     st.pyplot(plt.gcf())

# if st.checkbox("Show Pairplot for Hypertension Dataset"):
#     sns.pairplot(hypertension_dataset, hue='target')
#     st.pyplot(plt.gcf())
