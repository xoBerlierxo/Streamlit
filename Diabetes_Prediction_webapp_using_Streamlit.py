import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))
st.title('Diabetes Prediction Web App')

# Prediction function
def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function
def main():
    st.title('Diabetes Prediction Web App')

    # Input fields
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')

    # Predict button
    if st.button('Diabetes Test Result'):
        try:
            result = predict_diabetes([
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ])
            st.success(result)
        except ValueError:
            st.error("Please enter valid numerical values in all fields.")

if __name__ == '__main__':
    main()
