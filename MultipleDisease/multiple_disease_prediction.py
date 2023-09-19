import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu

# loading the saved model
BreastCancer_model = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/BreastCancermodel.sav', 'rb'))
HeartDisease_model = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/HeartDiseasemodel.sav', 'rb'))
ParkinsonsDisease_model = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/ParkinsonsDiseasemodel.sav', 'rb'))

# loading the saved scaler
HeartDisease_Scaler = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/HeartDiseaseScaler.sav', 'rb'))
ParkinsonsDisease_Scaler = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/ParkinsonsDiseaseScaler.sav', 'rb'))
BreastCancer_Scaler = pickle.load(open(r'C:/Users/6917/PycharmProjects/Youtube/MultipleDisease/BreastCancerScaler.sav', 'rb'))


# slider for navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           ["Breast Cancer Prediction",
                                    "Heart Disease Prediction",
                                    "Parkinson's Prediction"],
                           icons=['bandaid', 'heart', 'person'],
                           default_index=0)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # page title
    st.title('Breast Cancer Prediction using ML')

    # Heart Disease Prediction Page
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        radiusM = st.text_input('Radius (Mean)')
    with col2:
        textureM = st.text_input('Texture Mean')
    with col3:
        perimeterM = st.text_input('Perimeter Mean')
    with col4:
        areaM = st.text_input('Area Mean')
    with col1:
        smoothnessM = st.text_input('Smoothness Mean')
    with col2:
        compactnessM = st.text_input('Compactness Mean')
    with col3:
        concavityM = st.text_input('Concavity Mean')
    with col4:
        concavepointsM = st.text_input('Concave Points Mean')
    with col1:
        symmetryM = st.text_input('Symmetry Mean')
    with col2:
        fractaldimensionM = st.text_input('Fractal Dimension Mean')
    with col3:
        radiusSE = st.text_input('Radius Se')
    with col4:
        textureSE = st.text_input('Texture Se')
    with col1:
        perimeterSE = st.text_input('Perimeter Se')
    with col2:
        areaSE = st.text_input('Area Se')
    with col3:
        smoothnessSE = st.text_input('Smoothness Se')
    with col4:
        compactnessSE = st.text_input('Compactness Se')
    with col1:
        concavitySE = st.text_input('Concavity Se')
    with col2:
        concavepointsSE = st.text_input('Concave Points Se')
    with col3:
        symmetrySE = st.text_input('Symmetry Se')
    with col4:
        fractaldimensionSE = st.text_input('Fractal Dimension Se')
    with col1:
        radiusW = st.text_input('Radius Worst')
    with col2:
        textureW = st.text_input('Texture Worst')
    with col3:
        perimeterW = st.text_input('Perimeter Worst')
    with col4:
        areaW = st.text_input('Area Worst')
    with col1:
        smoothnessW = st.text_input('Smoothness Worst')
    with col2:
        compactnessW = st.text_input('Compactness Worst')
    with col3:
        concavityW = st.text_input('Concavity Worst')
    with col4:
        concavepointsW = st.text_input('Concave Points Worst')
    with col1:
        symmetryW = st.text_input('Symmetry Worst')
    with col2:
        fractaldimensionW = st.text_input('Fractal Dimension Worst')

    #radiusM,textureM,perimeterM,areaM,smoothnessM,compactnessM,concavityM,concavepointsM,symmetryM,fractaldimensionM,radiusSE,textureSE,perimeterSE,areaSE,smoothnessSE,compactnessSE,concavitySE,concavepointsSE,symmetrySE,fractaldimensionSE,radiusW,textureW,perimeterW,areaW,smoothnessW,compactnessW,concavityW,concavepointsW,symmetryW,fractaldimensionW


    # code for prediction
    diab_diagnosis = ''

    if st.button('Breast Cancer Test Result',type='primary'):

        #X = [[radius_mean,texture_mean,perimeter_mean,area_mean, smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean, fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se, concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst, perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst, symmetry_worst,fractal_dimension_worst]]
        X = [[radiusM,textureM,perimeterM,areaM,smoothnessM,compactnessM,concavityM,concavepointsM,symmetryM,fractaldimensionM,radiusSE,textureSE,perimeterSE,areaSE,smoothnessSE,compactnessSE,concavitySE,concavepointsSE,symmetrySE,fractaldimensionSE,radiusW,textureW,perimeterW,areaW,smoothnessW,compactnessW,concavityW,concavepointsW,symmetryW,fractaldimensionW]]
        #X = [[12, 15.65, 76.95, 443.3, 0.09723, 0.07165, 0.04151, 0.01863, 0.2079, 0.05968, 0.2271, 1.255, 1.441, 16.16, 0.005969, 0.01812, 0.02007, 0.007027, 0.01972, 0.002607, 13.67, 24.9, 87.78, 567.9, 0.1377, 0.2003, 0.2267, 0.07632, 0.3379, 0.07924]]
        std_X = BreastCancer_Scaler.transform(X)
        diab_diagnosis = BreastCancer_model.predict(std_X)

        if diab_diagnosis[0] == 0:
            diab_diagnosis = 'The person has benign Breast Cancer :blush:'
        else:
            diab_diagnosis = 'The person has maglinant Breast Cancer :worried:'

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    # Heart Disease Prediction Page
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain Type')
    with col1:
        trestbhps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise induced Angina')
    with col1:
        oldpeak = st.text_input('Old Peak')
    with col2:
        slope = st.text_input('Slope of the peak')
    with col3:
        ca = st.text_input('No of Major Vessels colored by Flourosopy')
    with col1:
        thal = st.text_input('Thal')

    # code for prediction
    diab_diagnosis = ''

    if st.button('Heart Test Result',type='primary'):

        X = [[age,sex,cp,trestbhps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
        std_X = HeartDisease_Scaler.transform(X)
        diab_diagnosis = HeartDisease_model.predict(std_X)

        if diab_diagnosis[0] == 0:
            diab_diagnosis = 'The person is does not have any heart disease :blush:'
        else:
            diab_diagnosis = 'The person is having heart disease :worried:'

    st.success(diab_diagnosis)



# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    # page title
    st.title("Parkinson's Prediction using ML")

    # Parkinson's Disease Prediction Page
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Fo = st.text_input("MDVP Fo(Hz)")
    with col2:
        Fhi = st.text_input('MDVP Fhi(Hz)')
    with col3:
        Flo = st.text_input('MDVP Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP Jitter(%)')
    with col1:
        Jitter_Abs = st.text_input('MDVP Jitter(Abs)')
    with col2:
        RAP = st.text_input('MDVP RAP')
    with col3:
        PPQ = st.text_input('MDVP PPQ')
    with col4:
        DDP = st.text_input('Jitter DDP')
    with col1:
        Shimmer = st.text_input('MDVP Shimmer')
    with col2:
        Shimmer_dB = st.text_input('MDVP Shimmer(dB)')
    with col3:
        Shimmer_APQ3 = st.text_input('Shimmer APQ3')
    with col4:
        Shimmer_APQ5 = st.text_input('Shimmer APQ5')
    with col1:
        MDVP_APQ = st.text_input('MDVP APQ')
    with col2:
        Shimmer_DDA = st.text_input('Shimmer DDA')
    with col3:
        NHR = st.text_input('NHR')
    with col4:
        HNR = st.text_input('HNR')
    with col1:
        RPDE = st.text_input('RPDE')
    with col2:
        DFA = st.text_input('DFA')
    with col3:
        spread1 = st.text_input('spread1')
    with col4:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # code for prediction
    diab_diagnosis = ''

    if st.button("Parkinson's Test Result",type="primary"):
        X = [[Fo,Fhi,Flo,Jitter_percent,Jitter_Abs,RAP,PPQ,DDP,Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]
        std_X = ParkinsonsDisease_Scaler.transform(X)
        diab_diagnosis = ParkinsonsDisease_model.predict(std_X)


        if diab_diagnosis[0] == 0:
            diab_diagnosis = "The person is does not have any Parkinson's disease :smile:"

        else:
            diab_diagnosis = "The person is having Parkinson's disease :worried:"

    st.success(diab_diagnosis)

    #uploaded_files = st.file_uploader("Choose a CSV file", type=["csv"])







