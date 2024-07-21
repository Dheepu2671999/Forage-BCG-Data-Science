import streamlit as st 
import joblib
import pandas as pd
import numpy as np

rf_clf = joblib.load(r'models\random_forest.mt')

title = "PowerCo Customer Churn Prediction"
style = f"<style>h1 {{ color: blue; font-size: 40px; font-weight: bold; }}</style>"
st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

def data():
    data = {
            'margin_net_pow_ele': [margin_net_pow_ele],
            'forecast_cons_12m': [forecast_cons_12m],
            'forecast_meter_rent_12m': [forecast_meter_rent_12m],
            'net_margin': [net_margin],
            'cons_12m': [cons_12m]
        }

    df = pd.DataFrame(data)
    return df

def change_log(x):
    return np.log10(1+x)


info = ''
st.write(":green[_Enter all the below details to check if the customer will retain or not_]:woman-running:")
cons_12m= change_log(st.number_input(":blue[**cons_12m**]: Electricity consumption of the past 12 months"))
net_margin= st.number_input(":blue[**net_margin**]: Total net margin")
forecast_meter_rent_12m= change_log(st.number_input(":blue[**forecast_meter_rent_12m**]: Forecasted bill of meter rental for the next 2 months"))
forecast_cons_12m= change_log(st.number_input(":blue[**forecast_cons_12m**]: Forecasted electricity consumption for next 12 months"))
margin_net_pow_ele= st.number_input(":blue[**margin_net_pow_ele**]: Net margin on power subscription")

if st.button("Predict"):
            prediction = rf_clf.predict(data())            
            if(prediction[0] == 0):
                 info = 'The user is still customer'
            elif(prediction[0] == 1):
              info = 'The user is not  customer'
              
st.success('Prediction: {}'.format(info))


with st.sidebar:
    sidebar_title = "Summary"
    sidebar_title_style = f"<h2 style='color: blue; font-size: 40px;font-weight: bold;'>{sidebar_title}</h2>"
    st.markdown(sidebar_title_style, unsafe_allow_html=True)
    st.write("PowerCo is a major gas and electricity utility that supplies to corporate,"
                    " SME (Small & Medium Enterprise), and residential customers. "
                    "The power-liberalization of the energy market in Europe has led to significant customer churn,"
                    " especially in the SME segment. "
                    "They have partnered with BCG to help diagnose the source of churning SME customers.")
    st.write("**Sample Prediction Data for the user :red[not a customer]**")
    st.write("""
    - cons_12m= 53819
    - net_margin= 25.52
    - forecast_meter_rent_12m= 18.02
    - forecast_cons_12m= 294.7
    - margin_net_pow_ele= 21.64
    """)
    st.write("**Sample Prediction Data for the user :red[is a customer]**")
    st.write("""
    - cons_12m= 1591
    - net_margin= 18.25
    - forecast_meter_rent_12m= 16.32
    - forecast_cons_12m= 332.51
    - margin_net_pow_ele= 10.08
    """)
    
    st.write("Note: _The Dataset was provided by the BCG-Forage for experimental purpose_ ")


button = st.button("More Info")
if button:
    with st.expander("See details"):
        st.write("""
                 - The above 5 features are selected as per its importance in affecting the model
                 - EDA and Feature Engineering was performed before bulding the model
                 - Random Forest is used for the model prediction with acuraccy of 90%
                 """
                 )