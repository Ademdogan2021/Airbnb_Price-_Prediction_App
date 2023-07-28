#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 07:03:58 2022

@author: NETA
"""

#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image



#load the model from disk
import joblib
filename = 'finalized_model.sav'
model = joblib.load(filename)

#Import python scripts
# from preprocessing import preprocess


def main():
    #Setting Application title
    st.title('NETA Airbnb Model App')

      #Setting Application description
    st.markdown("""
     :dart: Airbnb is a popular accommodation sharing platform where travelers can discover
    unique lodging options worldwide, while hosts can rent out their properties to earn extra income.\n
     :dart:  This user-friendly app is designed for Airbnb price prediction. 
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.png')
    # image1 = Image.open('importance.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Airbnb use case')
    st.sidebar.image(image)
    st.sidebar.info('This app uses Gradient Boosting Model (GBM)')
    #st.sidebar.image(image1)

    if add_selectbox == "Online":
        st.info("Input data below")
        # Based on our optimal features selection

        st.subheader("Property Type data")
        property_types = ['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft', 'Guesthouse', 'Bed & Breakfast', 'Bungalow', 'Villa', 'Other']
        selected_property_type = st.selectbox('Select Property Type:', property_types)
        
        # Assign 1 for the selected house type and 0 for the others
        property_type_values = [1 if property_type == selected_property_type else 0 for property_type in property_types]

        st.subheader("Room Type data")
        room_types = ['Private Room', 'Entire/home Apt.', 'Shared Room']
        selected_room_type = st.selectbox('Select Room Type:', room_types)

        # Assign 1 for the selected room type and 0 for the others
        room_type_values = [1 if room_type == selected_room_type else 0 for room_type in room_types]

        st.subheader("Amenities data")        
        amenities = [ 'Wireless Internet', 'Kitchen', 'Heating','Essentials','Smoke Detector','Aie Conditioning','TV','Shampoo','Hangers','Carbon Monoxide Detector','Internet','Laptop Friendly Workspace','Hair Dryer','Washer','Dryer','Iron','Famili/Kid Friendly','Fire Extinguister','First Aid Kit','Cable TV','Freeparking on Premises','24 Hour Check in','Lock on Bedroom Door','Buzzer Wireless Intercom']
        selected_amenities = st.multiselect('Select Amenities:', amenities)

        # Assign 1 for the selected amenities and 0 for the others
        amenities_values = [1 if amenity in selected_amenities else 0 for amenity in amenities]

        st.subheader("Information About Place") 
        
        accommodates = st.number_input('How many Guests can you host? ', min_value=1, max_value=16, value=2)
        bathrooms = st.number_input('How many Bathrooms do you have?', min_value=0, max_value=8, value=1)
        bedrooms = st.number_input('How many Bedrooms do you have?' , min_value=0, max_value=10, value=1)
        beds = st.number_input('How many Beds do you have?', min_value=0, max_value=18, value=1)
        cleaning_fee = st.selectbox('Is there an additional cleaning fee for the accommodation?',('No','Yes'))
        instant_bookable = st.selectbox('Is your place have instant bookable?', ('No','Yes'))
        cleaning_fee_value = 1 if cleaning_fee == 'Yes' else 0
        instant_bookable_value = 1 if instant_bookable == 'Yes' else 0

        st.subheader("Information About Beds") 
        bed_types = ['Air Bed', 'Couch', 'Futon','Pull-out Sofa','Real Bed']
        selected_bed_type = st.selectbox('Select Room Type:', bed_types)

        # Assign 1 for the selected bed_type and 0 for the others
        bed_type_values = [1 if bed_type == selected_bed_type else 0 for bed_type in bed_types]

        neighbourhood_df = pd.read_csv('neighbourhood_city.csv')

        st.subheader("Location data")
        cities = neighbourhood_df['city'].unique()
        selected_city = st.selectbox('Select City:', cities)

        # Bring the neighborhoods of the selected city
        selected_neighbourhoods = neighbourhood_df.loc[neighbourhood_df['city'] == selected_city, 'neighbourhood'].tolist()

        # Request user to select neighborhood
        selected_neighbourhood = st.selectbox('Select Neighbourhood:', selected_neighbourhoods)
        # Get the corresponding neighbourhood-level for the selected neighbourhood
        selected_neighbourhood_level = neighbourhood_df.loc[neighbourhood_df['neighbourhood'] == selected_neighbourhood, 'neighbourhood-level'].iloc[0]     
        
        # Assign 1 for the selected city and 0 for the others
        city_values = [1 if city == selected_city else 0 for city in cities]

        st.subheader("Cancellation Policy")
        cancellation_policies = ['Flexible', 'Moderate', 'Strict']
        selected_cancellation_policy = st.selectbox('Select Cancellation Policy:', cancellation_policies)

        cancellation_policy_values = [1 if policy == selected_cancellation_policy else 0 for policy in cancellation_policies]       
        
        data = {
                'accommodates': accommodates,
                'bathrooms': bathrooms,
                'cleaning_fee': cleaning_fee_value,
                'host_response_rate': 94, # 0,0064
                'instant_bookable': instant_bookable_value,
                'number_of_reviews' : 20, # 0,036
                'review_scores_rating'  :94, # 0,078
                'thumbnail_url': 1, # kodu yaz 
                'bedrooms': bedrooms,
                'beds': beds,         

                'property_type_Apartment': property_type_values[0],
                'property_type_bed_break': property_type_values[6],
                'property_type_Bungalow': property_type_values[7],
                'property_type_Condominium': property_type_values[2],
                'property_type_Dorm': property_type_values[8],
                'property_type_Guesthouse': property_type_values[5],
                'property_type_House': property_type_values[1],
                'property_type_Loft': property_type_values[4],
                'property_type_Other': property_type_values[9],
                'property_type_Townhouse': property_type_values[3],
                
                'cancellation_policy_flexible': cancellation_policy_values[0],
                'cancellation_policy_moderate': cancellation_policy_values[1],
                'cancellation_policy_strict': cancellation_policy_values[2],
                
                'room_type_entire_home': room_type_values[1],
                'room_type_private_room': room_type_values[0],
                'room_type_shared_room': room_type_values[2],

                'bed_type_Airbed': bed_type_values[0],
                'bed_type_Couch': bed_type_values[1],
                'bed_type_Futon': bed_type_values[2],
                'bed_type_Pull_out_Sofa': bed_type_values[3],
                'bed_type_real_Bed': bed_type_values[4],

                'city_Boston': city_values[0],
                'city_Chicago': city_values[1],
                'city_DC': city_values[2],
                'city_LA': city_values[3],
                'city_NYC': city_values[4],
                'city_SF': city_values[5],


                'wireless_internet': amenities_values[0],
                'Kitchen': amenities_values[1],
                'Heating': amenities_values[2],
                'Essentials': amenities_values[3],
                'smoke_detector': amenities_values[4],
                'air_conditioning': amenities_values[5],
                'TV': amenities_values[6],
                'Shampoo': amenities_values[7],
                'Hangers': amenities_values[8],
                'carbon_monoxide_detector': amenities_values[9],
                'Internet': amenities_values[10],
                'laptop_friendly_workspace': amenities_values[11],
                'hair_dryer': amenities_values[12],
                'Washer': amenities_values[13],
                'Dryer': amenities_values[14],
                'Iron': amenities_values[15],
                'family_kid_friendly': amenities_values[16],
                'fire_extinguister': amenities_values[17],
                'first_aid_kit': amenities_values[18],
                'cable_tv': amenities_values[19],
                'free_parking_on_premises': amenities_values[20],
                'alltime_check_in': amenities_values[21],
                'lock_on_bedroom_door': amenities_values[22],
                'buzzer_wireless,intercom': amenities_values[23],

                'neighbourhood_level' : selected_neighbourhood_level,
                }
        
        
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        # preprocess_df = preprocess(features_df, 'Online')

        # prediction = model.predict(preprocess_df)
        
     

        if st.button('Predict'):
            
            # Preprocess the input data
            # preprocess_df = preprocess(features_df, 'Online')

            # Make a price prediction
            prediction = model.predict(features_df)

            # Converting log prices to regular prices
            predicted_prices = np.exp(prediction)
            prediction_df = pd.DataFrame(predicted_prices, columns=["Prediction"])

            st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
            st.info(f"Predicted Price: {round(prediction_df['Prediction'][0])}")

            st.subheader("Prediction Result")
            st.dataframe(prediction_df)
            
                    
        

    # else:
    #     st.subheader("Dataset upload")
    #     uploaded_file = st.file_uploader("Choose a file")
    #     if uploaded_file is not None:
    #         data = pd.read_csv(uploaded_file,encoding= 'utf-8')
    #         #Get overview of data
    #         st.write(data.head())
    #         st.markdown("<h3></h3>", unsafe_allow_html=True)
    #         #Preprocess inputs
    #         preprocess_df = preprocess(data, "Batch")
    #         if st.button('Predict'):
    #             #Get batch prediction
    #             prediction = model.predict(preprocess_df)
    #             prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
    #             # prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

    #             st.markdown("<h3></h3>", unsafe_allow_html=True)
    #             st.subheader('Prediction')
    #             st.write(prediction_df)
            
if __name__ == '__main__':
        main()
