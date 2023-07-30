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
        property_types = ['Apartment','Bed & Breakfast','Bungalow', 'Condominium','Dorm','Guesthouse','House', 'Loft','Other','Townhouse']
        selected_property_type = st.selectbox('Select Property Type:', property_types)
        
        # Assign 1 for the selected house type and 0 for the others
        property_type_values = [1 if property_type == selected_property_type else 0 for property_type in property_types]

        st.subheader("Room Type data")
        room_types = ['Entire/home Apt.','Private Room',  'Shared Room']
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
        thumbnail_url = st.selectbox('Would you like to see a picture of the room you will stay in?', ('Yes','No'))
        cleaning_fee_value = 1 if cleaning_fee == 'Yes' else 0
        instant_bookable_value = 1 if instant_bookable == 'Yes' else 0
        thumbnail_url_value = 1 if thumbnail_url == 'Yes' else 0

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
                'thumbnail_url': thumbnail_url_value, 
                'bedrooms': bedrooms,
                'beds': beds,         

                'property_type_Apartment': property_type_values[0],
                'property_type_bed_break': property_type_values[1],
                'property_type_Bungalow': property_type_values[2],
                'property_type_Condominium': property_type_values[3],
                'property_type_Dorm': property_type_values[4],
                'property_type_Guesthouse': property_type_values[5],
                'property_type_House': property_type_values[6],
                'property_type_Loft': property_type_values[7],
                'property_type_Other': property_type_values[8],
                'property_type_Townhouse': property_type_values[9],
                
                'cancellation_policy_flexible': cancellation_policy_values[0],
                'cancellation_policy_moderate': cancellation_policy_values[1],
                'cancellation_policy_strict': cancellation_policy_values[2],
                
                'room_type_entire_home': room_type_values[0],
                'room_type_private_room': room_type_values[1],
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
            st.info(f"Predicted Price: ${round(prediction_df['Prediction'][0])}")

           
            
                    
        
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
          df = pd.read_csv(uploaded_file,encoding= 'utf-8', low_memory=False, index_col=False)
          #Get overview of data
          # st.write(df_batch.head())
          st.write(df.head(5))
          st.markdown("<h3></h3>", unsafe_allow_html=True)
          
        
          df.drop(labels = ["id", "description","first_review","host_has_profile_pic","host_identity_verified","host_since",
                            "zipcode","latitude","longitude","name", "last_review"], axis = 1, inplace = True)
          
          #property_type
          new_columns = [ 'property_type_Apartment',
                          'property_type_bed_break',
                          'property_type_Bungalow',
                          'property_type_Condominium',
                          'property_type_Dorm',
                          'property_type_Guesthouse',
                          'property_type_House',
                          'property_type_Loft',
                          'property_type_Other',
                          'property_type_Townhouse',

                          'room_type_entire_home',
                          'room_type_private_room',
                          'room_type_shared_room',
                          
                          'wireless_internet',
                          'Kitchen',
                          'Heating',
                          'Essentials',
                          'smoke_detector',
                          'air_conditioning',
                          'TV',
                          'Shampoo',
                          'Hangers',
                          'carbon_monoxide_detector',
                          'Internet',
                          'laptop_friendly_workspace',
                          'hair_dryer',
                          'Washer',
                          'Dryer',
                          'Iron',
                          'family_kid_friendly',
                          'fire_extinguister',
                          'first_aid_kit',
                          'cable_tv',
                          'free_parking_on_premises',
                          'alltime_check_in',
                          'lock_on_bedroom_door',
                          'buzzer_wireless,intercom',
                          
                          'bed_type_Airbed',
                          'bed_type_Couch',
                          'bed_type_Futon',
                          'bed_type_Pull_out_Sofa',
                          'bed_type_real_Bed',
                          
                          'cancellation_policy_flexible',
                          'cancellation_policy_moderate',
                          'cancellation_policy_strict',

                          'city_Boston',
                          'city_Chicago',
                          'city_DC',
                          'city_LA',
                          'city_NYC',
                          'city_SF',
                          
                          'neighbourhood_level',]

          df = df.reindex(columns=df.columns.tolist() + new_columns)
          property_types = ['Apartment','Bed & Breakfast','Bungalow', 'Condominium','Dorm','Guesthouse','House', 'Loft','Townhouse']
          #Check the values for each row.
          df['property_type_Apartment'] = df['property_type'].apply(lambda x: 1 if x == 'Apartment' else 0)
          df['property_type_bed_break'] = df['property_type'].apply(lambda x: 1 if x == 'Bed & Breakfast' else 0)
          df['property_type_Bungalow'] = df['property_type'].apply(lambda x: 1 if x == 'Bungalow' else 0)
          df['property_type_Condominium'] = df['property_type'].apply(lambda x: 1 if x == 'Condominium' else 0)
          df['property_type_Dorm'] = df['property_type'].apply(lambda x: 1 if x == 'Dorm' else 0)
          df['property_type_Guesthouse'] = df['property_type'].apply(lambda x: 1 if x == 'Guesthouse' else 0)
          df['property_type_House'] = df['property_type'].apply(lambda x: 1 if x == 'House' else 0)
          df['property_type_Loft'] = df['property_type'].apply(lambda x: 1 if x == 'Loft' else 0)
          df['property_type_Other'] = df['property_type'].apply(lambda x: 1 if x not in property_types else 0)
          df['property_type_Other'] = df['property_type'].apply(lambda x: 1 if x not in property_types else 0)
          df['property_type_Townhouse'] = df['property_type'].apply(lambda x: 1 if x == 'Townhouse' else 0)

          #room_type
          df['room_type_entire_home'] = df['room_type'].apply(lambda x: 1 if x == 'Entire home/apt' else 0)
          df['room_type_private_room'] = df['room_type'].apply(lambda x: 1 if x == 'Private room' else 0)
          df['room_type_shared_room'] = df['room_type'].apply(lambda x: 1 if x == 'Shared room' else 0)

          #amenities

          #cleaning data
          df['amenities'] = df['amenities'].map(lambda x: x.replace('"', '').replace('{', '').replace('}', '').split(','))

          #make list
          amenities = {x for xs in df['amenities'].tolist() for x in xs}
          
          #Check the values for each row.
          df['wireless_internet'] = df['amenities'].apply(lambda x: 1 if 'Wireless Internet' in x else 0)
          df['Kitchen'] = df['amenities'].apply(lambda x: 1 if 'Kitchen' in x else 0)
          df['Heating'] = df['amenities'].apply(lambda x: 1 if 'Heating' in x else 0)
          df['Essentials'] = df['amenities'].apply(lambda x: 1 if 'Essentials' in x else 0)
          df['smoke_detector'] = df['amenities'].apply(lambda x: 1 if 'Smoke detector' in x else 0)
          df['air_conditioning'] = df['amenities'].apply(lambda x: 1 if 'Air conditioning' in x else 0)
          df['TV'] = df['amenities'].apply(lambda x: 1 if 'TV' in x else 0)
          df['Shampoo'] = df['amenities'].apply(lambda x: 1 if 'Shampoo' in x else 0)
          df['Hangers'] = df['amenities'].apply(lambda x: 1 if 'Hangers' in x else 0)
          df['carbon_monoxide_detector'] = df['amenities'].apply(lambda x: 1 if 'Carbon monoxide detector' in x else 0)
          df['Internet'] = df['amenities'].apply(lambda x: 1 if 'Internet' in x else 0)
          df['laptop_friendly_workspace'] = df['amenities'].apply(lambda x: 1 if 'Laptop friendly workspace' in x else 0)
          df['hair_dryer'] = df['amenities'].apply(lambda x: 1 if 'Hair dryer' in x else 0)
          df['Washer'] = df['amenities'].apply(lambda x: 1 if 'Washer' in x else 0)
          df['Dryer'] = df['amenities'].apply(lambda x: 1 if 'Dryer' in x else 0)
          df['Iron'] = df['amenities'].apply(lambda x: 1 if 'Iron' in x else 0)
          df['family_kid_friendly'] = df['amenities'].apply(lambda x: 1 if 'Family/kid friendly' in x else 0)
          df['fire_extinguister'] = df['amenities'].apply(lambda x: 1 if 'Fire extinguisher' in x else 0)
          df['first_aid_kit'] = df['amenities'].apply(lambda x: 1 if 'First aid kit' in x else 0)
          df['cable_tv'] = df['amenities'].apply(lambda x: 1 if 'Cable TV' in x else 0)
          df['free_parking_on_premises'] = df['amenities'].apply(lambda x: 1 if 'Free parking on premises' in x else 0)
          df['alltime_check_in'] = df['amenities'].apply(lambda x: 1 if '24-hour check-in' in x else 0)
          df['lock_on_bedroom_door'] = df['amenities'].apply(lambda x: 1 if 'Lock on bedroom door' in x else 0)
          df['buzzer_wireless,intercom'] = df['amenities'].apply(lambda x: 1 if 'Buzzer/wireless intercom' in x else 0)
         
          #accommodates column will not be processed.

          #bathrooms column will not be processed.
          
          #bed_type
          df['bed_type_Airbed'] = df['bed_type'].apply(lambda x: 1 if x == 'Airbed' else 0)
          df['bed_type_Couch'] = df['bed_type'].apply(lambda x: 1 if x == 'Couch' else 0)
          df['bed_type_Futon'] = df['bed_type'].apply(lambda x: 1 if x == 'Futon' else 0)
          df['bed_type_Pull_out_Sofa'] = df['bed_type'].apply(lambda x: 1 if x == 'Pull-out Sofa' else 0)
          df['bed_type_real_Bed'] = df['bed_type'].apply(lambda x: 1 if x == 'Real Bed' else 0)
          
          #cancellation_policy
          df['cancellation_policy_flexible'] = df['cancellation_policy'].apply(lambda x: 1 if x == 'flexible' else 0)
          df['cancellation_policy_moderate'] = df['cancellation_policy'].apply(lambda x: 1 if x == 'moderate' else 0)
          df['cancellation_policy_strict'] = df['cancellation_policy'].apply(lambda x: 1 if x == 'strict' else 0)

          #cleaning_fee
          df['cleaning_fee'] = df['cleaning_fee'].astype(int)
          
          #city
          df['city_Boston'] = df['city'].apply(lambda x: 1 if x == 'Boston' else 0)
          df['city_Chicago'] = df['city'].apply(lambda x: 1 if x == 'Chicago' else 0)
          df['city_DC'] = df['city'].apply(lambda x: 1 if x == 'DC' else 0)
          df['city_LA'] = df['city'].apply(lambda x: 1 if x == 'LA' else 0)
          df['city_NYC'] = df['city'].apply(lambda x: 1 if x == 'NYC' else 0)
          df['city_SF'] = df['city'].apply(lambda x: 1 if x == 'SF' else 0)
          
          #host_response_rate
          df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(int)

          #instant_bookable
          df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})

          #neighbourhood & neighbourhood_level
          neighbourhood_city_df = pd.read_csv('neighbourhood_city.csv')
          neighbourhood_levels = []

          for neighbourhood in df["neighbourhood"]:
             # Search for "neighborhood" value in DataFrame "neighborhood_city_df"  
              result = neighbourhood_city_df.loc[neighbourhood_city_df["neighbourhood"] == neighbourhood, "neighbourhood-level"]              
              # If there are results, add the corresponding value to the "neighborhood_levels" list 
              if not result.empty:
                  neighbourhood_levels.append(int(result.iloc[0]))
              
              else:  # If no result, insert null               
                  neighbourhood_levels.append(1)

           # Add "neighborhood_level" values to df
          df["neighbourhood_level"] = neighbourhood_levels

          #number_of_reviews column will not be processed.

          #review_scores_rating column will not be processed.

          #thumbnail_url
          df["thumbnail_url"] = df["thumbnail_url"].notnull().astype(int)
          
          #bedrooms column will not be processed.
          
          #beds column will not be processed.
          column_names = [
                'accommodates', 'bathrooms', 'cleaning_fee', 'host_response_rate',
                'instant_bookable', 'number_of_reviews', 'review_scores_rating', 'thumbnail_url',
                'bedrooms', 'beds', 'property_type_Apartment', 'property_type_bed_break',
                'property_type_Bungalow', 'property_type_Condominium', 'property_type_Dorm',
                'property_type_Guesthouse', 'property_type_House', 'property_type_Loft',
                'property_type_Other', 'property_type_Townhouse', 'cancellation_policy_flexible',
                'cancellation_policy_moderate', 'cancellation_policy_strict', 'room_type_entire_home',
                'room_type_private_room', 'room_type_shared_room', 'bed_type_Airbed', 'bed_type_Couch',
                'bed_type_Futon', 'bed_type_Pull_out_Sofa', 'bed_type_real_Bed', 'city_Boston',
                'city_Chicago', 'city_DC', 'city_LA', 'city_NYC', 'city_SF', 'wireless_internet',
                'Kitchen', 'Heating', 'Essentials', 'smoke_detector', 'air_conditioning', 'TV',
                'Shampoo', 'Hangers', 'carbon_monoxide_detector', 'Internet', 'laptop_friendly_workspace',
                'hair_dryer', 'Washer', 'Dryer', 'Iron', 'family_kid_friendly', 'fire_extinguister',
                'first_aid_kit', 'cable_tv', 'free_parking_on_premises', 'alltime_check_in',
                'lock_on_bedroom_door', 'buzzer_wireless,intercom', 'neighbourhood_level'
                ]

# Tüm sütunları döngü ile integer veri türüne dönüştürüyoruz
          for col in column_names:
                    df[col] = df[col].astype(int)
          data = {
                'accommodates': df['accommodates'],
                'bathrooms': df['bathrooms'],
                'cleaning_fee': df['cleaning_fee'],
                'host_response_rate': df['host_response_rate'],
                'instant_bookable': df['instant_bookable'],
                'number_of_reviews' : df['number_of_reviews'],
                'review_scores_rating'  :df['review_scores_rating'],
                'thumbnail_url': df['thumbnail_url'], 
                'bedrooms': df['bedrooms'],
                'beds': df['beds'],         

                'property_type_Apartment': df['property_type_Apartment'],
                'property_type_bed_break': df['property_type_bed_break'],
                'property_type_Bungalow': df['property_type_Bungalow'],
                'property_type_Condominium': df['property_type_Condominium'],
                'property_type_Dorm': df['property_type_Dorm'],
                'property_type_Guesthouse': df['property_type_Guesthouse'],
                'property_type_House': df['property_type_House'],
                'property_type_Loft': df['property_type_Loft'],
                'property_type_Other': df['property_type_Other'],
                'property_type_Townhouse': df['property_type_Townhouse'],
                
                'cancellation_policy_flexible': df['cancellation_policy_flexible'],
                'cancellation_policy_moderate': df['cancellation_policy_moderate'],
                'cancellation_policy_strict': df['cancellation_policy_strict'],
                
                'room_type_entire_home': df['room_type_entire_home'],
                'room_type_private_room': df['room_type_private_room'],
                'room_type_shared_room': df['room_type_shared_room'],
               
                'bed_type_Airbed': df['bed_type_Airbed'],
                'bed_type_Couch': df['bed_type_Couch'],
                'bed_type_Futon': df['bed_type_Futon'],
                'bed_type_Pull_out_Sofa': df['bed_type_Pull_out_Sofa'],
                'bed_type_real_Bed': df['bed_type_real_Bed'],

                'city_Boston': df['city_Boston'],
                'city_Chicago': df['city_Chicago'],
                'city_DC': df['city_DC'],
                'city_LA': df['city_LA'],
                'city_NYC': df['city_NYC'],
                'city_SF': df['city_SF'],


                'wireless_internet': df['wireless_internet'],
                'Kitchen': df['Kitchen'],
                'Heating': df['Heating'],
                'Essentials': df['Essentials'],
                'smoke_detector': df['smoke_detector'],
                'air_conditioning': df['air_conditioning'],
                'TV': df['TV'],
                'Shampoo': df['Shampoo'],
                'Hangers': df['Hangers'],
                'carbon_monoxide_detector': df['carbon_monoxide_detector'],
                'Internet': df['Internet'],
                'laptop_friendly_workspace': df['laptop_friendly_workspace'],
                'hair_dryer': df['hair_dryer'],
                'Washer': df['Washer'],
                'Dryer': df['Dryer'],
                'Iron': df['Iron'],
                'family_kid_friendly': df['family_kid_friendly'],
                'fire_extinguister': df['fire_extinguister'],
                'first_aid_kit': df['first_aid_kit'],
                'cable_tv': df['cable_tv'],
                'free_parking_on_premises': df['free_parking_on_premises'],
                'alltime_check_in': df['alltime_check_in'],
                'lock_on_bedroom_door': df['lock_on_bedroom_door'],
                'buzzer_wireless,intercom': df['buzzer_wireless,intercom'],

                'neighbourhood_level' : df['neighbourhood_level'],
                
                }
          features_df = pd.DataFrame(data)
          #features_df = pd.DataFrame.from_dict([data])
          st.markdown("<h3></h3>", unsafe_allow_html=True)
        #   st.write('Overview of input is shown below')
        #   st.markdown("<h3></h3>", unsafe_allow_html=True)
        #   st.dataframe(features_df)
          

   
          if st.button('Predict'):
              
              prediction = model.predict(features_df)

              # Converting log prices to regular prices
              predicted_prices = np.exp(prediction)
              prediction_df = pd.DataFrame(predicted_prices, columns=["Prediction"])

              st.markdown("<h3>Prediction Result:</h3>", unsafe_allow_html=True)
              st.info(f"Predicted Price: ${round(prediction_df['Prediction'])}")
              #Get batch prediction

            
            
if __name__ == '__main__':
        main()
