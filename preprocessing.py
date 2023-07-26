#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:05:05 2022

@author: demir
"""

import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

def preprocess(df, option):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0})
    
    #Drop values based on operational options
    if (option == "Online"):
        # Encode binary categorical features
        binary_list = ['Title_0','Title_1','Title_2','Title_3','family_size_0','family_size_1','Embarked_C','Embarked_Q','Embarked_S','T_A','T_A4',
                   'T_A5','T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON', 'T_FC', 'T_FCC','T_Fa','T_LINE', 'T_LP', 
                   'T_PC','T_PP', 'T_PPP', 'T_SC', 'T_SCA3','T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS', 'T_SCParis', 'T_SOC','T_SOP',
                   'T_SOPP',  'T_SOTONO2',  'T_SOTONOQ', 'T_SP', 'T_STONO','T_STONO2','T_STONOQ', 'T_SWPP', 'T_WC', 'T_WEP', 'T_x',
                   'Pclass_1','Pclass_2','Pclass_3', 'Sex_female', 'Sex_male']
        df[binary_list] = df[binary_list].apply(binary_map)
        columns = ['Age','Sibsp','Parch','Fare',
                   'Title_0','Title_1','Title_2','Title_3','Fsize','family_size_0','family_size_1','Embarked_C','Embarked_Q','Embarked_S','T_A','T_A4',
                   'T_A5','T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON', 'T_FC', 'T_FCC','T_Fa','T_LINE', 'T_LP', 
                   'T_PC','T_PP', 'T_PPP', 'T_SC', 'T_SCA3','T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS', 'T_SCParis', 'T_SOC','T_SOP',
                   'T_SOPP',  'T_SOTONO2',  'T_SOTONOQ', 'T_SP', 'T_STONO','T_STONO2','T_STONOQ', 'T_SWPP', 'T_WC', 'T_WEP', 'T_x',
                   'Pclass_1','Pclass_2','Pclass_3', 'Sex_female', 'Sex_male']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        #print(df.head())
    elif (option == "Batch"):
        #pass
        #Drop Passenger ID and Cabin
        df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
        #print(df.isna().sum().sum())
        #Name title operatiom
        name = df["Name"]
        df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
        df["Title"] = df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
        df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in df["Title"]]
        #print("test",df["Title"])
        df.drop(labels = ["Name"], axis = 1, inplace = True)
        df = pd.get_dummies(df,columns=["Title"])
        #family size operation
        df["Fsize"] = df["SibSp"] + df["Parch"] + 1
        df["family_size"] = [1 if i < 5 else 0 for i in df["Fsize"]]
        df = pd.get_dummies(df, columns= ["family_size"])
        
        #embark operation
        df = pd.get_dummies(df, columns=["Embarked"])
        #print(df.head())
        #ticket operation
        tickets = []
        for i in list(df.Ticket):
            if not (str(i).isdigit()):
                tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
            else:
                tickets.append("x")
        df["Ticket"] = tickets
        df = pd.get_dummies(df, columns= ["Ticket"], prefix = "T")
        #print(df.head())
        #pclass operation
        df["Pclass"] = df["Pclass"].astype("category")
        df = pd.get_dummies(df, columns= ["Pclass"])
        #gender operation
        #df["Sex"] = df["Sex"].astype("category")
        df = pd.get_dummies(df, columns=["Sex"])
        #print(df.head())
        columns = ['Age','Sibsp','Parch','Fare',
                   'Title_0','Title_1','Title_2','Title_3','Fsize','family_size_0','family_size_1','Embarked_C','Embarked_Q','Embarked_S','T_A','T_A4',
                   'T_A5','T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON', 'T_FC', 'T_FCC','T_Fa','T_LINE', 'T_LP', 
                   'T_PC','T_PP', 'T_PPP', 'T_SC', 'T_SCA3','T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS', 'T_SCParis', 'T_SOC','T_SOP',
                   'T_SOPP',  'T_SOTONO2',  'T_SOTONOQ', 'T_SP', 'T_STONO','T_STONO2','T_STONOQ', 'T_SWPP', 'T_WC', 'T_WEP', 'T_x',
                   'Pclass_1','Pclass_2','Pclass_3', 'Sex_female', 'Sex_male']
        #Encoding the other categorical categoric features with more than two categories
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        #print(df.head())
    else:
        print("Incorrect operational options")


    #feature scaling
    #sc = MinMaxScaler()
    #df['tenure'] = sc.fit_transform(df[['tenure']])
    #df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    #df['TotalCharges'] = sc.fit_transform(df[['TotalCharges']])
    return df
        
