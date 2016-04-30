# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:37:15 2016

@author: cweinshenker
"""

import numpy as np
import pandas as pd
import sys
import math
import datetime, time

"""
This file cleans the data for the pet classification
by creating binary variables for all the base classification features,
e.g., color, breed, and outcomes
Well over 2000 distinct breed and color categories
are pared to just under 300 base variables
"""

def add_color_columns(data):
    colors_unique = data.Color.unique()
    for color in colors_unique:
        colors = color.split("/")
        for color in colors:
            base_color = color.split()[0].capitalize()
            data[base_color] = [0 for i in range(data.shape[0])]

def add_breed_columns(data):
    breeds_unique = data.Breed.unique()
    for breed in breeds_unique:
        breeds = breed.split("/")
        for breed in breeds:
            if breed[-3:] == "Mix":
                base_breed = " ".join(breed.split()[:-1])
            else:
                base_breed = breed
            data[base_breed] = [0 for i in range(data.shape[0])]
    data["Mix"] = [0 for i in range(data.shape[0])]


def get_outcomes_dict(train):
    outcomes_dict = {}
    outcomes_unique = train.OutcomeType.unique()
    for i in range(len(outcomes_unique)):
        outcomes_dict[outcomes_unique[i]] = i
    return outcomes_dict
    

def get_outcomes_subtype_dict(train):
    outcomes_subtype_dict = {}
    outcomes_subtypes_unique = train.OutcomeSubtype.unique()
    for i in range(len(outcomes_subtypes_unique)):
        outcomes_subtype_dict[outcomes_subtypes_unique[i]] = i
    return outcomes_subtype_dict

def numerify_outcome(i, outcome_dict, data):
    data.Outcome_Numeric[i] = outcome_dict[data.OutcomeType.iloc[i]]

def numerify_outcome_subtype(i, outcome_subtype_dict, data):
    data.Outcome_Subtype_Numeric[i] = outcome_subtype_dict[data.OutcomeSubtype.iloc[i]]

def numerify_date(i, data):
    date = data.DateTime[i].split()
    month_day_year = date[0].split("-")
    print(month_day_year)
    minutes_seconds = date[1].split(":")
    print(minutes_seconds)
    month = int(month_day_year[1])
    day = int(month_day_year[2])
    year = int(month_day_year[0])
    minutes = int(minutes_seconds[0])
    seconds = int(minutes_seconds[1])
    date = datetime.datetime(year, month, day, minutes, seconds)
    seconds = time.mktime(date.timetuple())
    data["DateTime_Numeric"][i] = seconds 
    

def reassign_color(i, data):
     colors = data.Color[i].split("/")
     for color in colors:
         base_color = color.split()[0].capitalize()
         data[base_color][i] = 1

def reassign_breed(i, data):
    breeds  = data.Breed[i].split("/")
    for breed in breeds:
        breed = breed.split()
        if breed[-1] == "Mix":
            data.Mix[i] = 1
            base_breed = " ".join(breed[:-1])
        else:
            base_breed = " ".join(breed)                
        data[base_breed][i] = 1
    

def assign_named(i, data):
    if type(data.Name[i]) is str:
        data["Named"][i] = 1
    else:
        data["Named"][i] = 0
        
        
def numerify_age(i, data):
    """Standardize ages as a continous variable with years as the unit"""
    age_list = data.AgeuponOutcome[i].split()
    age = float(age_list[0])
    if age_list[1].lower() == "weeks" or age_list[1].lower() == "week":
        age = age/52
    elif age_list[1].lower() == "months" or age_list[1].lower() == "months":
        age = age/12
    elif age_list[1].lower() == "days" or age_list[1].lower() == "day":
        age = age/365
    print(age)
    data.AgeuponOutcome_Numeric[i] = age
    
def main():
    train = pd.read_csv("train.csv", header = 0)
    #Create new columns for categorical outcomes (numeric)
    train["Outcome_Numeric"] = [0 for i in range(train.shape[0])]
    train["Outcome_Subtype_Numeric"] = [0 for i in range(train.shape[0])]
    #Create dummies for sexuponoutcome, and animaltype
    for column in ['AnimalType', 'SexuponOutcome']: 
        dummies = pd.get_dummies(train[column])
        train[dummies.columns] = dummies
    train["Named"] = [0 for i in range(train.shape[0])]
     #Create numeric variables for strings
    train["DateTime_Numeric"] = [float(0) for i in range(train.shape[0])]
    train["AgeuponOutcome_Numeric"] = [float(0) for i in range(train.shape[0])]
    #Create new columns for dummy variables and 
    add_color_columns(train)
    add_breed_columns(train)
    #Get dictionaries for outcomes and subtype outcomes
    outcomes_dict = get_outcomes_dict(train)
    outcomes_subtype_dict = get_outcomes_subtype_dict(train)
    #Reassign colors, breeds, and named to dummy variables in one pass over the data
    for i in range(10):
        numerify_date(i, train)
        numerify_outcome(i, outcomes_dict, train)
        numerify_outcome_subtype(i, outcomes_subtype_dict, train)
        reassign_color(i, train)
        reassign_breed(i, train)
        assign_named(i, train)
        numerify_age(i, train)
    print(train.shape)
    train.to_csv("train_cleaned.csv")
    
    
    
main()



        

