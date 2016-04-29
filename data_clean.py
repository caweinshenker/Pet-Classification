# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:37:15 2016

@author: cweinshenker
"""

import numpy as np
import pandas as pd

"""
This file cleans the data for the pet classification
by creating binary variables for all the base classification features,
e.g., color, breed, and outcomes
Well over 1700 distinct breed and color categories
are pared to just over 100 base variables
"""

def reassign_color(i, found_colors, data):
    base_colors = []
    colors = data.Color.iloc[i].split("/")
    for color in colors:
        base_color = color.split()[0]
        base_colors.append(base_color)
    for color in base_colors:
        color = color.capitalize()
        if color not in found_colors:
            found_colors.add(color)
            data[color] = [0 for i in range(data.shape[0])]
        data[color][i] = 1
        
def reassign_breed(i, found_breeds, data):
    breeds = data.Breed.iloc[i].split()
    base_breed = " ".join(breeds[:-1])
    print(base_breed)
    mixed = (breeds[-1] == "Mix")
    if mixed:
        data["Mix"][i] = 1
    if base_breed not in found_breeds:
        found_breeds.add(base_breed)
        data[base_breed] = [0 for i in range(data.shape[0])]
    data[base_breed][i] = 1


def assign_named(i, data):
    if data.Name.iloc[i] == None:
        data["Named"] = 0
    else:
        data["Named"] = 1
        



def main():

    train = pd.read_csv("train.csv", header = 0)
    #Create dummies for outcomes, outcome subtypes, color, sexuponoutcome, breed
    for column in ['OutcomeType', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome']: 
        dummies = pd.get_dummies(train[column])
        train[dummies.columns] = dummies
    #Reassign colors, breeds, and named to dummy variables in one pass over the data
    #Track found breeds and colors
    found_colors = set()
    found_breeds = set()
    #Mixed breeds show up often enough that it is worth creating a separate variable for it
    train["Mix"] = [0 for i in range(train.shape[0])]
    train["Named"] = [0 for i in range(train.shape[0])]
    #The idea is to pare complex colors down to their base colors--same with breeds
    for i in range(train.shape[0]):
        reassign_color(i, found_colors, train)
        reassign_breed(i, found_breeds, train)
        assign_named(i, train)
        
main()



        

