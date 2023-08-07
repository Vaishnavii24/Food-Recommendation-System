import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix #Used to create a compressed sparse row matrix for efficient storage of ratings data
from sklearn.neighbors import NearestNeighbors

st.title("Food Recommendation System")
st.text("Let us help you with ordering")
st.image("foood.jpg")

st.subheader("Whats your preference?")
vegn = st.radio("Vegetables or none!", ["veg", "non-veg"], index=1)

st.subheader("What Cuisine do you prefer?")
cuisine = st.selectbox("Choose your favorite!", ['Healthy Food', 'Snack', 'Dessert', 'Japanese', 'Indian', 'French',
                                                   'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai'])

val = 8 # ratings which are equal to or greater than 8

#loading and preparing the data
food = pd.read_csv("food.csv")
ratings = pd.read_csv("ratings.csv")
#creating combined dataframe i.e merging food & ratings
combined = pd.merge(ratings, food, on='Food_ID')

#filtering the combined dataframe
ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegn) & (combined.Rating >= val),
                   ['Name', 'C_Type', 'Veg_Non']]
names = ans['Name'].tolist()
ans1 = np.unique(names)

finallist = ""
dish = st.checkbox("Choose your Dish")
if dish:
    finallist = st.selectbox("Our Choices", ans1)

##### IMPLEMENTING RECOMMENDER ######
#transforming ratings into pivot table
dataset = ratings.pivot_table(index='Food_ID', columns='User_ID', values='Rating')
dataset.fillna(0, inplace=True)
#converting pivot table into compressed sparse row(csr) matrix
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)


def food_recommendation(Food_Name, vegn):
    n = 5 #no. of recommendations to return 
    FoodList = food[food['Name'].str.contains(Food_Name)]
    if len(FoodList):
        Foodi = FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0] #This index will be used to access the corresponding row in the csr_dataset.
        distances, indices = model.kneighbors(csr_dataset[Foodi], n_neighbors=n + 1)
        #creating list of tuples representing indices and dist. of nneighbors
        Food_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            #retriving food id of each nneighbor
            Foodi = dataset.iloc[val[0]]['Food_ID']
            # checks if the neighbor is vegetarian or non-vegetarian and if it exists in the food DataFrame.
            i = food[(food['Food_ID'] == Foodi) & (food['Veg_Non'] == vegn)].index
            if len(i) > 0:
                Recommendations.append({'Name': food.iloc[i]['Name'].values[0], 'Distance': val[1]})
                if len(Recommendations) >= n:
                    break  # Exit the loop if desired number of recommendations is reached
        df = pd.DataFrame(Recommendations, index=range(1, len(Recommendations) + 1))
        return df['Name']
    else:
        return "No Similar Foods."



if dish:
    bruh1 = st.checkbox("We also Recommend:")
    if bruh1:
        display = food_recommendation(finallist, vegn)
        for i in display:
            st.write(i)
