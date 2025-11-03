import pickle
import streamlit as st
import requests


def recommend(ani):
  index=animes[animes['name']==ani].index[0]
  distance=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x:x[1])
  recommended_anime_name, recommended_anime_posters = [], []
  for i in distance[1:6]:
    recommended_anime_name.append(animes['name'][i[0]])
    recommended_anime_posters.append(animes['image'][i[0]])
  return recommended_anime_name, recommended_anime_posters


st.header("Anime Recommendation System")
animes = pickle.load(open("artifacts/anime_list.pkl", "rb"))
similarity = pickle.load(open("artifacts/similarity.pkl", "rb"))

anime_list = animes['name'].values
selected_anime = st.selectbox (
    'Type or select an anime to get recommendations', 
    anime_list

)

if st.button('Show Recommendation'):
    recommended_anime_name, recommended_anime_posters = recommend(selected_anime)
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(5)
    with col1:
        st.text(recommended_anime_name[0])
        st.image(recommended_anime_posters[0])
    with col2:
        st.text(recommended_anime_name[1])
        st.image(recommended_anime_posters[1])
    with col3:
        st.text(recommended_anime_name[2])
        st.image(recommended_anime_posters[2])
    with col4:
        st.text(recommended_anime_name[3])
        st.image(recommended_anime_posters[3])
    with col5:
        st.text(recommended_anime_name[4])
        st.image(recommended_anime_posters[4])