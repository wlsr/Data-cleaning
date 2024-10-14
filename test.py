import pandas as pd
import matplotlib.pyplot as plt
from unidecode import unidecode

clean_data = pd.read_csv('data/rotten_tomatoes_movies_clean.csv')
rating_perfect = clean_data[(clean_data['tomatometer_rating'] ==100) & (clean_data['audience_rating'] == 100)]
print(rating_perfect[rating_perfect['tomatometer_status'] == 'fresh'])

# Descomponer la cadena de generos en una lista
df_normalized = clean_data[clean_data['directors'] != 'nr'].copy()

df_normalized['genres_normalized'] = df_normalized['genres'].str.split(',') 

# Normalized genres
df_normalized['genres_normalized'] = df_normalized['genres_normalized'].apply(lambda x: [unidecode(genre.strip()) for genre in x])

# Descomponer la cadena directores en una lista

df_normalized['directors_normalized'] = df_normalized['directors'].str.split(',')   

# Normalized directors
df_normalized['directors_normalized'] = df_normalized['directors_normalized'].apply(lambda x: [unidecode(director.strip()) for director in x])

# Explotar la lista de géneros
genres = df_normalized['genres_normalized'].explode().str.strip()  # Descompone la lista de géneros

# Explotar la lista de directores
directors = df_normalized['directors_normalized'].explode().str.strip().str.title()

# Contar la frecuencia de cada género
genre_counts = genres.value_counts()
# Contar la frecuencia de cada director
top_directors = directors.value_counts().nlargest(15)

# # Graficar los 10 directores con mejor calificación promedio
# plt.figure(figsize=(10,6))
# plt.barh(top_directors.index, top_directors.values, color='purple', alpha=0.7)
# plt.xlabel('Number of movies')
# plt.ylabel('Director')
# plt.title('Top 15 Directors with the Most Movies Directed')
# plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar el mejor director en la parte superior
# plt.tight_layout()
# plt.show()
# top_15 = df_normalized.explode('genres_normalized').groupby('genres_normalized')['audience_rating'].median().nlargest(15)
# top_15.plot(kind='barh', color='red', alpha=0.7)
# plt.title('Top 15 Géneros con Mejor Audience Rating')
# plt.xlabel('Género')
# plt.ylabel('Median Audience Rating')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.gca().invert_yaxis() 
# plt.show()






