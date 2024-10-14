import pandas as pd
from pandasgui import show
from rapidfuzz import process, fuzz
from joblib import Parallel, delayed
from collections import Counter
import time
import re
from unidecode import unidecode
from colorama import Style,Fore, init
init()
# https://deepnote.com/app/film-data-management-project/Rating-movies-data-cleaning-9fedf1da-d519-4e3a-9f03-5cc227c7f4f5
#################################################### FUNCTIONS ######################################
def validate_change_date(df, columns, fill_date, amount_null):
    """
    Validate the data filled with the date 1900-01-01 is the same that null data
    """ 
    count = (df[columns] == fill_date).sum().sum()
    if count == amount_null:
        return 'OK'
    else:
        return f"Fill data error: {count} != {amount_null}"
    
def transform_to_string(df, value):
    """
    Transform type of data in a column to string.
    This function assumes that each cell in the specified columns contains an iterable (e.g., a list).
    It sorts the elements in each iterable and joins them into a single string, separated by commas.
    """
    for obj in value:
        df[obj] = df[obj].apply(lambda x: ', '.join(sorted(x)))
    
def normalize_category_fuzzy(value, choices, threshold=80):
    """ 
    Normalizes a given string value by finding the closest match in a list of choices 
    using fuzzy matching with rapidfuzz.
    """
    if isinstance(value, str):
        best_match = process.extractOne(value, choices)     # get the best match and its score
        if best_match and best_match[1] >= threshold:       # Ensure the score is sufficient
            return best_match[0]                            # Return the name of the best match
    return 'nr'                                             # Return 'nr' if no suitable match was found

def fill_date_null(df, dates, date_fict):
    """
    Fill null dates in specified columns of a DataFrame with a fictitious date
    """ 
    for col in dates:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].fillna(pd.Timestamp(date_fict))
    return df[col]

# Compile the regular expression to delete irrelevants terms
irrelevant_terms = re.compile(r'\b(inc|corp|corporation|limited|ltd|plc|pictures|picture|co|entertainment|films|film|studios|corporat)\b')

def preprocess_company(company):
    """ 
    Preprocesses a company name by applying several cleaning steps.
    
    The function converts the company name to lowercase, removes irrelevant terms, 
    strips special characters, and normalizes spaces.
    """
    company = company.lower().strip()               # Convert the company name to lowercase and remove spaces
    company = irrelevant_terms.sub('', company)     # Remove irrelevant terms that do not affect the main name
    company = re.sub(r'[^\w\s]', '', company)       # Remove special characters from the company name
    company = re.sub(r'\s+', ' ', company).strip()  # Replace multiple spaces with a single space and strip spaces
    return company

def group_similar(company, all_companies, threshold=85):
    """
    Finding the best matches from a list of company names using fuzzy matching.
    """
    # Find matches with process.extract from RapidFuzz
    matches = process.extract(company, all_companies, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    
    if matches:
        try:
            # Extract the matched company names from the tuples
            match_names = [match[0] for match in matches]  
            most_common = Counter(match_names).most_common(1)
            return most_common[0][0] if most_common else company   # Return the most common match
        except IndexError:
            return "ERROR"
    return company  # Return the original company name if no matches are found

# Función para procesar compañías en paralelo
def process_company(companies, threshold=85):
    """
    Processes a list of company names in parallel to group similar companies, applying the group_similar function
    """
    return Parallel(n_jobs=-1)(
        delayed(group_similar)(company, companies, threshold) for company in companies
    )

############################################### END FUNCTIONS #################################################

# 1: Read dataset with pandas
original_data = pd.read_csv('data/rotten_tomatoes_movies.csv')
raw_data = original_data.copy()
amount_dirty_data = raw_data.shape

# 2: Definition of data types
categorical = ['content_rating','tomatometer_status','audience_status']
irrelevant = ['rotten_tomatoes_link', 'movie_info', 'critics_consensus','tomatometer_count', 
    'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count','audience_count']
relevant = ['movie_title']
objs = ['directors', 'genres', 'authors', 'actors', 'production_company']
numeric = ['runtime']
ratings = ['tomatometer_rating', 'audience_rating']
dates = ['original_release_date', 'streaming_release_date']

# 2.1: Definition of categories
rating = ['g', 'nc17', 'nr', 'pg', 'pg13', 'r']
t_status = ['rotten', 'fresh', 'certified-fresh', 'nr']
a_status = ['spilled', 'upright', 'nr']

# 3: Revisamos datos nulos en el df y sus tipos
amount_dirty_nulls = raw_data.isnull().sum().sum()
type_dirty_data = raw_data.dtypes
amount_companies = raw_data['production_company'].nunique()

# 4: remove irrelevant columns
raw_data = raw_data.drop(columns= irrelevant, errors='ignore')

# 4.1: remove rows with missing values in relevant columns
raw_data = raw_data.dropna(subset=relevant)

# 4.2 remove duplicates
raw_data = raw_data.drop_duplicates(subset=['movie_title','original_release_date','directors'])

# 5: fill missing category values
for col in categorical:
    raw_data[col] = raw_data[col].str.lower().fillna('nr').astype('category')
    
# 5.1: fill missing objects values
raw_data[objs] = raw_data[objs].fillna('nr')

# 5.2: fill missing numerica values
raw_data[numeric] = raw_data[numeric].fillna(0.0)

# 5.3: fill missing data vulues 
amount_date_nulls = raw_data[dates].isnull().sum().sum()
fill_date_null(raw_data, dates, '1900-01-01')
validate_change_date(raw_data,dates,'1900-01-01',amount_date_nulls)

# 6: Normalize object columns
lower = raw_data.select_dtypes(include=['object','category']).columns
for col in lower:
    raw_data[col] = raw_data[col].str.lower()
    
# 6.1: Normalize objects with frozenset
raw_data[objs] = (
    raw_data[objs]
    .replace('&', ',', regex=True)
    .apply(lambda col: col.str.split(',').apply(lambda x: frozenset(map(str.strip, x))))
)   

# 7: Fill nulls of tomatometer_rating with audience_rating and vice versa
raw_data[ratings] = raw_data[ratings].apply(lambda x: x.fillna(raw_data[ratings].mean(axis=1)))
# 7.2: Fill nulls in both columns with median
raw_data[ratings] = raw_data.groupby('genres')[ratings].transform(lambda x: x.fillna(x.median()))
# rellena los nulos restantes, si los hay, con la media general
raw_data[ratings] = raw_data[ratings].fillna(raw_data[ratings].mean())

# 8: Transfor frozenset to string
transform_to_string(raw_data,objs)

# 9: Normalize categories
category_dict = {
    'content_rating': rating,
    'tomatometer_status': t_status,
    'audience_status': a_status
    }
for col, choices in category_dict.items():
    raw_data[col] = raw_data[col].apply(lambda x: normalize_category_fuzzy(x, choices)).astype('category')
    
# 9.1: Normalize companies with rapidfuzz
start_time = time.time()

raw_data['production_company'] = raw_data['production_company'].apply(preprocess_company)
companies_list = raw_data['production_company'].tolist()
raw_data['production_company'] = process_company(companies_list, threshold=85)

# 10: define clean_data dataframe
clean_data = raw_data

# 11: load data
clean_data.to_csv("data/rotten_tomatoes_movies_clean.csv", index = False)

show(clean_data.head(200))

amount_clean_data = clean_data.shape
amount_clean_nulls = clean_data.isnull().sum().sum()
type_clean_data = clean_data.dtypes

print(Fore.RED + "DIRTY DATA" + Style.RESET_ALL)
print(f"Cantidad de filas inicial: {amount_dirty_data[0]}")
print(f"Cantidad de columnas inicial: {amount_dirty_data[1]}")
print(f"Cantidad Datos nulos: {amount_dirty_nulls}")
print(f"Cantidad de compañias: {amount_companies} ")
print(f"Tipo de datos por columna: \n{type_dirty_data} \n")

print(Fore.GREEN + "CLEAN DATA" + Style.RESET_ALL)
print(f"Cantidad de filas: {amount_clean_data[0]}")
print(f"Cantidad de columnas: {amount_clean_data[1]}")
print(f"cantidad Datos nulos: {amount_clean_nulls} ")
print(f"Cantidad de compañias: {clean_data['production_company'].nunique()}" )
print(f"Tipo de datos por columna: \n{type_clean_data}\n")

# # Imprimir tiempo total
print(f"Tiempo de procesamiento: {time.time() - start_time} segundos")

############################################################################### DESCOMPOSE ################################################### 
# Decompose the genre string into a list
df_normalized = clean_data[clean_data['directors'] != 'nr'].copy()

df_normalized['genres_normalized'] = df_normalized['genres'].str.split(',') 

# Normalized genres
df_normalized['genres_normalized'] = df_normalized['genres_normalized'].apply(lambda x: [unidecode(genre.strip()) for genre in x])

# Descompose the directors string into a list

df_normalized['directors_normalized'] = df_normalized['directors'].str.split(',')   

# Normalized directors
df_normalized['directors_normalized'] = df_normalized['directors_normalized'].apply(lambda x: [unidecode(director.strip()) for director in x])

# Explode the list of genres
genres = df_normalized['genres_normalized'].explode().str.strip()  

# Explode the list of directors
directors = df_normalized['directors_normalized'].explode().str.strip().str.title()

# Count the frecuency of each genre
genre_counts = genres.value_counts()

# Count the frecuency of each director
top_directors = directors.value_counts().nlargest(15)



