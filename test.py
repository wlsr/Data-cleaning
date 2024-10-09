import pandas as pd
from pandasgui import show
from thefuzz import process, fuzz
from joblib import Parallel, delayed
from collections import Counter
import time
import re

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
    for obj in value:
        df[obj] = df[obj].apply(lambda x: ', '.join(sorted(x)))
    
def normalize_category_fuzzy(value, choices, threshold=80):
    """ 
    Normalize the categories for each categorical data
    """
    if isinstance(value, str):
        best_match, best_score = process.extractOne(value, choices)
        if best_score >= threshold:
            return best_match
    return 'nr'

def fill_date_null(df, dates, date_fict):
    """
    fill the null dates with fictitious dates: 1900-01-01
    """
    for col in dates:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].fillna(pd.Timestamp(date_fict))
    return df[col]

# Compilar la expresión regular fuera del bucle para evitar compilarla en cada llamada
irrelevant_terms = re.compile(r'\b(inc|corp|corporation|limited|ltd|plc|pictures|picture|co|entertainment|films|film|studios|corporat)\b')

# Preprocesamiento de compañías para eliminar términos irrelevantes
def preprocess_company(company):
    company = company.lower().strip()               # Convertir a minúsculas y eliminar espacios
    company = irrelevant_terms.sub('', company)     # Remover términos comunes que no afectan el nombre principal
    company = re.sub(r'[^\w\s]', '', company)       # Eliminar caracteres especiales
    company = re.sub(r'\s+', ' ', company).strip()  # Reemplazar múltiples espacios por uno solo
    return company

# Función para agrupar compañías similares
def group_similar(company, all_companies, threshold=85):
    """
    Usa extractBests para obtener las mejores coincidencias y agrupar compañías
    """
    # Obtener coincidencias
    matches = process.extractBests(company, all_companies, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    
    if matches:
        # Validar que las coincidencias tienen suficiente información antes de intentar acceder a ellas
        try:
            # Solo procesar si hay más de un elemento en el tuple
            match_names = [match[0] for match in matches]  # Obtenemos los nombres directamente
            most_common = Counter(match_names).most_common(1)
            return most_common[0][0] if most_common else company  # Devuelve la coincidencia más común
        except IndexError:
            return "ERROR"
    return company  # Devuelve la compañía original si no hay coincidencias


# Función para procesar compañías en paralelo
def process_company(companies,threshold=85):
    """
    Usar Parallel para ejecutar la función en paralelo
    """
    return Parallel(n_jobs=-1)(
        delayed(group_similar)(company, companies, threshold) for company in companies
    )

# Función para dividir en chunks dinámicos
def chunkify_dynamic(lst, chunk_size):
    """Divide la lista en chunks de tamaño 'chunk_size'."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# Función para procesar en chunks
def process_company_in_chunks_dynamic(companies, threshold=85):
    """
    Procesar la lista de compañías en chunks más pequeños para optimizar el rendimiento
    """
    # Definir tamaño del chunk basado en el tamaño de la lista
    chunk_size = max(len(companies) // 4000, 1)
    chunks = chunkify_dynamic(companies, chunk_size)
    
    # Procesamos cada chunk en paralelo
    results = Parallel(n_jobs=-1)(
        delayed(process_company)(chunk, companies, threshold) for chunk in chunks
    )
    
    # Combinamos los resultados
    return [company for chunk_result in results for company in chunk_result]


############################################### END FUNCTIONS #################################################

# 1: Cargamos el dataset con pandas
raw_data = pd.read_csv('data/rotten_tomatoes_movies.csv')
amount_dirty_data = raw_data.shape

# 2: Armamos listas segun el tipo de dato
categorical = ['content_rating','tomatometer_status','audience_status']
irrelevant = ['rotten_tomatoes_link', 'movie_info', 'critics_consensus','tomatometer_count', 
    'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count','audience_count']
relevant = ['movie_title']
objs = ['directors', 'genres', 'authors', 'actors', 'production_company']
numeric = ['runtime']
ratings = ['tomatometer_rating', 'audience_rating']
dates = ['original_release_date', 'streaming_release_date']

# 2.1: definición de categorias
rating = ['g', 'nc17', 'nr', 'pg', 'pg13', 'r', 'unknown']
t_status = ['rotten', 'fresh', 'certified-fresh', 'unknown']
a_status = ['spilled', 'upright', 'unknown']

# 3: Eliminamos columnas irrelevantes para este proceso
raw_data = raw_data.drop(columns= irrelevant, errors='ignore')

# 4: Revisamos datos nulos en el df y sus tipos
amount_dirty_nulls = raw_data.isnull().sum()
type_dirty_data = raw_data.dtypes
amount_companies = raw_data['production_company'].nunique()

# 5: Eliminamos las filas que tengan datos nulos en las columnas relevantes
raw_data = raw_data.dropna(subset=relevant)

# Eliminamos las filas duplicadas
raw_data = raw_data.drop_duplicates(subset=['movie_title','original_release_date','directors'])

# 6: lower
lower = raw_data.select_dtypes(include=['object','category']).columns
for col in lower:
    raw_data[col] = raw_data[col].str.lower()
    
# 7: llenamos los datos categoricos nulos
raw_data[categorical] = raw_data[categorical].fillna('nr').astype('category')

# 8: llenamos los datos objetc nulos
raw_data[objs] = raw_data[objs].fillna('nr')

# 10: llenamos datos numericos nulos
raw_data[numeric] = raw_data[numeric].fillna(0.0)

# 9: normalizamos los datos de tipo objeto, se remplaza & con , y elimina espacio vacios

#raw_data[objs] = raw_data[objs].replace('&', ',', regex=True).apply(lambda col: col.str.split(',')).apply(lambda col: col.apply(lambda x: frozenset(map(str.strip, x))))
raw_data[objs] = (
    raw_data[objs]
    .replace('&', ',', regex=True)
    .apply(lambda col: col.str.split(',').apply(lambda x: frozenset(map(str.strip, x))))
)

# 11: llenamos los datos rating nulos (Rellena nulos de tomatometer_rating con audience_rating y viceversa)
raw_data[ratings] = raw_data[ratings].apply(lambda x: x.fillna(raw_data[ratings].mean(axis=1)))
# rellena nulos en ambas columnas
raw_data[ratings] = raw_data.groupby('genres')[ratings].transform(lambda x: x.fillna(x.median()))
# rellena los nulos restantes, si los hay, con la media general
raw_data[ratings] = raw_data[ratings].fillna(raw_data[ratings].mean())

# 12: llenamos date nulos con 1900-01-01
fill_date_null(raw_data, dates, '1900-01-01')

# 13: Transformamos los frozenset a cadena
transform_to_string(raw_data,objs)

# 14: Normalizar categorias
category_dict = {
    'content_rating': rating,
    'tomatometer_status': t_status,
    'audience_status': a_status
    }
for col, choices in category_dict.items():
    raw_data[col] = raw_data[col].apply(lambda x: normalize_category_fuzzy(x, choices))
    
# 15: normalizar productoras
start_time = time.time()

raw_data['production_company'] = raw_data['production_company'].apply(preprocess_company)
companies_list = raw_data['production_company'].tolist()
raw_data['production_company'] = process_company(companies_list)

#show(raw_data.head(200))
amount_clean_data = raw_data.shape
amount_clean_nulls = raw_data.isnull().sum()
type_clean_data = raw_data.dtypes

# print("DIRTY DATA ")
# print(f"Cantidad de filas inicial: {amount_dirty_data[0]}")
# print(f"Cantidad de columnas inicial: {amount_dirty_data[1]}\n")
# print(f"Datos nulos por columna: \n{amount_dirty_nulls} \n")
# print(f"Tipo de datos por columna: \n{type_dirty_data} \n")
print(f"Cantidad de compañias: {amount_companies}")
# print("CLEAN DATA")
# print(f"Cantidad de filas: {amount_clean_data[0]}")
# print(f"Cantidad de columnas: {amount_clean_data[1]}\n")
# print(f"Datos nulos por columna: \n{amount_clean_nulls} \n")
# print(f"Tipo de datos por columna: \n{type_clean_data}\n")
print(f"Cantidad de compañias: {raw_data['production_company'].nunique()}" )
# # Imprimir tiempo total
print(f"Tiempo de procesamiento: {time.time() - start_time} segundos")



