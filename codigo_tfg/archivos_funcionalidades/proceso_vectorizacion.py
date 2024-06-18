# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Vectorización de palabras
#   - Bag-of-Words (BOW)
#   - Term Frequency - Inverse Document Frequency (TF-IDF)
#   - Word2Vec
# Preprocesamiento del texto 
#   - Limpieza de texto + Tokenización + Palabras en minúscula + Eliminación de palabras vacías



### NECESIDAD AJUSTAR PARÁMETROS DE LOS MÉTODOS



import os                                   # Importa módulo 'os' - Acceder a funcionalidades del Sistema Operativo
import json                                 # Importa biblioteca para formato JSON
import nltk                                 # Importa biblioteca NLTK (Natural Language Toolkit) - Herramientas PLN
import gensim                               # Importa biblioteca gensim

from nltk.tokenize import word_tokenize                         # Dividir texto en palabras o tokens
from nltk.corpus import stopwords                               # Conjunto palabras comunes generalmente irrelevantes
from sklearn.feature_extraction.text import CountVectorizer     # Permite la vectorización de Bag-of-Words   
from sklearn.feature_extraction.text import TfidfVectorizer     # Permite la vectorización de TF-IDF         
from gensim.models import Word2Vec                              # Permite la vectorización de Word2Vec



# Descargar recursos para almacenamiento en sistema
nltk.download('stopwords')
nltk.download('punkt')

# Obtener las stopwords en español
stopwords_es = list(set(stopwords.words('spanish')))



# Definir parámetros necesarios                
storage_dir = r"C:\Ficheros sobre Ansiedad\Archivos extraidos Telegram\Vectorizacion"     # Obtener directorio de almacenamiento
nombre_fichero = "subject1.json"                # Nombre del archivo



#### OBTENER LISTADO DE FICHEROS EN DIRECTORIO #### 

# Lista para almacenar los nombres de los archivos
lista_ficheros = []

# Obtener los nombres de los archivos en la carpeta
for fichero in os.listdir(storage_dir):   # listdir('dir') --> Listar todos los archivos y carpetas en un directorio
    lista_ficheros.append(fichero)

# Imprimir los nombres de los archivos
print("Nombres de archivos en la carpeta:")
for nombre in lista_ficheros:
    print(nombre)




#### PREPROCESAMIENTO DEL TEXTO ####


# Función para preprocesar los documentos
def preprocesamiento(document):
    # Tokenización y eliminación de stopwords
    tokens = word_tokenize(document)
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stopwords_es]
    return " ".join(tokens)




#### BOLSA DE PALABRAS o Bag-of-Words #### 


# Función de Vectorización Bag-of-Words  
def bag_of_words(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    documents = []  # Conjunto de mensajes a vectorizar

    # Obtener conjunto de mensajes
    documents = [preprocesamiento(mensaje['message']) for mensaje in datos]

    # Crear el vectorizador CountVectorizer con parámetros ajustados
    vectorizer = CountVectorizer()
    # vectorizer = CountVectorizer(
    #     stop_words=stopwords_es,  # Especificar las stopwords en español - Eliminación
    #     min_df=2,                 # Considerar términos que aparecen en al menos 2 documentos
    #     max_df=0.5,               # Descartar términos que aparecen en más del 50% de los documentos
    #     max_features=1000,        # Considerar solo los 1000 términos más frecuentes
    #     ngram_range=(1, 2)        # Considerar unigramas y bigramas
    # )

    # Ajustar el vectorizador al corpus y transformar los documentos en vectores BoW
    X = vectorizer.fit_transform(documents)

    # Obtener el vocabulario
    vocabulario = vectorizer.get_feature_names_out()

    # Obtener los vectores BoW para cada mensaje
    vectors = X.toarray()

    # Imprimir la matriz de términos-documentos
    # print("Matriz término-documento (BoW):\n", X.toarray())

    # Imprimir el vocabulario
    # print(f"El tamaño es: {len(vocabulario)}")
    # print("\nVocabulario:")
    # print(vocabulario)

    # Reemplazar el campo 'message' por el vector BoW correspondiente
    for i, mensaje in enumerate(datos):
        mensaje['message'] = json.dumps([int(x) for x in vectors[i]])

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)

        


#### TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY (TF-IDF) #### 


# Función de Vectorización TF-IDF  
def TF_IDF(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    documents = []      # Conjunto de mensajes a vectorizar

    # Obtener conjunto de mensajes
    documents = [preprocesamiento(mensaje['message']) for mensaje in datos]
    
    # Crear el vectorizador TF-IDF con parámetros ajustados
    vectorizer = TfidfVectorizer()
    # vectorizer = TfidfVectorizer(
    #     stop_words=stopwords_es,  # Especificar las stopwords en español
    #     min_df=2,                 # Considerar términos que aparecen en al menos 2 documentos
    #     max_df=0.5,               # Descartar términos que aparecen en más del 50% de los documentos
    #     max_features=1000,        # Considerar solo los 1000 términos más frecuentes
    #     ngram_range=(1, 2),       # Considerar unigramas y bigramas
    #     sublinear_tf=True         # Aplicar transformación logarítmica a la frecuencia de términos
    # )

    # Ajustar el vectorizador al corpus y transformar los documentos en vectores TF-IDF
    X = vectorizer.fit_transform(documents)

    # Obtener el vocabulario
    vocabulario = vectorizer.get_feature_names_out()

    # Obtener los vectores TF-IDF para cada mensaje
    vectors = X.toarray()

    # Imprimir la matriz de términos-documentos
    # print("Matriz término-documento (TF-IDF):\n", X.toarray())

    # Imprimir el vocabulario
    # print(f"El tamaño es: {len(vocabulario)}")
    # print("\nVocabulario:")
    # print(vocabulario)

    # Reemplazar el campo 'message' por el vector TF-IDF correspondiente
    for i, mensaje in enumerate(datos):
        mensaje['message'] = json.dumps([float(x) for x in vectors[i]])

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### Word Embeddings - Word2Vec #### 

# Función de Vectorización Word2Vec
def word2Vec(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    documents = []      # Conjunto de mensajes a vectorizar

    # Obtener conjunto de mensajes
    documents = [preprocesamiento(mensaje['message']) for mensaje in datos]

    # Entrenamiento del modelo Word2Vec
    #   - Lista de mensajes preprocesados para entrenar el modelo Word2Vec
    #   - vector_size --> Dimensionalidad de vectores de palabras
    #   - window --> Tamaño de la ventana de contexto - Cantidad máxima de palabras vecinas consideradas para predecir la palabra objetivo
    #   - min_count --> Umbral de frecuencia mínimo de las palabras 
    #   - workers --> Cantidad de hilos de procesamiento para entrenar el modelo
    model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

    # Sobreescribir el campo 'message' con vectores correspondientes
    for mensaje, preprocessed_message in zip(datos, documents):
        mensaje['message'] = json.dumps([model.wv[word].tolist() for word in preprocessed_message if word in model.wv])

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### EXTRACCIÓN DE CARACTERÍSTICAS - VECTORIZACIÓN ####    


## OPCIÓN 1 - FICHERO ÚNICO ##
# bag_of_words(f"{storage_dir}/{nombre_fichero}")       # Llama a la función para Vectorización Bag-of-Words  
TF_IDF(f"{storage_dir}/{nombre_fichero}")               # Llama a la función para Vectorización TF-IDF  
# word2Vec(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para Vectorización Word2Vec


## OPCIÓN 2 - CONJUNTO DE FICHEROS ##
# for nombre_fichero in lista_ficheros:   # Cada fichero extraído
#     bag_of_words(f"{storage_dir}/{nombre_fichero}")       # Llama a la función para Vectorización Bag-of-Words  
#     TF_IDF(f"{storage_dir}/{nombre_fichero}")             # Llama a la función para Vectorización TF-IDF  
#     word2Vec(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para Vectorización Word2Vec  