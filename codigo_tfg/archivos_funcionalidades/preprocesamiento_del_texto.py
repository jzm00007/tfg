# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Preprocesamiento de datos 
#   - Tokenización
#   - Conversión palabras a minúsculas 
#   - Eliminacion Palabras Vacías
#   - Estructura gramatical
#   - Stemming
#   - Lematización



import os                                   # Importa módulo 'os' - Acceder a funcionalidades del Sistema Operativo
import json                                 # Importa biblioteca para formato JSON
import nltk                                 # Importa biblioteca NLTK (Natural Language Toolkit) - Herramientas PLN
import spacy.cli                            # Importa módulo 'cli' de SpaCy - Interfaz de linea de comandos para tareas          
from nltk.tokenize import word_tokenize     # Importa función word_tokenize desde módulo nltk.tokenize -   Dividir texto en palabras o tokens
from nltk.corpus import stopwords           # Importa conjunto de datos de stopwords desde módulo nltk.corpus - Conjunto palabras comunes generalmente irrelevantes
from nltk.stem import SnowballStemmer       # Importa la clase 'SnowballStemmer' del módulo 'stem' de la biblioteca NLTK - Extraer raíz



# Descargar recursos para almacenamiento en sistema
nltk.download('punkt')
nltk.download('stopwords')

# Modelo en español - Cantidad datos y capacidad vectorización más grande
# spacy.cli.download("es_core_news_lg")   
import es_core_news_lg    
nlp = es_core_news_lg.load()



# Definir parámetros necesarios                
storage_dir = r"C:\Ficheros sobre Ansiedad\Archivos extraidos Telegram\Preprocesamiento del texto"     # Obtener directorio de almacenamiento
nombre_fichero = "subject1.json"               # Nombre del archivo




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




#### TOKENIZACIÓN ####


# Función de Tokenización
def tokenizacion(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Iterar sobre cada mensaje y tokenizar texto del campo 'message'
    for mensaje in datos:       
        mensaje['message'] = word_tokenize(mensaje['message'])      # Tokenización de palabras de NLTK

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)


# Función para generar N-grammas - n: número de elementos/palabras agrupadas juntas en secuencia contigua
def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words)-n+1):
        ngrams.append(' '.join(words[i:i+n]))
    return ngrams




#### CONVERSIÓN PALABRAS A MINÚSCULAS ###


# Función de Conversión palabras a Minúscula
def palabra_min(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Iterar sobre cada mensaje y convertir a minúscula texto del campo 'message'

    # Con Tokenización
    for mensaje in datos:
       palabras_min = []   # Listado de palabras en minúsucla
       
       for palabra in mensaje['message']:
           palabras_min.append(palabra.lower())
    
       mensaje['message'] = palabras_min

    # Sin Tokenización
    # for mensaje in datos:
    #     mensaje['message'] = mensaje['message'].lower()

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### ELIMINACIÓN DE PALABRAS VACÍAS ####        


# Función de Eliminar Palabras Vacías del texto
def palabra_vacias(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Obtener las stopwords en español
    stopwords_es = set(stopwords.words('spanish'))

    # Iterar sobre cada mensaje y eliminar palabras vacías

    # Con Tokenización
    for mensaje in datos:
        tokens = mensaje['message']
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stopwords_es]
        mensaje['message'] = tokens_sin_stopwords
       
    # Sin Tokenización
    # for mensaje in datos:
    #     mensaje['message'] = ' '.join([palabra for palabra in mensaje['message'].split() if palabra.lower() not in stopwords_es])

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### Etiquetado gramatical o POS-tagging ####
# - Ejecución independiente al resto


# Función para obtener Estructura Gramatical cada palabra
def estructura_gramatical(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Iterar sobre cada mensaje y obtener estructura gramatical cada palabra
    for mensaje in datos:
        # Unir las palabras en una oración
        frase = " ".join(mensaje['message'])

        # Procesamiento de la oración con spaCy
        doc = nlp(frase)

        texto_gramatical = []  # Palabras con análisis gramatical

        # Palabras con sus etiquetas gramaticales
        for token in doc:
            texto_gramatical.append([token.text, token.pos_])

        mensaje['message'] = texto_gramatical
    
    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### STEMMING ####
# - Obtener raiz o forma base


# Función para stemming
def stemming(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Crear el stemmer para español
    stemmer = SnowballStemmer('spanish')

    # Iterar sobre cada mensaje y obtener la raiz
    for mensaje in datos:
        # Aplicar stemming a cada palabra
        stemmed_words = [stemmer.stem(word) for word in mensaje['message']]

        mensaje['message'] = stemmed_words
    
    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### LEMATIZACIÓN ####
# - Obtener lema


# Función para lematizacion
def lematizacion(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Crear el stemmer para español
    stemmer = SnowballStemmer('spanish')

    # Iterar sobre cada mensaje y obtener la raiz
    for mensaje in datos:
        # Unir las palabras en una oración
        frase = " ".join(mensaje['message'])

        # Procesamiento de la oración con spaCy
        doc = nlp(frase)

        # Obtener lemas de cada palabra
        lemmas = [token.lemma_ for token in doc]

        mensaje['message'] = lemmas
    
    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### PREPROCESAMIENTO DEL TEXTO ####    


## OPCIÓN 1 - FICHERO ÚNICO ##
tokenizacion(f"{storage_dir}/{nombre_fichero}")                # Llama a la función para tokenización
palabra_min(f"{storage_dir}/{nombre_fichero}")                 # Llama a la función para convertir palabras a minúsculas
palabra_vacias(f"{storage_dir}/{nombre_fichero}")              # Llama a la función para eliminar palabras vacías texto
# estructura_gramatical(f"{storage_dir}/{nombre_fichero}")      # Llama a la función para obtener la estructura gramatical de cada palabra
# stemming(f"{storage_dir}/{nombre_fichero}")                   # Llama a la función para stemming
lematizacion(f"{storage_dir}/{nombre_fichero}")                # Llama a la función para lematización


## OPCIÓN 2 - CONJUNTO DE FICHEROS ##
# for nombre_fichero in lista_ficheros:   # Cada fichero extraído
#     tokenizacion(f"{storage_dir}/{nombre_fichero}")            # Llama a la función para tokenización
#     palabra_min(f"{storage_dir}/{nombre_fichero}")             # Llama a la función para convertir palabras a minúsculas
#     palabra_vacias(f"{storage_dir}/{nombre_fichero}")          # Llama a la función para eliminar palabras vacias texto
#     estructura_gramatical(f"{storage_dir}/{nombre_fichero}")   # Llama a la función para obtener la estructura gramatical de cada palabra
#     stemming(f"{storage_dir}/{nombre_fichero}")                # Llama a la función para stemming
#     lematizacion(f"{storage_dir}/{nombre_fichero}")            # Llama a la función para lematización