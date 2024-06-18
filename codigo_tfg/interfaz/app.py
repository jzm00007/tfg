from flask import Flask, render_template, request, redirect, url_for, flash

import os                                   # Acceder a funcionalidades del Sistema Operativo
import json                                 # Formato JSON
import nltk                                 # Herramientas PLN
import spacy                                # Herramientas PLN
import numpy as np                          # Trabajar con matrices y operaciones matemáticas de alto nivel


from sklearn import svm                                 # Máquinas de vectores de soporte - SVM 
from sklearn.naive_bayes import MultinomialNB           # Naive Bayes
from sklearn import tree                                # Árboles de decisión
from sklearn.ensemble import RandomForestClassifier     # Random Forest - Clasificación
from sklearn.ensemble import RandomForestRegressor      # Random Forest - Regresión
from sklearn.neighbors import KNeighborsClassifier      # K vecinos - Clasificación
from sklearn.neighbors import KNeighborsRegressor       # K vecinos - Regresión
from sklearn.linear_model import LogisticRegression     # Logistic Regression


from nltk.corpus import stopwords                               # Conjunto palabras comunes generalmente irrelevantes 
from sklearn.feature_extraction.text import CountVectorizer     # Permite la vectorización de Bag-of-Words
from sklearn.feature_extraction.text import TfidfVectorizer     # Permite la vectorización de TF-IDF         
from gensim.models import Word2Vec                              # Permite la vectorización de Word2Vec


import re                           # Manejo expresiones regulares
import emoji                        # Importa biblioteca 'emoji' - Convertir emojis gráficos en texto descriptivo
from googletrans import Translator  # Importa biblioteca 'Translator' - Traducción de texto -  Google Translate
import string                       # Importa módulo 'string' - Manipular cadena de caracteres

from collections import Counter     # Estructura de datos para contar objetos - Frecuencia



# Descargar recursos para almacenamiento en sistema
nltk.download('stopwords')

# Obtener las stopwords en español
stopwords_es = list(set(stopwords.words('spanish')))


# Cargar modelo de idioma español de SpaCy
# Modelo en español - Cantidad datos y capacidad vectorización más grande
import es_core_news_lg    
nlp = es_core_news_lg.load()


# Obtener la ruta del directorio raíz de la aplicación
root_dir = os.path.dirname(os.path.abspath(__file__))

# Añadir directorio static
static_dir = os.path.join(root_dir, 'static')

# Añadir directorio corpus 
corpus_dir = os.path.join(static_dir, 'corpus')

# Añadir directorio lexicon 
lexicon_dir = os.path.join(static_dir, 'lexicon')

# Añadir directorio lexicon propio
lexicon_propio_dir = os.path.join(lexicon_dir, 'lexicon_propio')


# Definir parámetros necesarios    
entrenamiento_dir = corpus_dir                  # Ruta al directorio corpus
lexicon_dir = lexicon_dir                       # Ruta al directorio lexicon
lexicon_propio_dir = lexicon_propio_dir         # Ruta al directorio lexicon propio
nombre_fichero_etiquetas_A = "gold_a.txt"       # Nombre del archivo etiquetado A
nombre_fichero_etiquetas_B = "gold_b.txt"       # Nombre del archivo etiquetado B
fichero_listado_palabras_ansiedad = "listado_palabras_ansiedad_jack_moreno.txt"            # Obtener listado de palabras ansiedad (Jack Moreno)
fichero_listado_palabras_negativas = "listado_palabras_negativas_isol.txt"                 # Obtener listado de palabras negativas (ISOL)

ficheros_test = []                              # Ficheros a evaluar
mensajes_test = []                              # Contenido ficheros a evaluar
proceso_vectorizacion = ""                      # Proceso de vectorización seleccionado

# Inicializar contadores para cada tipo de palabra
verb_counter_lexicon = Counter()
noun_counter_lexicon = Counter()
adj_counter_lexicon = Counter()
adv_counter_lexicon = Counter()




#### OBTENER LISTADO DE FICHEROS ENTRENAMIENTO EN DIRECTORIO #### 

# Lista para almacenar los nombres de los archivos de entrenamiento
lista_ficheros_entrenamiento = []

# Obtener los nombres de los archivos en la carpeta entrenamiento
for fichero in os.listdir(entrenamiento_dir):   
    if fichero.endswith(".json"):  # Solo añadir archivos con extensión .json
        lista_ficheros_entrenamiento.append(fichero)




#### OBTENER ETIQUETADOS DE ENTRENAMIENTO #### 

# Lista para almacenar los nombres de los archivos de etiquetado
lista_ficheros_etiquetado = []

# Lista para almacenar los sujetos
listado_sujetos = []

# Lista para almacenar las etiquetas de tipo A
listado_etiquetas_A = []

# Lista para almacenar las etiquetas de tipo B
listado_etiquetas_B = []


# Obtener los nombres de los archivos en la carpeta
for fichero in os.listdir(entrenamiento_dir):
    if fichero.endswith(".txt"):  # Solo añadir archivos con extensión .txt
        lista_ficheros_etiquetado.append(fichero)


# Filtrar los archivos de entrenamiento sin extensión
archivos_entrenamiento  = []
for fichero_entrenamiento in lista_ficheros_entrenamiento:
    nombre_archivo_sin_extension = os.path.splitext(fichero_entrenamiento)[0]  # Obtener nombre sin extensión
    archivos_entrenamiento.append(nombre_archivo_sin_extension)


# Leer cada archivo de etiquetado
for nombre_fich in lista_ficheros_etiquetado:
    with open(os.path.join(entrenamiento_dir, nombre_fich), 'r', encoding='utf-8') as archivo:
        # Ignorar la primera línea del archivo
        next(archivo)

        # Leer cada línea del archivo
        for linea in archivo:
            # Dividir la línea en subject y label
            subject, label = linea.strip().split(',')

            if subject in archivos_entrenamiento:   # Comprobar archivo entrenamiento - sujeto
                # Almacenar en las listas correspondientes según el nombre del archivo
                if subject not in listado_sujetos:
                    listado_sujetos.append(subject)

                if nombre_fich == nombre_fichero_etiquetas_A:
                    listado_etiquetas_A.append(label)
                elif nombre_fich == nombre_fichero_etiquetas_B:
                    listado_etiquetas_B.append(label)




#### OBTENER CONTENIDO MENSAJES DE ENTRENAMIENTO ####

# Lista para almacenar los documentos
documents = []

# Iterar sobre cada nombre de archivo en la lista de sujetos 
for nombre_sujeto in listado_sujetos:
    nombre_archivo = nombre_sujeto + ".json"

    if nombre_archivo.endswith(".json"):
        # Construir la ruta completa del archivo JSON
        ruta_json = os.path.join(entrenamiento_dir, nombre_archivo)

        # Almacenar los mensajes del archivo JSON actual
        mensajes_fichero = ""
        
        # Verificar si el archivo JSON existe
        if os.path.exists(ruta_json):
            # Abrir el archivo JSON
            with open(ruta_json, 'r', encoding='utf-8') as archivo_json:
                # Cargar el contenido JSON
                contenido_json = json.load(archivo_json)
                
                # Iterar sobre cada objeto en el archivo JSON
                for objeto in contenido_json:
                    # Verificar si el objeto tiene el campo 'message'
                    if 'message' in objeto:
                        # Agregar el mensaje al conjunto de mensajes
                        mensajes_fichero += objeto['message'] + "\n"
        
        # Agregar mensajes al documento actual
        documents.append(mensajes_fichero)




#### OBTENER CONTENIDO MENSAJES DE TEST ####

# Función para generar listado almacenamiento de todos los mensajes de los archivos JSON de test
def obtener_mensajes_test(): 
    global ficheros_test    # Ficheros test a evaluar
    global mensajes_test    # Mensajes test


    # Iterar sobre cada nombre de archivo en la lista de archivos de test
    for fichero in ficheros_test:
        if fichero:
            filename = fichero.filename     # Fichero
            try:
                # Leer el contenido del archivo
                fichero.seek(0)  # Puntero al principio del archivo
                content = fichero.read().decode('utf-8')
                contenido_json = json.loads(content)    # JSON

                # Almacenar los mensajes del archivo JSON actual
                mensajes_fichero = ""
                
                # Iterar sobre cada objeto en el archivo JSON
                for objeto in contenido_json:
                    # Verificar si el objeto tiene el campo 'message'
                    if 'message' in objeto:
                        # Agregar el mensaje del archivo JSON actual
                        mensajes_fichero += objeto['message'] + "\n"
                
                # Agregar la lista de mensajes del archivo JSON actual a la lista general
                mensajes_test.append(mensajes_fichero)

            except json.JSONDecodeError:
                print(f"Nombre: {filename}")
                print("Error: JSON no valido")
            except Exception as e:
                print(f"Nombre: {filename}")
                print(f"Error leyendo archivo: {e}")
                



#### EXTRACCIÓN DE CARACTERÍSTICAS - VECTORIZACIÓN ####


# Crear un objeto Translator
translator = Translator()

# Función para limpieza del texto
def limpieza_texto(texto):
    # Convertir emojis gráficos a texto descriptivo en inglés
    texto = emoji.demojize(texto)

    # Encontrar y traducir palabras que representan emojis
    palabras = re.findall(r':[a-zA-Z_-]+:', texto)
    for palabra in palabras:
        palabra_modificada = palabra.strip(':').replace("_", " ").replace("-", " ")
        palabra_esp = translator.translate(palabra_modificada, src='en', dest='es').text
        texto = texto.replace(palabra, f" {palabra_esp} ")

    # Eliminar URLs
    texto = re.sub(r'(https?://|www\.)\S+?(?=\s|")', '', texto)

    # Eliminar saltos de línea
    texto = texto.replace('\\n', '')

    # Eliminar correos electrónicos
    texto = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', texto)

    # Eliminar menciones
    texto = re.sub(r'@ ?\S+?(?=\s|")', '', texto)

    return texto


# Función para preprocesar los documentos
#   - Tokenización
#   - Eliminación de palabras vacías
#   - Conversión palabras a minúscula
#   - Lematización
def preprocesamiento_texto(document):
    texto = limpieza_texto(document)    # Limpieza del texto

    # Procesamiento del texto con SpaCy
    doc = nlp(texto)
    
    # Lematización, tokenización, conversión a minúsculas y eliminación de palabras vacías
    tokens = [token.lemma_ for token in doc if token.lemma_.lower() not in stopwords_es]
    
    return " ".join(tokens)


# Obtener conjunto de mensajes
documents_train = [preprocesamiento_texto(mensaje) for mensaje in documents]        # Entrenamiento



# Crear el vectorizador CountVectorizer con parámetros ajustados - Ajustar parámetros - Bag of Words    
# vectorizer_BOW = CountVectorizer()
vectorizer_BOW = CountVectorizer(
    stop_words=None,          # Especificar las stopwords en español
    min_df=3,                 # Considerar términos que aparecen en al menos 3 documentos
    max_df=1.0,               # Descartar términos que aparecen en más del 100% de los documentos
    max_features=2000,        # Considerar solo los 2000 términos más frecuentes
    ngram_range=(1, 2)        # Considerar unigramas y bigramas
)

# Ajustar el vectorizador al corpus y transformar los documentos en vectores BOW
X_BOW = vectorizer_BOW.fit_transform(documents_train)



# Crear el vectorizador TF-IDF con parámetros ajustados - Ajustar parámetros
# vectorizer_TF_IDF = TfidfVectorizer()
vectorizer_TF_IDF = TfidfVectorizer(
    stop_words=None,          # Especificar las stopwords en español
    min_df=2,                 # Considerar términos que aparecen en al menos 2 documentos
    max_df=1.0,               # Descartar términos que aparecen en más del 100% de los documentos
    max_features=2000,        # Considerar solo los 2000 términos más frecuentes
    ngram_range=(1, 1),       # Considerar unigramas
    sublinear_tf=False        # No aplicar transformación logarítmica a la frecuencia de términos
)

# Ajustar el vectorizador al corpus y transformar los documentos en vectores TF-IDF
X_TF_IDF = vectorizer_TF_IDF.fit_transform(documents_train)



# Entrenamiento del modelo Word2Vec
model_SG = Word2Vec(documents_train, sg=1, vector_size=175, window=5, min_count=2, workers=4)
model_CBOW = Word2Vec(documents_train, sg=0, vector_size=175, window=5, min_count=2, workers=4)

# Obtener representaciones vectoriales de los datos de texto
def representaciones_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Obtener conjunto de mensajes entrenamiento - SG
X_W2V_SG = [representaciones_vector(mensajes_usuarios, model_SG) for mensajes_usuarios in documents_train]

# Obtener conjunto de mensajes entrenamiento - CBOW
X_W2V_CBOW = [representaciones_vector(mensajes_usuarios, model_CBOW) for mensajes_usuarios in documents_train]




#### LEXICON ####


#### OBTENER LISTADO DE FICHEROS EN DIRECTORIO DE ENTRENAMIENTO #### 
# - Puntuación entre 0.9 - 1

# Lista para almacenar los nombres de los archivos
lista_ficheros_lexicon_propio = []

# Obtener los nombres de los archivos en la carpeta
for nombre_fichero in os.listdir(lexicon_propio_dir):  
    lista_ficheros_lexicon_propio.append(nombre_fichero)




#### OBTENER LISTADO DE PALABRAS SOBRE ANSIEDAD - JACK MORENO #### 

# Ruta completa del archivo
ruta = os.path.join(lexicon_dir, fichero_listado_palabras_ansiedad)

# Lista para almacenar las palabras sobre ansiedad
listado_palabras_ansiedad = []

# Leer el archivo y almacenar las palabras en la lista
with open(ruta, 'r', encoding='utf-8') as file:
    for line in file:
        listado_palabras_ansiedad.extend(line.strip().split())




#### OBTENER LISTADO DE PALABRAS NEGATIVAS - ISOL #### 

# Ruta completa del archivo
ruta = os.path.join(lexicon_dir, fichero_listado_palabras_negativas)

# Lista para almacenar las palabras negativas
listado_palabras_negativas = []

# Leer el archivo y almacenar las palabras en la lista
with open(ruta, 'r', encoding='utf-8') as file:
    for line in file:
        listado_palabras_negativas.extend(line.strip().split())




#### LEXICON DE PALABRAS CON ANSIEDAD - FICHEROS DE ENTRENAMIENTO #### 


# Función para tokenizar y contar palabras por tipo
def contar_palabras(message):
    # Tokenizar el mensaje usando spaCy
    doc = nlp(message)

    # Filtrar y contar verbos, sustantivos, adjetivos y adverbios
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    adjectives = [token.lemma_ for token in doc if token.pos_ == "ADJ"]
    adverbs = [token.lemma_ for token in doc if token.pos_ == "ADV"]

    return Counter(verbs), Counter(nouns), Counter(adjectives), Counter(adverbs)


# Función para procesar el archivo JSON y contar palabras por tipo
def contar_palabras_en_fichero(file_path):
    # Abrir el archivo JSON
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Inicializar contadores para cada tipo de palabra
    verb_counter_fichero = Counter()
    noun_counter_fichero = Counter()
    adj_counter_fichero = Counter()
    adv_counter_fichero = Counter()

    # Iterar sobre los mensajes en el archivo JSON
    for message in data:
        # Contar palabras por tipo en el mensaje actual
        verbs, nouns, adjectives, adverbs = contar_palabras(message['message'])

        # Actualizar los contadores totales
        verb_counter_fichero.update(verbs)
        noun_counter_fichero.update(nouns)
        adj_counter_fichero.update(adjectives)
        adv_counter_fichero.update(adverbs)

    # Devolver los contadores finales
    return verb_counter_fichero, noun_counter_fichero, adj_counter_fichero, adv_counter_fichero


# Función para generar el lexicón de palabras
def lexicon():
    # Inicializar contadores totales
    verb_total = Counter()
    noun_total = Counter()
    adj_total = Counter()
    adv_total = Counter()

    
    # Iterar sobre los archivos en el directorio de entrenamiento
    for filename in os.listdir(lexicon_propio_dir):
        if filename.endswith(".json"):
            # Obtener la ruta completa del archivo
            file_path = os.path.join(lexicon_propio_dir, filename)

            # Obtener los contadores de palabras por tipo para el archivo actual
            verb_counter, noun_counter, adj_counter, adv_counter = contar_palabras_en_fichero(file_path)

            # Actualizar los contadores totales
            verb_total.update(verb_counter)
            noun_total.update(noun_counter)
            adj_total.update(adj_counter)
            adv_total.update(adv_counter)

    # Devolver los contadores finales
    return verb_total, noun_total, adj_total, adv_total


# Generar lexicon
verb_counter_lexicon, noun_counter_lexicon, adj_counter_lexicon, adv_counter_lexicon = lexicon()




#### CLASIFICACIÓN FICHEROS DE TEST CON CONJUNTO FICHEROS DE ENTRENAMIENTO - LEXICON PROPIO #### 

# Función para verificar si un mensaje contiene al menos una de las palabras más comunes en una lista
def verificar_coincidencias(message, word_list):
    for word, _ in word_list:   
        if word in message:     # Comprobar
            return True
    return False


# Función para contar el número de ocurrencias de las palabras del texto en los listados
def contar_palabras_coincidencias(texto, lista_palabras):
    # Convertir el texto a minúsculas para hacer la búsqueda insensible a mayúsculas
    texto = texto.lower()
    # Dividir el texto en palabras
    palabras_texto = texto.split()
    
    # Inicializar una variable para contar el total de ocurrencias
    total_ocurrencias = 0
    
    # Contar las ocurrencias de cada palabra en el texto
    for palabra, _ in lista_palabras:
        # Contar la ocurrencia de cada palabra en el texto y sumarla al total
        total_ocurrencias += palabras_texto.count(palabra.lower())
    
    # Devolver el total de ocurrencias como un entero
    return total_ocurrencias




#### PREDICCIONES #### 

# Función para obtener si hay más valores '1' o '0'
def obtener_prediccion_mayoritaria(predictions):
    # Inicializar contadores para 1, 0
    count_1 = 0
    count_0 = 0
    
    # Contar la cantidad de veces que aparece cada valor en las predicciones
    for prediction in predictions:
        if prediction == "1":
            count_1 += 1
        elif prediction == "0":
            count_0 += 1
    
    # Determinar el valor mayoritario
    if count_1 > count_0:
        return "1"
    elif count_0 > count_1:
        return "0"
    else:
        return None


# Función para obtener media valores
def obtener_media(predictions):
    # Convertir los strings a floats
    valores = list(map(float, predictions))

    # Calcular la media
    media = sum(valores) / len(valores)

    return media




#### INTERFAZ ####


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



# Algoritmos de Clasificación
@app.route('/algoritmo_clasificacion', methods=['POST'])
def algoritmo_clasificacion():
    global ficheros_test
    global mensajes_test
    global proceso_vectorizacion

    ficheros_test = request.files.getlist('files')
    mensajes_test = []

    obtener_mensajes_test()
    
    proceso_vectorizacion = request.form['vectorizacion']

    selected_algorithm = request.form.get('algoritmo')
    if selected_algorithm == 'svm_c':
        return redirect(url_for('svm_clasificacion'))
    elif selected_algorithm == 'nb_c':
        return redirect(url_for('nb_clasificacion'))
    elif selected_algorithm == 'tree_c':
        return redirect(url_for('tree_clasificacion'))
    elif selected_algorithm == 'rf_c':
        return redirect(url_for('rf_clasificacion'))
    elif selected_algorithm == 'knn_c':
        return redirect(url_for('knn_clasificacion'))
    elif selected_algorithm == 'lr_c':
        return redirect(url_for('lr_clasificacion'))
    elif selected_algorithm == 'ensemble_c':
        return redirect(url_for('ensemble_clasificacion'))
    else:
        flash('Selecciona un algoritmo válido.', 'error')
        return redirect(url_for('index'))



# Algoritmo SVM - Clasificación
@app.route('/svm_clasificacion', methods=['GET', 'POST'])
def svm_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    #   - Kernel - 'rbf', 'linear', 'precomputed', 'poly', 'sigmoid'
    model_svm = svm.SVC(kernel='linear')


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"
        
        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = model_svm.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = model_svm.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions = model_svm.predict(Y_W2V_SG)

    elif proceso_vectorizacion == "w2v_cbow_c":
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions = model_svm.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>SVM – {vectorizacion}</u></p>"
    # print("Predicciones con Máquinas de Vectores de Soporte - SVM:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Algoritmo Naive Bayes - Clasificación
@app.route('/nb_clasificacion', methods=['GET', 'POST'])
def nb_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_nb = MultinomialNB()


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_nb.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = model_nb.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_nb.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = model_nb.predict(Y_TF_IDF)

    else:
        return render_template('index.html', prediction_clasificacion="<p style = 'margin-left:50px; margin-bottom:0px;'>Naive Bayes <b>incompatible</b> con Word2Vec</p>")


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Naive Bayes – {vectorizacion}</u></p>"
    # print("Predicciones con Naive Bayes:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Algoritmo Árboles de decisión - Clasificación
@app.route('/tree_clasificacion', methods=['GET', 'POST'])
def tree_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_decision_tree = tree.DecisionTreeClassifier()


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_decision_tree.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = model_decision_tree.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_decision_tree.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = model_decision_tree.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_decision_tree.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions = model_decision_tree.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_decision_tree.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions = model_decision_tree.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Árboles de decisión – {vectorizacion}</u></p>"
    # print("Predicciones con Árboles de decisión:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Algoritmo Random Forest - Clasificación
@app.route('/rf_clasificacion', methods=['GET', 'POST'])
def rf_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_rf.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = model_rf.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_rf.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = model_rf.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_rf.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions = model_rf.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_rf.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions = model_rf.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Random Forest – {vectorizacion}</u></p>"
    # print("Predicciones con Random Forest:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"

        
    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Algoritmo K-vecinos - Clasificación
@app.route('/knn_clasificacion', methods=['GET', 'POST'])
def knn_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    knn_classifier = KNeighborsClassifier(n_neighbors=8)


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        knn_classifier.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = knn_classifier.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        knn_classifier.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = knn_classifier.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        knn_classifier.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions = knn_classifier.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        knn_classifier.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions = knn_classifier.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>K-vecinos – {vectorizacion}</u></p>"
    # print("Predicciones con K-vecinos:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Algoritmo Logistic Regression - Clasificación
@app.route('/lr_clasificacion', methods=['GET', 'POST'])
def lr_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_lr = LogisticRegression(max_iter=1000)


    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_lr.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions = model_lr.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_lr.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions = model_lr.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_lr.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions = model_lr.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_lr.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions = model_lr.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Logistic Regression – {vectorizacion}</u></p>"
    # print("Predicciones con Logistic Regression:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"

        
    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)



# Ensemble
@app.route('/ensemble_clasificacion', methods=['GET', 'POST'])
def ensemble_clasificacion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test
   
    # Creación del modelo
    model_svm = svm.SVC(kernel='linear')
    model_nb = MultinomialNB()
    model_decision_tree = tree.DecisionTreeClassifier()
    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    knn_classifier = KNeighborsClassifier(n_neighbors=8)
    model_lr = LogisticRegression(max_iter=1000)

    predictions_svm = []
    predictions_nb = []
    predictions_decision_tree = []
    predictions_rf = []
    predictions_knn = []
    predictions_lr = []


    for i in enumerate(documents_test):
        predictions_svm.append(None)
        predictions_nb.append(None)
        predictions_decision_tree.append(None)
        predictions_rf.append(None)
        predictions_knn.append(None)
        predictions_lr.append(None)

    if proceso_vectorizacion == "bow_c":
        vectorizacion = "BOW"
        
        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_BOW, listado_etiquetas_A)
        model_nb.fit(X_BOW, listado_etiquetas_A)
        model_decision_tree.fit(X_BOW, listado_etiquetas_A)
        model_rf.fit(X_BOW, listado_etiquetas_A)
        knn_classifier.fit(X_BOW, listado_etiquetas_A)
        model_lr.fit(X_BOW, listado_etiquetas_A)

        # Predicción
        predictions_svm = model_svm.predict(Y_BOW)
        predictions_nb = model_nb.predict(Y_BOW)
        predictions_decision_tree = model_decision_tree.predict(Y_BOW)
        predictions_rf = model_rf.predict(Y_BOW)
        predictions_knn = knn_classifier.predict(Y_BOW)
        predictions_lr = model_lr.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_c":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_TF_IDF, listado_etiquetas_A)
        model_nb.fit(X_TF_IDF, listado_etiquetas_A)
        model_decision_tree.fit(X_TF_IDF, listado_etiquetas_A)
        model_rf.fit(X_TF_IDF, listado_etiquetas_A)
        knn_classifier.fit(X_TF_IDF, listado_etiquetas_A)
        model_lr.fit(X_TF_IDF, listado_etiquetas_A)

        # Predicción
        predictions_svm = model_svm.predict(Y_TF_IDF)
        predictions_nb = model_nb.predict(Y_TF_IDF)
        predictions_decision_tree = model_decision_tree.predict(Y_TF_IDF)
        predictions_rf = model_rf.predict(Y_TF_IDF)
        predictions_knn = knn_classifier.predict(Y_TF_IDF)
        predictions_lr = model_lr.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_c":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_SG, listado_etiquetas_A)
        model_decision_tree.fit(X_W2V_SG, listado_etiquetas_A)
        model_rf.fit(X_W2V_SG, listado_etiquetas_A)
        knn_classifier.fit(X_W2V_SG, listado_etiquetas_A)
        model_lr.fit(X_W2V_SG, listado_etiquetas_A)

        # Predicción
        predictions_svm = model_svm.predict(Y_W2V_SG)
        predictions_decision_tree = model_decision_tree.predict(Y_W2V_SG)
        predictions_rf = model_rf.predict(Y_W2V_SG)
        predictions_knn = knn_classifier.predict(Y_W2V_SG)
        predictions_lr = model_lr.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_CBOW, listado_etiquetas_A)
        model_decision_tree.fit(X_W2V_CBOW, listado_etiquetas_A)
        model_rf.fit(X_W2V_CBOW, listado_etiquetas_A)
        knn_classifier.fit(X_W2V_CBOW, listado_etiquetas_A)
        model_lr.fit(X_W2V_CBOW, listado_etiquetas_A)

        # Predicción
        predictions_svm = model_svm.predict(Y_W2V_CBOW)
        predictions_decision_tree = model_decision_tree.predict(Y_W2V_CBOW)
        predictions_rf = model_rf.predict(Y_W2V_CBOW)
        predictions_knn = knn_classifier.predict(Y_W2V_CBOW)
        predictions_lr = model_lr.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_clasificacion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Ensemble – {vectorizacion}</u></p>"
    # print("Predicciones con Ensemble:")

    for i, prediction_svm, prediction_nb, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr in zip(range(len(predictions_svm)), predictions_svm, predictions_nb, predictions_decision_tree, predictions_rf, predictions_knn, predictions_lr):
        nombre_fichero = ficheros_test[i].filename

        predictions = [prediction_svm, prediction_nb, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr]

        prediction = obtener_prediccion_mayoritaria(predictions)

        if prediction == "1":
            prediction_clasificacion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        elif prediction == "0":
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        else:
            prediction_clasificacion +=  f"<p style = 'margin-left:50px; color:gray; margin-bottom:0px;'><i>Incertidumbre para el usuario del documento '{nombre_fichero}'</i></p>"
  
        # Resultados algoritmos
        texto_etiquetas = []
        
        for i, prediccion in zip(range(len(predictions)), predictions):
            if prediccion == "1":
                texto_etiquetas.append("<span style='color: red;'>Ansiedad</span>")
            elif prediccion == "0":
                texto_etiquetas.append("<span style='color: green;'>No ansiedad</span>")
            else:
                texto_etiquetas.append("<span style='color: gray;'>Incompatible</span>")
    

        prediction_clasificacion += f"<ul style = 'margin-left:70px;'><li><b>SVM: </b>{texto_etiquetas[0]}</li><li><b>Naive Bayes: </b>{texto_etiquetas[1]}</li><li><b>Árboles de decisión: </b>{texto_etiquetas[2]}</li><li><b>Random Forest: </b>{texto_etiquetas[3]}</li><li><b>K-vecinos: </b>{texto_etiquetas[4]}</li><li><b>Logistic Regression: </b>{texto_etiquetas[5]}</li></ul>"
        

    return render_template('index.html', prediction_clasificacion=prediction_clasificacion)




# Algoritmos de Regresión
@app.route('/algoritmo_regresion', methods=['POST'])
def algoritmo_regresion():
    global ficheros_test
    global mensajes_test
    global proceso_vectorizacion

    ficheros_test = request.files.getlist('files')
    mensajes_test = []

    obtener_mensajes_test()
    
    proceso_vectorizacion = request.form['vectorizacion']

    selected_algorithm = request.form.get('algoritmo')
    if selected_algorithm == 'svm_r':
        return redirect(url_for('svm_regresion'))
    elif selected_algorithm == 'tree_r':
        return redirect(url_for('tree_regresion'))
    elif selected_algorithm == 'rf_r':
        return redirect(url_for('rf_regresion'))
    elif selected_algorithm == 'knn_r':
        return redirect(url_for('knn_regresion'))
    elif selected_algorithm == 'lr_r':
        return redirect(url_for('lr_regresion'))
    elif selected_algorithm == 'ensemble_r':
        return redirect(url_for('ensemble_regresion'))
    else:
        flash('Selecciona un algoritmo válido.', 'error')
        return redirect(url_for('index'))



# Algoritmo SVM - Regresión
@app.route('/svm_regresion', methods=['GET', 'POST'])
def svm_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_svm = svm.SVR(kernel='linear')


    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_BOW, listado_etiquetas_B)

        # Predicción
        predictions = model_svm.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_TF_IDF, listado_etiquetas_B)

        # Predicción
        predictions = model_svm.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_SG, listado_etiquetas_B)

        # Predicción
        predictions = model_svm.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_CBOW, listado_etiquetas_B)

        # Predicción
        predictions = model_svm.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>SVM – {vectorizacion}</u></p>"
    # print("Predicciones con Máquinas de Vectores de Soporte - SVM:")

    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)



# Algoritmo Árboles de decisión - Regresión
@app.route('/tree_regresion', methods=['GET', 'POST'])
def tree_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_decision_tree = tree.DecisionTreeRegressor()


    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_decision_tree.fit(X_BOW, listado_etiquetas_B)

        # Predicción
        predictions = model_decision_tree.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_decision_tree.fit(X_TF_IDF, listado_etiquetas_B)

        # Predicción
        predictions = model_decision_tree.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_decision_tree.fit(X_W2V_SG, listado_etiquetas_B)

        # Predicción
        predictions = model_decision_tree.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_decision_tree.fit(X_W2V_CBOW, listado_etiquetas_B)

        # Predicción
        predictions = model_decision_tree.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Árboles de decisión – {vectorizacion}</u></p>"
    # print("Predicciones con Árboles de decisión:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)



# Algoritmo Random Forest - Regresión
@app.route('/rf_regresion', methods=['GET', 'POST'])
def rf_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_rf = RandomForestRegressor(n_estimators=200, random_state=42)


    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_rf.fit(X_BOW, listado_etiquetas_B)

        # Predicción
        predictions = model_rf.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_rf.fit(X_TF_IDF, listado_etiquetas_B)

        # Predicción
        predictions = model_rf.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_rf.fit(X_W2V_SG, listado_etiquetas_B)

        # Predicción
        predictions = model_rf.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_rf.fit(X_W2V_CBOW, listado_etiquetas_B)

        # Predicción
        predictions = model_rf.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Random Forest – {vectorizacion}</u></p>"
    # print("Predicciones con Random Forest:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)



# Algoritmo K-vecinos - Regresión
@app.route('/knn_regresion', methods=['GET', 'POST'])
def knn_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    knn_regressor  = KNeighborsRegressor(n_neighbors=8)

    etiquetas_knn = np.array(listado_etiquetas_B, dtype=float)

    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        knn_regressor.fit(X_BOW, etiquetas_knn)

        # Predicción
        predictions = knn_regressor.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        knn_regressor.fit(X_TF_IDF, etiquetas_knn)

        # Predicción
        predictions = knn_regressor.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        knn_regressor.fit(X_W2V_SG, etiquetas_knn)

        # Predicción
        predictions = knn_regressor.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        knn_regressor.fit(X_W2V_CBOW, etiquetas_knn)

        # Predicción
        predictions = knn_regressor.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>K-vecinos – {vectorizacion}</u></p>"
    # print("Predicciones con K-vecinos:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)



# Algoritmo Logistic Regression - Regresión
@app.route('/lr_regresion', methods=['GET', 'POST'])
def lr_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test

    # Creación del modelo
    model_lr = LogisticRegression(max_iter=1000)


    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"

        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_lr.fit(X_BOW, listado_etiquetas_B)

        # Predicción
        predictions = model_lr.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_lr.fit(X_TF_IDF, listado_etiquetas_B)

        # Predicción
        predictions = model_lr.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_lr.fit(X_W2V_SG, listado_etiquetas_B)

        # Predicción
        predictions = model_lr.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_lr.fit(X_W2V_CBOW, listado_etiquetas_B)

        # Predicción
        predictions = model_lr.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Logistic Regression – {vectorizacion}</u></p>"
    # print("Predicciones con Logistic Regression:")
    for i, prediction in enumerate(predictions):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)


# Ensemble
@app.route('/ensemble_regresion', methods=['GET', 'POST'])
def ensemble_regresion():
    global proceso_vectorizacion
    vectorizacion = ""

    documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test
   
    # Creación del modelo
    model_svm = svm.SVR(kernel='linear')
    model_decision_tree = tree.DecisionTreeRegressor()
    model_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    knn_regressor = KNeighborsRegressor(n_neighbors=8)
    model_lr = LogisticRegression(max_iter=1000)


    predictions_svm = []
    predictions_decision_tree = []
    predictions_rf = []
    predictions_knn = []
    predictions_lr = []


    if proceso_vectorizacion == "bow_r":
        vectorizacion = "BOW"
        
        Y_BOW = vectorizer_BOW.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_BOW, listado_etiquetas_B)
        model_decision_tree.fit(X_BOW, listado_etiquetas_B)
        model_rf.fit(X_BOW, listado_etiquetas_B)

        etiquetas_knn = np.array(listado_etiquetas_B, dtype=float)
        knn_regressor.fit(X_BOW, etiquetas_knn)

        model_lr.fit(X_BOW, listado_etiquetas_B)

        # Predicción
        predictions_svm = model_svm.predict(Y_BOW)
        predictions_decision_tree = model_decision_tree.predict(Y_BOW)
        predictions_rf = model_rf.predict(Y_BOW)
        predictions_knn = knn_regressor.predict(Y_BOW)
        predictions_lr = model_lr.predict(Y_BOW)

    elif proceso_vectorizacion == "tfidf_r":
        vectorizacion = "TF–IDF"

        Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

        # Entrenamiento del modelo
        model_svm.fit(X_TF_IDF, listado_etiquetas_B)
        model_decision_tree.fit(X_TF_IDF, listado_etiquetas_B)
        model_rf.fit(X_TF_IDF, listado_etiquetas_B)
        
        etiquetas_knn = np.array(listado_etiquetas_B, dtype=float)
        knn_regressor.fit(X_TF_IDF, etiquetas_knn)

        model_lr.fit(X_TF_IDF, listado_etiquetas_B)

        # Predicción
        predictions_svm = model_svm.predict(Y_TF_IDF)
        predictions_decision_tree = model_decision_tree.predict(Y_TF_IDF)
        predictions_rf = model_rf.predict(Y_TF_IDF)
        predictions_knn = knn_regressor.predict(Y_TF_IDF)
        predictions_lr = model_lr.predict(Y_TF_IDF)

    elif proceso_vectorizacion == "w2v_sg_r":
        vectorizacion = "W2V – SG"

        Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_SG, listado_etiquetas_B)
        model_decision_tree.fit(X_W2V_SG, listado_etiquetas_B)
        model_rf.fit(X_W2V_SG, listado_etiquetas_B)
        
        etiquetas_knn = np.array(listado_etiquetas_B, dtype=float)
        knn_regressor.fit(X_W2V_SG, etiquetas_knn)

        model_lr.fit(X_W2V_SG, listado_etiquetas_B)

        # Predicción
        predictions_svm = model_svm.predict(Y_W2V_SG)
        predictions_decision_tree = model_decision_tree.predict(Y_W2V_SG)
        predictions_rf = model_rf.predict(Y_W2V_SG)
        predictions_knn = knn_regressor.predict(Y_W2V_SG)
        predictions_lr = model_lr.predict(Y_W2V_SG)

    else:
        vectorizacion = "W2V – CBOW"

        Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]

        # Entrenamiento del modelo
        model_svm.fit(X_W2V_CBOW, listado_etiquetas_B)
        model_decision_tree.fit(X_W2V_CBOW, listado_etiquetas_B)
        model_rf.fit(X_W2V_CBOW, listado_etiquetas_B)
        
        etiquetas_knn = np.array(listado_etiquetas_B, dtype=float)
        knn_regressor.fit(X_W2V_CBOW, etiquetas_knn)

        model_lr.fit(X_W2V_CBOW, listado_etiquetas_B)

        # Predicción
        predictions_svm = model_svm.predict(Y_W2V_CBOW)
        predictions_decision_tree = model_decision_tree.predict(Y_W2V_CBOW)
        predictions_rf = model_rf.predict(Y_W2V_CBOW)
        predictions_knn = knn_regressor.predict(Y_W2V_CBOW)
        predictions_lr = model_lr.predict(Y_W2V_CBOW)


    # Mostrar predicción
    prediction_regresion = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Ensemble – {vectorizacion}</u></p>"
    # print("Predicciones con Ensemble:")

    for i, prediction_svm, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr in zip(range(len(predictions_svm)), predictions_svm, predictions_decision_tree, predictions_rf, predictions_knn, predictions_lr):
        nombre_fichero = ficheros_test[i].filename

        predictions = [prediction_svm, prediction_decision_tree, prediction_rf, prediction_knn, float(prediction_lr)]

        prediction = obtener_media(predictions)

        if prediction >= 0.5:
            prediction_regresion += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_regresion +=  f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
  
        # Resultados algoritmos
        texto_etiquetas = []
        
        for i, prediccion in zip(range(len(predictions)), predictions):
            if float(prediccion) >= 0.5:
                texto_etiquetas.append("<span style='color: red;'>Ansiedad</span>")
            else:
                texto_etiquetas.append("<span style='color: green;'>No ansiedad</span>")
    

        prediction_regresion += f"<ul style = 'margin-left:70px;'><li><b>SVM: </b>{texto_etiquetas[0]}</li><li><b>Árboles de decisión: </b>{texto_etiquetas[1]}</li><li><b>Random Forest: </b>{texto_etiquetas[2]}</li><li><b>K-vecinos: </b>{texto_etiquetas[3]}</li><li><b>Logistic Regression: </b>{texto_etiquetas[4]}</li></ul>"
        

    return render_template('index.html', prediction_regresion=prediction_regresion)




# Lexicones de Palabras
@app.route('/lexicones', methods=['POST'])
def lexicones():
    global ficheros_test
    global mensajes_test

    ficheros_test = request.files.getlist('files')
    mensajes_test = []

    obtener_mensajes_test()

    selected_lexicon = request.form.get('algoritmo')
    if selected_lexicon == 'lex_propio':
        return redirect(url_for('lex_propio'))
    elif selected_lexicon == 'lex_ans':
        return redirect(url_for('lex_ans'))
    elif selected_lexicon == 'lex_neg':
        return redirect(url_for('lex_neg'))
    else:
        flash('Selecciona un léxico válido.', 'error')
        return redirect(url_for('index'))


# Lexicon propio
@app.route('/lex_propio', methods=['GET', 'POST'])
def lex_propio():
    global verb_counter_lexicon, noun_counter_lexicon, adj_counter_lexicon, adv_counter_lexicon 
    global ficheros_test
    global mensajes_test

    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 85

    # Iterar sobre los archivos en el directorio de test
    for mensaje in mensajes_test:
        # Contar coincidencias en cada archivo
        coincidencias_total = 0

        # Contar palabras por tipo en el mensaje actual
        verbs, nouns, adjectives, adverbs = contar_palabras(mensaje)
                
        # Verificar coincidencias en cada tipo de palabra
        if verificar_coincidencias(mensaje, verb_counter_lexicon.most_common(15)):
            coincidencias_total += contar_palabras_coincidencias(mensaje, verb_counter_lexicon.most_common(15))
        if verificar_coincidencias(mensaje, noun_counter_lexicon.most_common(15)):
            coincidencias_total += contar_palabras_coincidencias(mensaje, noun_counter_lexicon.most_common(15))
        if verificar_coincidencias(mensaje, adj_counter_lexicon.most_common(15)):
            coincidencias_total += contar_palabras_coincidencias(mensaje, adj_counter_lexicon.most_common(15))

        # Categorizar el archivo
        if coincidencias_total >= umbral_coincidencias:
            etiquetas_ficheros.append("1")
        else:
            etiquetas_ficheros.append("0")


    # Mostrar predicción
    prediction_lexicon = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Lexicon Propio</u></p>"
    # print("Predicciones con Lexicon Propio:")
    for i, prediction in enumerate(etiquetas_ficheros):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_lexicon += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_lexicon += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_lexicon=prediction_lexicon)


# Lexicon proporcionado - Palabras con ansiedad
@app.route('/lex_ans', methods=['GET', 'POST'])
def lex_ans():
    global ficheros_test
    global mensajes_test

    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 5

    # Convertir la lista de palabras de ansiedad a un conjunto para una búsqueda más eficiente
    palabras_ansiedad = set(listado_palabras_ansiedad)

    # Iterar sobre los ficheros en el directorio
    for mensaje in mensajes_test:
        # Contador de ocurrencias de palabras de ansiedad
        contador_ansiedad = 0

        for palabra in mensaje.strip().split():
            if palabra in palabras_ansiedad:
                contador_ansiedad += 1
            
        # Clasificar el fichero según el contador
        if contador_ansiedad >= umbral_coincidencias:
            etiquetas_ficheros.append("1")
        else:
            etiquetas_ficheros.append("0")
        

    # Mostrar predicción
    prediction_lexicon = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Lexicón proporcionado  – Listado palabras sobre ansiedad</u></p>"
    # print("Predicciones con Lexicon Proporcionado - Palabras sobre ansiedad:")
    for i, prediction in enumerate(etiquetas_ficheros):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_lexicon += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_lexicon += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_lexicon=prediction_lexicon)


# Lexicon proporcionado - Palabras negativas
@app.route('/lex_neg', methods=['GET', 'POST'])
def lex_neg():
    global ficheros_test
    global mensajes_test

    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 25

    # Convertir la lista de palabras negativas a un conjunto para una búsqueda más eficiente
    palabras_negativas = set(listado_palabras_negativas)

     # Iterar sobre los ficheros en el directorio
    for mensaje in mensajes_test:
        # Contador de ocurrencias de palabras negativas
        contador_negativas = 0

        for palabra in mensaje.strip().split():
            if palabra in palabras_negativas:
                contador_negativas += 1
            
        # Clasificar el fichero según el contador
        if contador_negativas >= umbral_coincidencias:
            etiquetas_ficheros.append("1")
        else:
            etiquetas_ficheros.append("0")
        
        print("Número de ocurrencias - Palabras negativas: ", contador_negativas)


    # Mostrar predicción
    prediction_lexicon = f"<p style = 'margin-left:50px; margin-bottom:0px;'><u>Lexicón proporcionado  – Listado palabras negativas</u></p>"
    # print("Predicciones con Lexicon Proporcionado - Palabras negativas:")
    for i, prediction in enumerate(etiquetas_ficheros):
        nombre_fichero = ficheros_test[i].filename
        if prediction == "1":
            prediction_lexicon += f"<p style = 'margin-left:50px; color:red; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' sufre ansiedad</i></p>"
        else:
            prediction_lexicon += f"<p style = 'margin-left:50px; color:green; margin-bottom:0px;'><i>El usuario del documento '{nombre_fichero}' no sufre ansiedad</i></p>"
        

    return render_template('index.html', prediction_lexicon=prediction_lexicon)




if __name__ == '__main__':
    app.run(debug=True)