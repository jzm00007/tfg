# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Algoritmos de aprendizaje supervisado



import os                                   # Importa módulo 'os' - Acceder a funcionalidades del Sistema Operativo
import json                                 # Importa biblioteca para formato JSON
import nltk                                 # Importa biblioteca NLTK - Natural Language Toolkit - Herramientas PLN
import spacy                                # Importa biblioteca SpaCy - Herramientas PLN
import numpy as np                          # Importa biblioteca NumPy - Trabajar con matrices y operaciones matemáticas de alto nivel


from sklearn import svm                                 # Máquinas de vectores de soporte - SVM 
from sklearn.naive_bayes import MultinomialNB           # Naive Bayes
from sklearn import tree                                # Árboles de decisión
from sklearn.ensemble import RandomForestClassifier     # Random Forest - Clasificación
from sklearn.ensemble import RandomForestRegressor      # Random Forest - Regresión
from sklearn.neighbors import KNeighborsClassifier      # K vecinos - Clasificación
from sklearn.neighbors import KNeighborsRegressor       # K vecinos - Regresión
from sklearn.linear_model import LogisticRegression     # Logistic Regression


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error           # R2, MSE, MAE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score     # Accurancy, Precision, Recall, F1
from sklearn.metrics import confusion_matrix                                            # Matriz de confusión


from nltk.corpus import stopwords                               # Importa conjunto de datos de stopwords - Conjunto palabras comunes generalmente irrelevantes 
from sklearn.feature_extraction.text import CountVectorizer     # Permite la vectorización de Bag-of-Words
from sklearn.feature_extraction.text import TfidfVectorizer     # Permite la vectorización de TF-IDF         
from gensim.models import Word2Vec                              # Permite la vectorización de Word2vec


import re                           # Importa módulo 're' - Manejo expresiones regulares
import emoji                        # Importa biblioteca 'emoji' - Convertir emojis gráficos en texto descriptivo
from googletrans import Translator  # Importa biblioteca 'Translator' - Traducción de texto -  Google Translate
import string                       # Importa módulo 'string' - Manipular cadena de caracteres


import sys      # Acceso a algunas variables y funciones que interactúan con el intérprete de Python
import io       # Proporciona las herramientas necesarias para trabajar con flujos de entrada y salida en Python



# Descargar recursos para almacenamiento en sistema
nltk.download('stopwords')

# Obtener las stopwords en español
stopwords_es = list(set(stopwords.words('spanish')))


# Cargar modelo de idioma español de SpaCy
# Modelo en español - Cantidad datos y capacidad vectorización más grande
# spacy.cli.download("es_core_news_lg")   
import es_core_news_lg    
nlp = es_core_news_lg.load()



# Definir parámetros necesarios                
directorio = 'C:\\Ficheros sobre Ansiedad'          # Directorio principal
entrenamiento_dir = r"C:\Ficheros sobre Ansiedad/Archivos entrenamiento"     # Obtener directorio de ficheros entrenamiento
test_dir = r"C:\Ficheros sobre Ansiedad/Archivos test"                       # Obtener directorio de ficheros test
nombre_fichero_etiquetas_A = "gold_a.txt"           # Nombre del archivo etiquetado A
nombre_fichero_etiquetas_B = "gold_b.txt"           # Nombre del archivo etiquetado B
resultados = "output_algoritmos.txt"                # Fichero con resultados




# Verificar y crear el directorio si no existe
if not os.path.exists(directorio):
    os.makedirs(directorio)

# Crear un objeto StringIO para capturar la salida
old_stdout = sys.stdout
sys.stdout = io.StringIO()




#### OBTENER LISTADO DE FICHEROS ENTRENAMIENTO EN DIRECTORIO #### 

# Lista para almacenar los nombres de los archivos de entrenamiento
lista_ficheros_entrenamiento = []

# Obtener los nombres de los archivos en la carpeta entrenamiento
for fichero in os.listdir(entrenamiento_dir):   
    if fichero.endswith(".json"):  # Solo añadir archivos con extensión .json
        lista_ficheros_entrenamiento.append(fichero)

# Imprimir los nombres de los archivos
# print("Nombres de archivos de entrenamiento en la carpeta:")
# for nombre in lista_ficheros_entrenamiento:
#    print(nombre)
# print()




#### OBTENER LISTADO DE FICHEROS TEST EN DIRECTORIO #### 

# Lista para almacenar los nombres de los archivos de test
lista_ficheros_test = []

# Obtener los nombres de los archivos en la carpeta test
for fichero in os.listdir(test_dir):   
    if fichero.endswith(".json"):  # Solo añadir archivos con extensión .json
        lista_ficheros_test.append(fichero)

# Imprimir los nombres de los archivos
print("Nombres de archivos de test en la carpeta:")
for nombre in lista_ficheros_test:
   print(nombre)
print()




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


# Imprimir los resultados
# print("Sujetos y etiquetas del conjunto de entrenamiento:")
# for i in range(len(listado_sujetos)):
#     print(listado_sujetos[i], ";", listado_etiquetas_A[i], ";",listado_etiquetas_B[i])
# print()

# Imprimir los archivos a procesar
# print("Archivos de entrenamiento - sujetos a procesar:")
# for archivo in archivos_entrenamiento:
#     print(archivo)
# print()




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

# Mostrar los documentos
print("Archivos de entrenamiento:")
for i, documento in enumerate(documents):
    print(f"Documento {i+1}:", f"[{listado_sujetos[i]}, {listado_etiquetas_A[i]}, {listado_etiquetas_B[i]}]")
    #print(documento)
    #print()
# print(documents_A)
print()




#### OBTENER CONTENIDO MENSAJES DE TEST ####

# Lista para almacenar los mensajes de todos los archivos JSON de test
mensajes_test = []

# Iterar sobre cada nombre de archivo en la lista de archivos de test
for nombre_archivo in lista_ficheros_test:
    # Construir la ruta completa del archivo JSON
    ruta_json = os.path.join(test_dir, nombre_archivo)
    
    # Verificar si el archivo JSON existe
    if os.path.exists(ruta_json):
        # Abrir el archivo JSON
        with open(ruta_json, 'r', encoding='utf-8') as archivo_json:
            # Cargar el contenido JSON
            contenido_json = json.load(archivo_json)
            
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


# Mostrar los mensajes de test almacenados
# print("\nMensajes de los archivos JSON de test:")
# for i, mensajes_fichero in enumerate(mensajes_test):
#    print(f"Archivo {i+1}:")
#    print(mensajes_fichero)
#    print()
# print(mensajes_test)
# print()




#### EXTRACCIÓN DE CARACTERÍSTICAS - VECTORIZACIÓN ####


# Crear un objeto Translator
translator = Translator()

# Función para Limpieza del Texto
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

    # Eliminar signos de puntuación
    # translator_table = str.maketrans('', '', string.punctuation + '¿¡')
    # texto = texto.translate(translator_table)

    return texto


# Función para preprocesar los documentos
# - Limpieza de texto
# - Tokenización
# - Eliminación de palabras vacías
# - Conversión palabras a minúscula
# - Lematización
def preprocesamiento_texto(document):
    texto = limpieza_texto(document)

    # Procesamiento del texto con SpaCy
    doc = nlp(texto)
    
    # Lematización, tokenización, conversión a minúsculas y eliminación de palabras vacías
    tokens = [token.lemma_ for token in doc if token.lemma_.lower() not in stopwords_es]
    
    return " ".join(tokens)


# Obtener conjunto de mensajes
documents_train = [preprocesamiento_texto(mensaje) for mensaje in documents]        # Entrenamiento
documents_test = [preprocesamiento_texto(mensaje) for mensaje in mensajes_test]     # Test


# Mostrar los documentos entrenamiento preprocesados
# for i, documento in enumerate(documents):
#    print(f"Documento {i+1}:", f"[{listado_sujetos[i]}, {listado_etiquetas_A[i]}, {listado_etiquetas_B[i]}]")
#    print(documento)
#    print()
# print(documents)
# print()
    
# Mostrar los documentos test preprocesados
# for i, documento in enumerate(documents_test):
#    print(f"Documento {i+1}:", f"{lista_ficheros_test[i]}")
#    print(documento)
#    print()
# print(documents)
# print()



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
Y_BOW = vectorizer_BOW.transform(documents_test)

# Obtener el vocabulario
vocabulario_BOW = vectorizer_BOW.get_feature_names_out()



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
Y_TF_IDF = vectorizer_TF_IDF.transform(documents_test)

# Obtener el vocabulario
vocabulario_TF_IDF = vectorizer_TF_IDF.get_feature_names_out()



# Entrenamiento del modelo Word2vec
#   - Lista documentos utilizados para entrenar el modelo - Entrenamiento + Test
#   - vector_size --> especificar dimensión de los vectores de palabra que el modelo generará
#       - [50, 100, 200], 
#   - window --> especificar ventana máxima de contexto - Cantidad máxima de palabras adyacentes a una palabra
#       - [3, 5, 7]
#   - min_count --> establecer umbral para contar las palabras - Nº mínimo apariciones palabra
#       - [1, 2, 5]
#   - workers --> especificar cantidad de subprocesos utilizados para entrenar el modelo

# Skip-gram --> eficaz para capturar relaciones entre palabras menos frecuentes en un corpus grandes
#   - sg = 1
# CBOW - Continuous Bag of Words --> más rápido de entrenar y  más adecuado para corpus más pequeños y frecuentes
#   - sg = 0

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

# Obtener conjunto de mensajes test - SG
Y_W2V_SG = [representaciones_vector(mensajes_usuarios_test, model_SG) for mensajes_usuarios_test in documents_test]

# Obtener conjunto de mensajes entrenamiento - CBOW
X_W2V_CBOW = [representaciones_vector(mensajes_usuarios, model_CBOW) for mensajes_usuarios in documents_train]

# Obtener conjunto de mensajes test - CBOW
Y_W2V_CBOW = [representaciones_vector(mensajes_usuarios_test, model_CBOW) for mensajes_usuarios_test in documents_test]




#### CLASIFICACIÓN - TIPO A ####




#### EVALUACIÓN DE LOS ALGORITMOS ####

# Función para evaluación del modelo
def metricas_algoritmos_clasificacion(etiquetas_prediccion, predictiones):
    accuracy = accuracy_score(etiquetas_prediccion, predictiones)
    precision_ponderada = precision_score(etiquetas_prediccion, predictiones, average='weighted', zero_division=1)
    recall_ponderada = recall_score(etiquetas_prediccion, predictiones, average='weighted')
    f1_ponderada = f1_score(etiquetas_prediccion, predictiones, average='weighted')

    print("Accuracy:", accuracy)
    print("Precisión:", precision_ponderada)
    print("Recall Ponderada:", recall_ponderada)
    print("Medida F1 Ponderada:", f1_ponderada)
    print()

    # Calcular la matriz de confusión y extraer TP, FP, FN y TN
    tn, fp, fn, tp = confusion_matrix(etiquetas_prediccion, predictiones).ravel()

    print(f"Verdaderos Positivos - VP: {tp}")
    print(f"Falsos Positivos - FP: {fp}")
    print(f"Verdaderos Negativos - VN: {tn}")
    print(f"Falsos Negativos - FN: {fn}")
    print()


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




#### SVM ####

# Función del algoritmo SVM
def algoritmo_svm_clasificacion(X, Y, etiquetas, etiquetas_prediccion):    
    # Creación del modelo
    #   - Kernel - 'rbf', 'linear', 'precomputed', 'poly', 'sigmoid'
    model_svm = svm.SVC(kernel='linear')

    # Entrenamiento del modelo
    model_svm.fit(X, etiquetas)

    # Predicción
    predictions = model_svm.predict(Y)

    # Mostrar predicción
    # print("Predicciones con SVM:")
    for i, prediction in enumerate(predictions):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predictions) 




#### NAIVE BAYES ####

# Función del algoritmo Naive Bayes
def algoritmo_nv(X, Y, etiquetas, etiquetas_prediccion):
    # Creación del modelo
    model_nb = MultinomialNB()

    # Entrenar el modelo de Naive Bayes
    #   - OJO - No valores negativos
    model_nb.fit(X, etiquetas)

    # Realizar predicciones
    predictions_nb = model_nb.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Naive Bayes:")
    for i, prediction in enumerate(predictions_nb):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predictions_nb) 




#### ÁRBOLES DE DECISIÓN ####

# Función del algoritmo Árboles de decisión
def algoritmo_arboles_decision_clasificacion(X, Y, etiquetas, etiquetas_prediccion):
    # Crear un clasificador de Árbol de Decisión
    model_decision_tree = tree.DecisionTreeClassifier()

    # Entrenar el modelo
    model_decision_tree.fit(X, etiquetas)

    # Realizar predicciones
    predictions_decision_tree = model_decision_tree.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Árboles de Decisión:")
    for i, prediction in enumerate(predictions_decision_tree):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predictions_decision_tree) 

   


#### RANDOM FOREST ####

# Función del algoritmo Random Forest
def algoritmo_rf_clasificacion(X, Y, etiquetas, etiquetas_prediccion):
    # Crear modelo Random Forest
    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Entrenar el clasificador Random Forest
    model_rf.fit(X, etiquetas)

    # Realizar predicciones en el conjunto de prueba
    predictions_rf = model_rf.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Random Forest:")
    for i, prediction in enumerate(predictions_rf):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predictions_rf) 




#### K-VECINOS ####

# Función del algoritmo K-vecinos
def algoritmo_k_vecinos_clasificacion(X, Y, etiquetas, etiquetas_prediccion):  
    # Crear el clasificador
    knn_classifier = KNeighborsClassifier(n_neighbors=8)

    # Entrenamiento
    knn_classifier.fit(X, etiquetas)

    # Realizar predicciones en el conjunto de prueba
    predicciones = knn_classifier.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con K vecinos")
    for i, prediction in enumerate(predicciones):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predicciones) 




#### LOGISTIC REGRESSION ####

# Función del Logistic Regression
def logistic_regression_clasificacion(X, Y, etiquetas, etiquetas_prediccion): 
    # Crear y entrenar el modelo
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X, etiquetas)

    # Realizar predicciones en el conjunto de prueba
    predictions_lr = model_lr.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Logistic Regression:")
    for i, prediction in enumerate(predictions_lr):
        print(f"Documento {i+1}:", f"{prediction}")

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion, predictions_lr) 




#### ENSEMBLE ####

# Función del Ensemble
def ensemble_clasificacion(X, Y, etiquetas, etiquetas_prediccion, proceso_vectorizacion): 
    etiquetas_prediccion_modificado = etiquetas_prediccion[:]      # Incertidumbre


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


    for i in enumerate(Y):
        predictions_svm.append(None)
        predictions_nb.append(None)
        predictions_decision_tree.append(None)
        predictions_rf.append(None)
        predictions_knn.append(None)
        predictions_lr.append(None)


    if proceso_vectorizacion == "bow":       
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_nb.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_classifier.fit(X, etiquetas)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_nb = model_nb.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_classifier.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "tfidf":
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_nb.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_classifier.fit(X, etiquetas)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_nb = model_nb.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_classifier.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "w2v_sg":
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_classifier.fit(X, etiquetas)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_classifier.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "w2v_cbow": 
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_classifier.fit(X, etiquetas)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_classifier.predict(Y)
        predictions_lr = model_lr.predict(Y)


    predicciones = []   # Predicciones finales de cada usuario


    # Voto de la mayoría
    pos = 0      # Corrección posición 
    for i, prediction_svm, prediction_nb, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr in zip(range(len(predictions_svm)), predictions_svm, predictions_nb, predictions_decision_tree, predictions_rf, predictions_knn, predictions_lr):
        predictions = [prediction_svm, prediction_nb, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr]
        
        prediction = obtener_prediccion_mayoritaria(predictions)
        print(f"Documento {i+1}:", f"{prediction}")

        if prediction == "1":
            predicciones.append("1")
        elif prediction == "0":
            predicciones.append("0")
        elif prediction == None:
            etiquetas_prediccion_modificado.pop(pos)
            pos -= 1
  
        pos += 1

        
    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_clasificacion(etiquetas_prediccion_modificado, predicciones) 




#### ALGORITMOS DE CLASIFICACIÓN ####

print("CLASIFICACIÓN - TIPO A")
print()


etiquetas_prediccion_A = ["1", "0", "1", "1", "1", "1", "1", "0", "0", "0", "0"]


# SVM
print("SVM - BoW:")
algoritmo_svm_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("SVM - TF-IDF:")
algoritmo_svm_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)
print("SVM - Skip-gram:")
algoritmo_svm_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A)
print("SVM - CBOW:")
algoritmo_svm_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A)


# Naive Bayes
print("Naive Bayes - BoW:")
algoritmo_nv(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("Naive Bayes - TF-IDF:")
algoritmo_nv(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)


# Árboles de decisión
print("Árboles de Decisión - BoW:")
algoritmo_arboles_decision_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("Árboles de Decisión - TF-IDF:")
algoritmo_arboles_decision_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)
print("Árboles de Decisión - Word2vec - Skip-gram:")
algoritmo_arboles_decision_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A)
print("Árboles de Decisión - Word2vec - CBOW:")
algoritmo_arboles_decision_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A)


# Random Forest
print("Random Forest - BoW:")
algoritmo_rf_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("Random Forest - TF-IDF:")
algoritmo_rf_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)
print("Random Forest - Word2vec - Skip-gram:")
algoritmo_rf_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A)
print("Random Forest - Word2vec - CBOW:")
algoritmo_rf_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A)


# K-vecinos
print("K-vecinos - BoW:")
algoritmo_k_vecinos_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("K-vecinos - TF-IDF:")
algoritmo_k_vecinos_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)
print("K-vecinos - Word2vec - Skip-gram:")
algoritmo_k_vecinos_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A)
print("K-vecinos - Word2vec - CBOW:")
algoritmo_k_vecinos_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A)


# Logistic Regression
print("Logistic Regression - BoW:")
logistic_regression_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A)
print("Logistic Regression - TF-IDF:")
logistic_regression_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A)
print("Logistic Regression - Word2vec - Skip-gram:")
logistic_regression_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A)
print("Logistic Regression - Word2vec - CBOW:")
logistic_regression_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A)


# Ensemble
print("Ensemble - BoW:")
ensemble_clasificacion(X_BOW, Y_BOW, listado_etiquetas_A, etiquetas_prediccion_A, "bow")
print("Ensemble - TF-IDF:")
ensemble_clasificacion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_A, etiquetas_prediccion_A, "tfidf")
print("Ensemble - Word2vec - Skip-gram:")
ensemble_clasificacion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_A, etiquetas_prediccion_A, "w2v_sg")
print("Ensemble - Word2vec - CBOW:")
ensemble_clasificacion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_A, etiquetas_prediccion_A, "w2v_cbow")




#### REGRESIÓN - TIPO B ####




#### EVALUACIÓN DE LOS ALGORITMOS ####

# Función para evaluación del modelo
def metricas_algoritmos_regresion(etiquetas_prediccion, predictiones):
    r2 = r2_score(etiquetas_prediccion, predictiones)
    mse = mean_squared_error(etiquetas_prediccion, predictiones)
    mae = mean_absolute_error(etiquetas_prediccion, predictiones)

    print("R²:", r2)
    print("Error Cuadrático Medio - MSE", mse)
    print("Error Absoluto Medio - MAE:", mae)
    print()




#### SVM  ####

# Función del algoritmo SVM
def algoritmo_svm_regresion(X, Y, etiquetas, etiquetas_prediccion):    
    # Creación modelo
    model_svm = svm.SVR(kernel='linear')

    # Entrenamiento modelo
    model_svm.fit(X, etiquetas)

    # Predicción
    predictions = model_svm.predict(Y)

    # Mostrar predicción
    # print("Predicciones con SVM:")
    for i, prediction in enumerate(predictions):
        print(f"Documento {i+1}:", f"{prediction}")

    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predictions)


    

#### ÁRBOLES DE DECISIÓN ####

# Función del algoritmo Árboles de decisión
def algoritmo_arboles_decision_regresion(X, Y, etiquetas, etiquetas_prediccion):
    # Crear un modelo de Árbol de Decisión
    model_decision_tree = tree.DecisionTreeRegressor()

    # Entrenar el modelo
    model_decision_tree.fit(X, etiquetas)

    # Realizar predicciones
    predictions_decision_tree = model_decision_tree.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Árboles de Decisión:")
    for i, prediction in enumerate(predictions_decision_tree):
        print(f"Documento {i+1}:", f"{prediction}")

    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predictions_decision_tree)




#### RANDOM FOREST ####

# Función del algoritmo Random Forest
def algoritmo_rf_regresion(X, Y, etiquetas, etiquetas_prediccion):
    # Creación del modelo Random Forest
    model_rf = RandomForestRegressor(n_estimators=200, random_state=42)

    # Entrenar modelo
    model_rf.fit(X, etiquetas)

    # Realizar predicciones en el conjunto de prueba
    predictions_rf = model_rf.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Random Forest:")
    for i, prediction in enumerate(predictions_rf):
        print(f"Documento {i+1}:", f"{prediction}")

    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predictions_rf)




#### K-VECINOS ####

# Función del algoritmo K-vecinos
def algoritmo_k_vecinos_regresion(X, Y, etiquetas, etiquetas_prediccion):  
    # Crear el regresor
    knn_regressor  = KNeighborsRegressor(n_neighbors=8)
    
    # Entrenar el modelo
    etiquetas_numeros = np.array(etiquetas, dtype=float)

    knn_regressor.fit(X, etiquetas_numeros)

    # Realizar predicciones en el conjunto de prueba
    predicciones = knn_regressor.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con K vecinos")
    for i, prediction in enumerate(predicciones):
        print(f"Documento {i+1}:", f"{prediction}")

    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predicciones)




#### LOGISTIC REGRESSION ####

# Función del Logistic Regression
def logistic_regression_regresion(X, Y, etiquetas, etiquetas_prediccion): 
    # Crear el modelo
    model_lr = LogisticRegression(max_iter=1000)

    # Entrenar el modelo
    model_lr.fit(X, etiquetas)

    # Realizar predicciones en el conjunto de prueba
    predictions_lr = model_lr.predict(Y)

    # Mostrar predicciones
    # print("Predicciones con Logistic Regression:")
    for i, prediction in enumerate(predictions_lr):
        print(f"Documento {i+1}:", f"{prediction}")

    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Convertir las predicciones a valores numéricos
    predictions_lr_numeric = [float(prediction) for prediction in predictions_lr]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predictions_lr_numeric)




#### ENSEMBLE ####

# Función del Ensemble
def ensemble_regresion(X, Y, etiquetas, etiquetas_prediccion, proceso_vectorizacion): 
    etiquetas_prediccion_modificado = etiquetas_prediccion[:]      # Incertidumbre


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


    # Entrenar el modelo
    etiquetas_numeros = np.array(etiquetas, dtype=float)

    if proceso_vectorizacion == "bow":       
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_regressor.fit(X, etiquetas_numeros)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_regressor.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "tfidf":
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_regressor.fit(X, etiquetas_numeros)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_regressor.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "w2v_sg":
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_regressor.fit(X, etiquetas_numeros)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_regressor.predict(Y)
        predictions_lr = model_lr.predict(Y)

    elif proceso_vectorizacion == "w2v_cbow": 
        # Entrenamiento del modelo
        model_svm.fit(X, etiquetas)
        model_decision_tree.fit(X, etiquetas)
        model_rf.fit(X, etiquetas)
        knn_regressor.fit(X, etiquetas_numeros)
        model_lr.fit(X, etiquetas)

        # Predicción
        predictions_svm = model_svm.predict(Y)
        predictions_decision_tree = model_decision_tree.predict(Y)
        predictions_rf = model_rf.predict(Y)
        predictions_knn = knn_regressor.predict(Y)
        predictions_lr = model_lr.predict(Y)


    predicciones = []   # Predicciones finales de cada usuario


    # Voto de la mayoría
    for i, prediction_svm, prediction_decision_tree, prediction_rf, prediction_knn, prediction_lr in zip(range(len(predictions_svm)), predictions_svm, predictions_decision_tree, predictions_rf, predictions_knn, predictions_lr):
        predictions = [prediction_svm, prediction_decision_tree, prediction_rf, prediction_knn, float(prediction_lr)]
        
        prediction = obtener_media(predictions)
        print(f"Documento {i+1}:", f"{prediction}")
        
        predicciones.append(prediction)   


    # Convertir las etiquetas de predicción a valores numéricos
    etiquetas_prediccion_numeric = [float(etiqueta) for etiqueta in etiquetas_prediccion]

    # Evaluación del modelo
    print("\nMétricas:")
    metricas_algoritmos_regresion(etiquetas_prediccion_numeric, predicciones)




#### ALGORITMOS DE REGRESIÓN ####


print("REGRESIÓN - TIPO B")
print()


etiquetas_prediccion_B = ["0.5", "0.0", "1.0", "0.9", "0.8", "0.7", "0.6", "0.2", "0.1", "0.4", "0.3"]


# SVM
print("SVM - BoW:")
algoritmo_svm_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B)
print("SVM - TF-IDF:")
algoritmo_svm_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B)
print("SVM - Word2vec - Skip-gram:")
algoritmo_svm_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B)
print("SVM - Word2vec - CBOW:")
algoritmo_svm_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B)


# Árboles de decisión
print("Árboles de Decisión - BoW:")
algoritmo_arboles_decision_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B)
print("Árboles de Decisión - TF-IDF:")
algoritmo_arboles_decision_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B)
print("Árboles de Decisión - Word2vec - Skip-gram:")
algoritmo_arboles_decision_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B)
print("Árboles de Decisión - Word2vec - CBOW:")
algoritmo_arboles_decision_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B)


# Random Forest
print("Random Forest - BoW:")
algoritmo_rf_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B)
print("Random Forest - TF-IDF:")
algoritmo_rf_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B)
print("Random Forest - Word2vec - Skip-gram:")
algoritmo_rf_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B)
print("Random Forest - Word2vec - CBOW:")
algoritmo_rf_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B)


# K-vecinos
print("K-vecinos - BoW:")
algoritmo_k_vecinos_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B)
print("K-vecinos - TF-IDF:")
algoritmo_k_vecinos_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B)
print("K-vecinos - Word2vec - Skip-gram:")
algoritmo_k_vecinos_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B)
print("K-vecinos - Word2vec - CBOW:")
algoritmo_k_vecinos_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B)


# Logistic Regression
print("Logistic Regression - BoW:")
logistic_regression_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B)
print("Logistic Regression - TF-IDF:")
logistic_regression_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B)
print("Logistic Regression - Word2vec - Skip-gram:")
logistic_regression_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B)
print("Logistic Regression - Word2Vec - CBOW:")
logistic_regression_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B)


# Ensemble
print("Ensemble - BoW:")
ensemble_regresion(X_BOW, Y_BOW, listado_etiquetas_B, etiquetas_prediccion_B, "bow")
print("Ensemble - TF-IDF:")
ensemble_regresion(X_TF_IDF, Y_TF_IDF, listado_etiquetas_B, etiquetas_prediccion_B, "tfidf")
print("Ensemble - Word2vec - Skip-gram:")
ensemble_regresion(X_W2V_SG, Y_W2V_SG, listado_etiquetas_B, etiquetas_prediccion_B, "w2v_sg")
print("Ensemble - Word2vec - CBOW:")
ensemble_regresion(X_W2V_CBOW, Y_W2V_CBOW, listado_etiquetas_B, etiquetas_prediccion_B, "w2v_cbow")




#### ALMACENAMIENTO RESULTADOS ####

# Combinar la ruta del directorio y el nombre del archivo
ruta_completa = directorio + '\\' + resultados

# Al final del código, guarda el contenido de la salida capturada en un archivo en la ruta específica
with open(ruta_completa, 'w', encoding='utf-8') as f:
    f.write(sys.stdout.getvalue())


# Restaurar sys.stdout
sys.stdout = old_stdout


print(f"El contenido ha sido guardado en {resultados} en la ruta especificada")
