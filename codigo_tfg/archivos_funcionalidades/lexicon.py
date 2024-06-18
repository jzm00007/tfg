# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Lexicon de palabras sobre ansiedad



# Importar módulos necesarios
import os                           # Proporciona funciones para interactuar con el Sistema Operativo
import json                         # Proporciona funciones para trabajar con JSON
import spacy                        # Biblioteca para Procesamiento de Lenguaje Natural
from collections import Counter     # Estructura de datos para contar objetos - Frecuencia
import sys      # Acceso a algunas variables y funciones que interactúan con el intérprete de Python
import io       # Proporciona las herramientas necesarias para trabajar con flujos de entrada y salida en Python

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score     # Accurancy, Precision, Recall, F1
from sklearn.metrics import confusion_matrix                                            # Matriz de confusión



# Modelo en español - Cantidad datos y capacidad vectorización más grande
# spacy.cli.download("es_core_news_lg")   
import es_core_news_lg    
nlp = es_core_news_lg.load()



# Definir parámetros necesarios      
directorio = 'C:\\Ficheros sobre Ansiedad'          # Directorio principal          
directorio_entrenamiento = r"C:\Ficheros sobre Ansiedad\Archivos entrenamiento lexicon"    # Obtener directorio de ficheros entrenamiento
directorio_test = r"C:\Ficheros sobre Ansiedad\Archivos test"                              # Obtener directorio de ficheros test
storage_dir = r"C:\Ficheros sobre Ansiedad"                                                # Obtener directorio de almacenamiento
fichero_listado_palabras_ansiedad = "listado_palabras_ansiedad_jack_moreno.txt"            # Obtener listado de palabras ansiedad (Jack Moreno)
fichero_listado_palabras_negativas = "listado_palabras_negativas_isol.txt"                 # Obtener listado de palabras negativas (ISOL)
resultados = "output_lexicon.txt"                   # Fichero con resultados




# Verificar y crear el directorio si no existe
if not os.path.exists(directorio):
    os.makedirs(directorio)

# Crear un objeto StringIO para capturar la salida
old_stdout = sys.stdout
sys.stdout = io.StringIO()




#### EVALUACIÓN DE RESULTADOS ####

# Función para evaluación de los resultados
def metricas_evaluacion(etiquetas_prediccion, predictiones):
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



#### OBTENER LISTADO DE FICHEROS EN DIRECTORIO DE ENTRENAMIENTO #### 
# - Puntuación entre 0.9 - 1

# Lista para almacenar los nombres de los archivos
lista_ficheros_entrenamiento = []

# Obtener los nombres de los archivos en la carpeta
for nombre_fichero in os.listdir(directorio_entrenamiento):  
    lista_ficheros_entrenamiento.append(nombre_fichero)

# Imprimir los nombres de los archivos
# print(f"Nombres de archivos en el directorio {directorio_entrenamiento}:")
# for nombre in lista_ficheros_entrenamiento:
#    print(nombre)




#### OBTENER LISTADO DE FICHEROS EN DIRECTORIO DE TEST #### 

# Lista para almacenar los nombres de los archivos
lista_ficheros_test = []

# Obtener los nombres de los archivos en la carpeta
for nombre_fichero in os.listdir(directorio_test):   
    lista_ficheros_test.append(nombre_fichero)

# Imprimir los nombres de los archivos
# print(f"Nombres de archivos en el directorio {directorio_test}:")
# for nombre in lista_ficheros_test:
#     print(nombre)




#### OBTENER LISTADO DE PALABRAS SOBRE ANSIEDAD - JACK MORENO #### 

# Ruta completa del archivo
ruta = os.path.join(storage_dir, fichero_listado_palabras_ansiedad)

# Lista para almacenar las palabras sobre ansiedad
listado_palabras_ansiedad = []

# Leer el archivo y almacenar las palabras en la lista
with open(ruta, 'r', encoding='utf-8') as file:
    for line in file:
        listado_palabras_ansiedad.extend(line.strip().split())

# Mostrar listado de palabras con ansiedad
# print(f"El listado de palabras sobre ansiedad son: {listado_palabras_ansiedad}")




#### OBTENER LISTADO DE PALABRAS NEGATIVAS - ISOL #### 

# Ruta completa del archivo
ruta = os.path.join(storage_dir, fichero_listado_palabras_negativas)

# Lista para almacenar las palabras negativas
listado_palabras_negativas = []

# Leer el archivo y almacenar las palabras en la lista
with open(ruta, 'r', encoding='utf-8') as file:
    for line in file:
        listado_palabras_negativas.extend(line.strip().split())

# Mostrar listado de palabras negativas
# print(f"El listado de palabras negativas son: {listado_palabras_negativas}")




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
    for filename in os.listdir(directorio_entrenamiento):
        if filename.endswith(".json"):
            # Obtener la ruta completa del archivo
            file_path = os.path.join(directorio_entrenamiento, filename)

            # Obtener los contadores de palabras por tipo para el archivo actual
            verb_counter, noun_counter, adj_counter, adv_counter = contar_palabras_en_fichero(file_path)

            # Actualizar los contadores totales
            verb_total.update(verb_counter)
            noun_total.update(noun_counter)
            adj_total.update(adj_counter)
            adv_total.update(adv_counter)


    # Mostrar los 15 elementos más comunes en cada contador
    # print("15 verbos más comunes:", verb_total.most_common(15))
    # print("15 sustantivos más comunes:", noun_total.most_common(15))
    # print("15 adjetivos más comunes:", adj_total.most_common(15))
    # print("15 adverbios más comunes:", adv_total.most_common(15))

    # Devolver los contadores finales
    return verb_total, noun_total, adj_total, adv_total




#### CLASIFICACIÓN FICHEROS DE TEST CON CONJUNTO FICHEROS DE ENTRENAMIENTO - LEXICON PROPIO #### 

# Función para verificar si un mensaje contiene al menos una de las palabras más comunes en una lista
def verificar_coincidencias(message, word_list):
    for word, _ in word_list:   
        if word in message:     # Comprobar
            return True
    return False


# Función para clasificación con el lexicón obtenido
def clasificacion_lexicon(directorio_test, lista_ficheros_test, etiquetas_test):
    ficheros_ansiedad = []          # Lista ficheros con ansiedad
    ficheros_no_ansiedad = []       # Lista ficheros sin ansiedad
    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 85

    # Obtener lexicón de palabras
    verb_total, noun_total, adj_total, adv_total = lexicon()

    # Iterar sobre los archivos en el directorio de test
    for filename in lista_ficheros_test:
        if filename.endswith(".json"):
            # Obtener la ruta completa del archivo
            file_path = os.path.join(directorio_test, filename)

            # Abrir el archivo JSON
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Contar coincidencias en cada archivo
            coincidencias_total = 0
            for message in data:
                # Contar palabras por tipo en el mensaje actual
                verbs, nouns, adjectives, adverbs = contar_palabras(message['message'])
                # Verificar coincidencias en cada tipo de palabra
                if verificar_coincidencias(message['message'], verb_total.most_common(15)):
                    coincidencias_total += 1
                if verificar_coincidencias(message['message'], noun_total.most_common(15)):
                    coincidencias_total += 1
                if verificar_coincidencias(message['message'], adj_total.most_common(15)):
                    coincidencias_total += 1
                # if verificar_coincidencias(message['message'], adv_total.most_common(10)):     # Revisar
                #     coincidencias_total += 1

            # Categorizar el archivo
            if coincidencias_total >= umbral_coincidencias:
                ficheros_ansiedad.append(filename)
                etiquetas_ficheros.append("1")
            else:
                ficheros_no_ansiedad.append(filename)
                etiquetas_ficheros.append("0")


    # Mostrar los resultados
    print("\nFicheros de ansiedad:")
    for fichero in ficheros_ansiedad:
        print(f"- {fichero}")

    print("\nFicheros no de ansiedad:")
    for fichero in ficheros_no_ansiedad:
        print(f"- {fichero}")


    # Evaluar resultados
    print("\nMétricas:")
    metricas_evaluacion(etiquetas_test, etiquetas_ficheros)




#### CLASIFICACIÓN FICHEROS DE TEST CON LISTADO DE PALABRAS - LEXICON PROPORCIONADO #### 


## Blog de Jack Moreno

# Función para clasificación con el lexicón proporcionado Jack Moreno
def clasificacion_listado_ansiedad(directorio_test, lista_ficheros_test, listado_palabras_ansiedad, etiquetas_test):
    # Lista para almacenar los nombres de los ficheros clasificados
    ficheros_ansiedad = []
    ficheros_no_ansiedad = []
    
    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 5

    # Convertir la lista de palabras de ansiedad a un conjunto para una búsqueda más eficiente
    palabras_ansiedad = set(listado_palabras_ansiedad)

    # Iterar sobre los ficheros en el directorio
    for fichero in lista_ficheros_test:
        if fichero.endswith(".json"):
            # Ruta completa del fichero
            ruta_fichero = os.path.join(directorio_test, fichero)
            
            # Contador de ocurrencias de palabras de ansiedad
            contador_ansiedad = 0

            # Leer el fichero y contar las ocurrencias de palabras de ansiedad
            with open(ruta_fichero, 'r', encoding='utf-8') as file:
                for line in file:
                    for palabra in line.strip().split():
                        if palabra in palabras_ansiedad:
                            contador_ansiedad += 1
            
            # Clasificar el fichero según el contador
            if contador_ansiedad >= umbral_coincidencias:
                ficheros_ansiedad.append(fichero)
                etiquetas_ficheros.append("1")
            else:
                ficheros_no_ansiedad.append(fichero)
                etiquetas_ficheros.append("0")


    # Mostrar los resultados
    print("\nFicheros de ansiedad:")
    for fichero in ficheros_ansiedad:
        print(f"- {fichero}")

    print("\nFicheros no de ansiedad:")
    for fichero in ficheros_no_ansiedad:
        print(f"- {fichero}")


    # Evaluar resultados
    print("\nMétricas:")
    metricas_evaluacion(etiquetas_test, etiquetas_ficheros)




## Palabras negativas ISOL

# Función para clasificación con el lexicón proporcionado ISOL
def clasificacion_listado_negativas(directorio_test, lista_ficheros_test, listado_palabras_negativas, etiquetas_test):
    # Lista para almacenar los nombres de los ficheros clasificados
    ficheros_ansiedad = []
    ficheros_no_ansiedad = []

    etiquetas_ficheros = []         # Lista ficheros de test

    # Definir el umbral de coincidencias
    umbral_coincidencias = 25

    # Convertir la lista de palabras negativas a un conjunto para una búsqueda más eficiente
    palabras_negativas = set(listado_palabras_negativas)

    # Iterar sobre los ficheros en el directorio
    for fichero in lista_ficheros_test:
        if fichero.endswith(".json"):
            # Ruta completa del fichero
            ruta_fichero = os.path.join(directorio_test, fichero)
            
            # Contador de ocurrencias de palabras negativas
            contador_negativas = 0

            # Leer el fichero y contar las ocurrencias de palabras negativas
            with open(ruta_fichero, 'r', encoding='utf-8') as file:
                for line in file:
                    for palabra in line.strip().split():
                        if palabra in palabras_negativas:
                            contador_negativas += 1
            
            # Clasificar el fichero según el contador
            if contador_negativas >= umbral_coincidencias:
                ficheros_ansiedad.append(fichero)
                etiquetas_ficheros.append("1")
            else:
                ficheros_no_ansiedad.append(fichero)
                etiquetas_ficheros.append("0")


    # Mostrar los resultados
    print("\nFicheros de ansiedad:")
    for fichero in ficheros_ansiedad:
        print(f"- {fichero}")

    print("\nFicheros no de ansiedad:")
    for fichero in ficheros_no_ansiedad:
        print(f"- {fichero}")


    # Evaluar resultados
    print("\nMétricas:")
    metricas_evaluacion(etiquetas_test, etiquetas_ficheros)




#### CLASIFICACIÓN DE TEXTOS CON LEXICONES DE PALABRAS ####


etiquetas_test = ["1", "0", "1", "1", "1", "1", "0", "1", "1", "0", "1"]


# Clasificación con lexicon propio
print("Clasificación con Lexicón Propio")
clasificacion_lexicon(directorio_test, lista_ficheros_test, etiquetas_test)
print()

# Clasificación con lexicón proporcionado - Palabras de ansiedad
print("Clasificación con Lexicón Proporcionado - Palabras de ansiedad")
clasificacion_listado_ansiedad(directorio_test, lista_ficheros_test, listado_palabras_ansiedad, etiquetas_test)
print()


# Clasificación con lexicón proporcionado - Palabras negativas
print("Clasificación con Lexicón Proporcionado - Palabras negativas")
clasificacion_listado_negativas(directorio_test, lista_ficheros_test, listado_palabras_negativas, etiquetas_test)
print()




#### ALMACENAMIENTO RESULTADOS ####

# Combinar la ruta del directorio y el nombre del archivo
ruta_completa = directorio + '\\' + resultados

# Al final del código, guarda el contenido de la salida capturada en un archivo en la ruta específica
with open(ruta_completa, 'w', encoding='utf-8') as f:
    f.write(sys.stdout.getvalue())


# Restaurar sys.stdout
sys.stdout = old_stdout


print(f"El contenido ha sido guardado en {resultados} en la ruta especificada")