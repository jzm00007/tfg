# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Preprocesamiento de datos - Limpieza del Texto
#   - Transformar emoticonos a texto 
#   - Corrección palabras con letras repetidas consecutivas
#   - Eliminación de elementos o caracteres especiales 
#       - Eliminación de enlaces - URLs
#       - Eliminación de saltos de línea
#       - Eliminación de correos electrónicos - Emails
#       - Eliminación de menciones
#       - Eliminación de signos de puntuación



import os                           # Importa módulo 'os' - Acceder a funcionalidades del Sistema Operativo
import re                           # Importa módulo 're' - Manejo expresiones regulares
import emoji                        # Importa biblioteca 'emoji' - Convertir emojis gráficos en texto descriptivo
from googletrans import Translator  # Importa biblioteca 'Translator' - Traducción de texto -  Google Translate
import json                         # Importa biblioteca para formato JSON
import string                       # Importa módulo 'string' - Manipular cadena de caracteres



# Definir parámetros necesarios                
storage_dir = r"C:\Ficheros sobre Ansiedad\Archivos extraidos Telegram\Preprocesamiento del texto"     # Obtener directorio de almacenamiento
nombre_fichero = "subject1.json"       # Nombre del archivo




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




#### TRANSFORMAR LOS EMOJIS A TEXTO #### 


# Crea un objeto Translator
translator = Translator()


# Convertir palabras que representan emojis en un texto equivalente descriptivo en español
def convertir_texto(texto):
    # Encontrar las palabras que comienzan y terminan con ':' exactamente
    palabras = re.findall(r':[a-zA-Z_-]+:', texto)  # Busca emojis en el formato ":emoji:" usando expresiones regulares

    # Mostrar las palabras encontradas
    for palabra in palabras:
        # Elimina los ':' de los extremos de la palabra
        palabra_modificada_sin_dos_puntos = palabra.strip(':')

        # Reemplaza guiones bajos con espacios en blanco
        palabra_modificada_sin_guiones_bajos = palabra_modificada_sin_dos_puntos.replace("_", " ")

        # Reemplaza guiones con espacios en blanco
        palabra_modificada_sin_guiones = palabra_modificada_sin_guiones_bajos.replace("-", " ")

        # Traduce la palabra al español
        palabra_esp = translator.translate(palabra_modificada_sin_guiones, src='en', dest='es')

        # Obtiene la traducción y agrega un espacio al final (evitar textos descriptivos de emoticonos sin espacios)
        palabra_final = " " + palabra_esp.text + " "

        # Reemplaza la palabra original por su traducción en el texto
        texto = texto.replace(palabra, palabra_final)

    return texto


# Convertir emojis gráficos en un texto equivalente descriptivo en español
def transformar_emojis_a_texto(archivo):
    # Leer el archivo
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()

    # Convertir emojis gráficos a texto descriptivo en español
    contenido_modificado = emoji.demojize(contenido)              # Convierte emojis en texto descriptivo
    contenido_modificado = convertir_texto(contenido_modificado)  # Utiliza la función para traducir emojis a español

    # Sobreescribir el archivo con los cambios
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write(contenido_modificado)  # Escribe el contenido modificado en el archivo




#### CORRECCIÓN PALABRAS CON LETRAS REPETIDAS CONSECUTIVAS #### 
        
# ACLARACIÓN: La corrección de las palabras con letras repetidas consecutivas se realiza así:
#   - Letras repetidas en la primera o última posición de la palabra - Reducción a una letra
#   - Letras repetidas en el interior de la palabra - Reducción a dos letras - Motivos:
#       - Posibilidad añadir comprobación para 'll' y 'rr' - Innecesaria con esta decisión
#       - Existencia de palabras con difícil comprobación - Ej: leer, innovar...


# Función auxiliar para corregir palabras con letras repetidas consecutivas
def corregir_palabra(palabra):
    nueva_palabra = ""  # Palabra corregida
    i = 0   # Variable de iteración para recorrer la palabra

    while i < len(palabra):
        # Si carácter actual es letra del alfabeto
        if palabra[i].isalpha():
            nueva_palabra += palabra[i]  # Añadir letra actual
            # Si la letra actual es igual a la siguiente, hay una repetición
            if i < len(palabra) - 1 and palabra[i] == palabra[i + 1]:
                i += 1  
                # Si hay más de dos letras consecutivas iguales, solo se añade dos
                while i < len(palabra) - 1 and palabra[i] == palabra[i + 1]:
                    i += 1
                nueva_palabra += palabra[i]  # Añadir segunda letra repetida
        else:
            nueva_palabra += palabra[i]  # Agregar caracter no alfabético
        i += 1  
    
    # Casos de URLs
    if palabra.startswith('www'):
        nueva_palabra = 'w'+ nueva_palabra

    # Eliminar letras repetidas al inicio, excepto si palabra comienza por 'll' ni 'www'
    if not palabra.startswith('ll') and not palabra.startswith('www'):
        while len(nueva_palabra) >= 2 and nueva_palabra[0] == nueva_palabra[1]:
            nueva_palabra = nueva_palabra[1:]   # Eliminar primera letra

    # Eliminar letras repetidas al final
    while len(nueva_palabra) >= 2 and nueva_palabra[-1] == nueva_palabra[-2]:
        nueva_palabra = nueva_palabra[:-1]      # Eliminar última letra
    
    return nueva_palabra


# Función auxiliar para corregir mensaje JSON
def corregir_mensaje(mensaje):
    mensaje_corregido = []  # Mensaje JSON corregido

    for item in mensaje:
        frase_original = item["message"]        # Guarda texto original antes de corregirla

        palabras = item["message"].split()      # Dividir la frase en palabras individuales
        nuevo_texto = ""                        # Variable para almacenar palabras corregidas

        for palabra in palabras:
            palabra_corregida = corregir_palabra(palabra)   # Corrige palabra
            nuevo_texto += palabra_corregida + " "          # Añadir cada palabra seguida de un espacio a la nueva frase

        nuevo_texto.strip()  # Eliminar espacio adicional al final

        mensaje_corregido.append({
            "id_message": item["id_message"],
            #"original_word": frase_original,     # Texto original
            "message": nuevo_texto,               # Texto corregido
            "date": item["date"]
        })
    
    return mensaje_corregido


# Función principal para corregir fichero JSON
def corregir_fichero_ruta(ruta):
    with open(ruta, 'r', encoding='utf-8') as file:
        contenido = json.load(file)

    contenido_corregido = corregir_mensaje(contenido)

    # Guarda contenido corregido en archivo
    # nuevo_nombre = ruta.split('.')[0] + "_corregido.txt"
    with open(ruta, 'w', encoding='utf-8') as file:
        json.dump(contenido_corregido, file, indent=4, ensure_ascii=False)

    #print(f"Archivo corregido guardado como '{nuevo_nombre}'")




#### ELIMINACIÓN DE ELEMENTOS O CARACTERES ESPECIALES ####    



## ELIMINACIÓN DE ENLACES - URLs ##
    

# Función eliminar URLs del texto
def eliminar_enlaces_texto(ruta_archivo):
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.read()

    # Buscar y eliminar enlaces usando expresiones regulares
    contenido_modificado = re.sub(r'(https?://|www\.)\S+?(?=\s|")', '', contenido)

    # Escribir el contenido modificado de vuelta al archivo
    with open(ruta_archivo, 'w') as archivo_modificado:
        archivo_modificado.write(contenido_modificado)



## ELIMINACIÓN DE SALTOS DE LINEA ##
    

# Función eliminar saltos de linea del texto
def eliminar_saltos_linea_texto(ruta_archivo):
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.read()

    # Sustituir saltos de línea por una cadena vacía
    contenido_modificado = contenido.replace('\\n', '')

    # Escribir el contenido modificado de vuelta al archivo
    with open(ruta_archivo, 'w') as archivo_modificado:
        archivo_modificado.write(contenido_modificado)



## ELIMINACIÓN DE CORREO ELECTRÓNICO - EMAILs ##
        

# Función eliminar correos electrónicos del texto
def eliminar_emails_texto(ruta_archivo):
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.read()

    # Buscar y eliminar direcciones de correo electrónico usando expresiones regulares
    contenido_modificado = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', contenido)

    # Escribir el contenido modificado de vuelta al archivo
    with open(ruta_archivo, 'w') as archivo_modificado:
        archivo_modificado.write(contenido_modificado)
    



## ELIMINACIÓN DE MENCIONES A USUARIOS ##


# Función eliminar menciones del texto
def eliminar_menciones_texto(ruta_archivo):
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.read()

    # Buscar y eliminar menciones usando expresiones regulares
    contenido_modificado = re.sub(r'@ ?\S+?(?=\s|")', '', contenido)

    # Escribir el contenido modificado de vuelta al archivo
    with open(ruta_archivo, 'w') as archivo_modificado:
        archivo_modificado.write(contenido_modificado)





## ELIMINACIÓN DE SIGNOS DE PUNTUACIÓN ##


# Función auxiliar para eliminar signos de puntuación
def eliminar_puntuacion(texto):
    # Generar una tabla de traducción para eliminar signos de puntuación
    translator = str.maketrans('', '', string.punctuation + '¿¡')

    # Aplicar la traducción al texto
    texto_sin_puntuacion = texto.translate(translator)
    
    return texto_sin_puntuacion


# Función eliminar signos de puntuación del texto
def eliminar_puntuacion_texto(ruta_archivo):
    # Abrir el archivo 
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        datos = json.load(file)

    # Iterar sobre cada mensaje y eliminar la puntuación del campo 'message'
    for mensaje in datos:
        mensaje['message'] = eliminar_puntuacion(mensaje['message'])

    # Guardar los datos modificados en archivo 
    with open(ruta_archivo, 'w', encoding='utf-8') as file:
        json.dump(datos, file, indent=4, ensure_ascii=False)




#### LIMPIEZA DEL TEXTO ####    


## OPCIÓN 1 - FICHERO ÚNICO ##
transformar_emojis_a_texto(f"{storage_dir}/{nombre_fichero}")      # Llama a la función para transformar emojis en el archivo especificado
corregir_fichero_ruta(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para corregir el archivo
eliminar_enlaces_texto(f"{storage_dir}/{nombre_fichero}")          # Llama a la función para eliminar URLs
eliminar_saltos_linea_texto(f"{storage_dir}/{nombre_fichero}")     # Llama a la función para eliminar saltos de linea
eliminar_emails_texto(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para eliminar emails
eliminar_menciones_texto(f"{storage_dir}/{nombre_fichero}")        # Llama a la función para eliminar menciones
eliminar_puntuacion_texto(f"{storage_dir}/{nombre_fichero}")       # Llama a la función para eliminar signos de puntuación


## OPCIÓN 2 - CONJUNTO DE FICHEROS ##
#for nombre_fichero in lista_ficheros:   # Cada fichero extraído
#    transformar_emojis_a_texto(f"{storage_dir}/{nombre_fichero}")      # Llama a la función para transformar emojis en el archivo especificado
#    corregir_fichero_ruta(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para corregir el archivo
#    eliminar_enlaces_texto(f"{storage_dir}/{nombre_fichero}")          # Llama a la función para eliminar URLs
#    eliminar_saltos_linea_texto(f"{storage_dir}/{nombre_fichero}")     # Llama a la función para eliminar saltos de linea
#    eliminar_emails_texto(f"{storage_dir}/{nombre_fichero}")           # Llama a la función para eliminar emails
#    eliminar_menciones_texto(f"{storage_dir}/{nombre_fichero}")        # Llama a la función para eliminar menciones
#    eliminar_puntuacion_texto(f"{storage_dir}/{nombre_fichero}")       # Llama a la función para eliminar signos de puntuación