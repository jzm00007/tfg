# Trabajo de Fin de Grado - Detección Ansiedad en Redes Sociales
# Jesús Zafra Moreno
# Extracción datos de Telegram


# Primera ejecución - Solicita Número de Teléfono (+34...) + Código comprobación


import asyncio                              # Importa módulo 'asyncio' - Trabajar con operaciones asíncronas y temporizadores
from telethon.sync import TelegramClient    # Importa clase 'TelegramClient' desde biblioteca 'Telethon' - Cliente Telegram
from pytz import timezone                   # Importa función 'timezone' - Manejo zonas horarias
from datetime import datetime               # Importa 'datetime' - Trabajar con fechas y horas
import json                                 # Importa 'json' - Manejar datos JSON
import os                                   # Importa módulo 'os' - Acceder a funcionalidades del Sistema Operativo



# Definir parámetros necesarios
api_id = '20572041'                                 # Credenciales de la API de Telegram - ID aplicación
api_hash = 'a02404a822ebb50759f014e26b2d69e1'       # Credenciales de la API de Telegram - Clave secreta de la aplicación
group_username = 'enluchaconstante'                 # Nombre grupo - 'Aprendiendo a vivir con la ansiedad'     
      
entity_name = "Fatima"         # Nombre usuario a buscar                        
storage_dir = r"C:\Ficheros sobre Ansiedad\Archivos extraidos Telegram"     # Obtener directorio de almacenamiento
MAX_TEXT_MESSAGES = 50         # Definir número máximo mensajes de texto a mostrar



# Obtener ID grupo
async def get_group_id(api_id, api_hash):
    # Crea un cliente Telegram
    async with TelegramClient('session_name', api_id, api_hash) as client:
        entity = await client.get_entity(group_username)    # Obtiene entidad del grupo
        return entity.id
    

# Obtener ID usuario - Elegir Nombre del Chat
async def get_entity_id(group_id, api_id, api_hash, name):
    # Crea un cliente Telegram
    async with TelegramClient('session_name', api_id, api_hash) as client:
        participants = await client.get_participants(group_id)  # Obtiene la lista de participantes del grupo
        for participant in participants:
            # Construcción nombre completo participante
            full_name = participant.first_name + " " + participant.last_name if participant.last_name else participant.first_name
            if full_name.strip() == name:  # Compara el nombre completo con el nombre buscado
                return participant.id  # ID del usuario
            elif participant.username and participant.username.lower() == name.lower():  # Compara nombre usuario con nombre buscado
                return participant.id   
        return None  # Si ningún usuario con dicho nombre   
    

# Opción Extra - Obtener ID usuario a partir listado IDs usuarios del grupo
async def get_list_id(group_id, api_id, api_hash, name):
    participants = [] # Listado participantes grupo

    # Crea un cliente Telegram
    async with TelegramClient('session_name', api_id, api_hash) as client:
        participants = await client.get_participants(group_id)  # Obtiene lista de participantes del grupo

    return participants



# Función principal
async def main():
    global group_id         # ID grupo
    global entity_id        # ID usuario

    group_id = await get_group_id(api_id, api_hash)     # Obtener ID grupo

    print(f"ID del grupo '{group_username}': {group_id}")

    entity_id = await get_entity_id(group_id, api_id, api_hash, entity_name)    # Obtener ID usuario   

    # Opción extra - Obtener ID usuario
    participants = await get_list_id(group_id, api_id, api_hash, entity_name)   # Obtener IDs usuarios grupo

    # for participant in participants:
    #     print(f"{participant.id} - {participant.first_name} {participant.last_name }")

    # entity_id = int(input("Introduce el ID deseado: "))  # Almacena ID deseado proporcionado por el usuario
    # print(f"ID elegido: {entity_id}")  # Mostrar ID elegido

    if entity_id:
        print("El ID de", entity_name, "es:", entity_id)  
    else:
        print("No se encontró ningún usuario con el nombre", entity_name) 

    async with TelegramClient('session_name', api_id, api_hash) as client:  # Iniciar sesión con cliente Telegram
            processed_message_ids = set()   # Conjunto para almacenar IDs de los mensajes procesados
            total_text_message_count = 0    # Contador para número total de mensajes de texto obtenidos
            last_message_id = None          # Almacena ID último mensaje procesado

            messages_json = []    # Lista para almacenar mensajes en formato JSON

            # Función para obtener mensajes más recientes grupo
            async def get_latest_messages():
                nonlocal last_message_id  # Variable externa modificable
                if last_message_id:
                    # Obtiene mensajes posteriores al último mensaje procesado
                    messages = await client.get_messages(group_id, limit=MAX_TEXT_MESSAGES, offset_id=last_message_id)
                else:
                    # Obtiene últimos mensajes sin desplazamiento
                    messages = await client.get_messages(group_id, limit=MAX_TEXT_MESSAGES)
                return messages

            # Función para convertir mensaje a formato JSON
            def message_to_json(message):
                if entity_id == message.sender_id:
                    # Convertir fecha y hora al horario de España
                    spain_tz = timezone('Europe/Madrid')
                    local_time = message.date.astimezone(spain_tz)    # Convierte hora del mensaje al horario de España
                    # Formatear fecha y hora
                    formatted_date = local_time.strftime('%Y-%m-%d %H:%M:%S')
                    # Crear un diccionario con los campos requeridos
                    message_data = {
                        "id_message": message.id,   # ID mensaje
                        "message": message.text,    # Contenido mensaje
                        "date": formatted_date      # Fecha y hora mensaje formateada
                    }
                    return message_data
                else:
                    return {}

            # Obtener y procesar los mensajes de texto
            start_time = datetime.now()   # Guardar tiempo de inicio del proceso
            while total_text_message_count < MAX_TEXT_MESSAGES:
                current_time = datetime.now()  # Obtener tiempo actual
                time_elapsed = (current_time - start_time).total_seconds()  # Calcular el tiempo transcurrido en segundos
                if time_elapsed >= 300:  # Comprobar si ha pasado más de 5 minutos (300 segundos)
                    print("Se ha excedido el tiempo de espera. Mostrando los mensajes obtenidos hasta ahora.")
                    break  # Salir del bucle si excede el tiempo de espera
                messages = await get_latest_messages()  # Obtener los mensajes más recientes
                for message in messages:
                    if message.text and message.id not in processed_message_ids:
                        message_json = message_to_json(message)     # Convierte mensaje a formato JSON
                        if message_json:    # Verifica si mensaje convertido a JSON no está vacío
                            messages_json.append(message_json)      # Añade mensaje JSON a la lista
                            processed_message_ids.add(message.id)   # Añade ID del mensaje procesado al conjunto
                            total_text_message_count += 1           # Incrementa contador de mensajes de texto
                        if total_text_message_count >= MAX_TEXT_MESSAGES:   # Comprueba si suficientes mensajes procesados
                            break   # Salir del bucle si suficientes mensajes procesados
                    last_message_id = message.id    # Actualiza ID último mensaje procesado
                if total_text_message_count >= MAX_TEXT_MESSAGES:   # Comprueba si suficientes mensajes procesados
                    break   # Salir del bucle si suficientes mensajes procesados
                await asyncio.sleep(1)  # Esperar 1 segundo antes siguiente iteración

            # Mostrar JSON resultante con codificación UTF-8
            # print(json.dumps(messages_json, indent=4, ensure_ascii=False).encode('utf-8').decode())

            nombre_fichero = str(entity_id) + ".json"    # Nombre fichero - ID usuario a buscar

            # Si el directorio no existe, créalo
            if not os.path.exists(storage_dir):
                os.makedirs(storage_dir)

            # Escritura del archivo en el disco
            file_path = os.path.join(storage_dir, f"{nombre_fichero}")

            print(f"Fichero JSON '{nombre_fichero}' escrito en el directorio")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages_json, f, indent=4, ensure_ascii=False)



# Ejecutar el bucle de eventos de asyncio
asyncio.run(main())