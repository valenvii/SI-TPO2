import csv
import random

def separar_wine_dataset(archivo_origen, train_path, test_path, porcentaje_train=0.8):
    # 1. Leer el archivo original (sin asumir filas de encabezado)
    with open(archivo_origen, 'r') as f:
        lector = csv.reader(f)
        datos = [fila for fila in lector if fila]
    
    # 2. Agrupar las filas por su clase (la clase está en la columna índice 0)
    datos_por_clase = {}
    for fila in datos:
        clase = fila[0] 
        if clase not in datos_por_clase:
            datos_por_clase[clase] = []
        datos_por_clase[clase].append(fila)
    
    list_train = []
    list_test = []
    
    # 3. Procesar cada clase por separado (Muestreo Estratificado)
    for clase, filas in datos_por_clase.items():
        random.shuffle(filas) # Mezclamos los vinos de esta clase
        
        # Calcular el corte del 80%
        corte = int(len(filas) * porcentaje_train)

        list_train.extend(filas[:corte])
        list_test.extend(filas[corte:])
    
    # Mezclar los conjuntos finales para que no queden ordenados por clase
    random.shuffle(list_train)
    random.shuffle(list_test)
    
    # 4. Guardar los archivos resultantes de forma automática
    with open(train_path, 'w', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerows(list_train)
        
    with open(test_path, 'w', newline='') as f:
        escritor = csv.writer(f)
        escritor.writerows(list_test)
    
    print(f"¡Completado con éxito!")
    print(f"Instancias guardadas en Entrenamiento ('{train_path}'): {len(list_train)}")
    print(f"Instancias guardadas en Testeo ('{test_path}'): {len(list_test)}")

# Ejecución directa apuntando a tu archivo
separar_wine_dataset('wine.data', 'wine_train.csv', 'wine_test.csv')