import csv
import math

def calcular_limites_rangos(datos_train, indices_atributos, num_rangos=3):
    """Calcula los umbrales para dividir cada atributo numérico en 'num_rangos' partes iguales."""
    limites = {}
    for idx in indices_atributos:
        # Extraer todos los valores numéricos de esa columna y ordenarlos
        valores = sorted([float(fila[idx]) for fila in datos_train])
        n = len(valores)
        
        # Calcular los umbrales dinámicamente
        umbrales = []
        for i in range(1, num_rangos):
            indice = int(n * i / num_rangos)
            umbrales.append(valores[indice])
            
        limites[idx] = umbrales
    return limites

def discretizar_instancia(fila, limites, indices_atributos):
    """Convierte una fila numérica en una fila de categorías enteras ('1', '2', ..., 'N')"""
    fila_discreta = [fila[0]] # Mantener la clase intacta en la primera columna
    
    for idx in indices_atributos:
        valor = float(fila[idx])
        umbrales = limites[idx]
        
        categoria = str(len(umbrales) + 1) 
        # Asignamos nivel "1", "2", etc.
        for i, umbral in enumerate(umbrales):
            if valor <= umbral:
                categoria = str(i + 1) 
                break
                
        fila_discreta.append(categoria)
        
    return fila_discreta

def calcular_entropia(instancias):
    if not instancias:
        return 0
    total = len(instancias)
    conteo_clases = {}
    for fila in instancias:
        clase = fila[0]
        conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
        
    entropia = 0.0
    for conteo in conteo_clases.values():
        p = conteo / total
        entropia -= p * math.log2(p)
    return entropia

def calcular_ganancia_informacion(instancias, indice_atrib):
    """Calcula la ganancia de ID3 tradicional para atributos categóricos (Bajo/Medio/Alto)"""
    entropia_total = calcular_entropia(instancias)
    
    # Agrupar instancias por los valores del atributo ('Bajo', 'Medio', 'Alto')
    grupos = {}
    for fila in instancias:
        val = fila[indice_atrib]
        if val not in grupos:
            grupos[val] = []
        grupos[val].append(fila)
        
    entropia_atributos = 0.0
    total_instancias = len(instancias)
    
    for sub_grupo in grupos.values():
        peso = len(sub_grupo) / total_instancias
        entropia_atributos += peso * calcular_entropia(sub_grupo)
        
    return entropia_total - entropia_atributos, grupos

class NodoID3:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase_predicha=None):
        self.atributo = atributo          # Índice de la característica química
        self.hijos = hijos or {}          # Diccionario: {'Bajo': Nodo, 'Medio': Nodo, 'Alto': Nodo}
        self.es_hoja = es_hoja
        self.clase_predicha = clase_predicha

def construir_arbol_id3(instancias, atributos_disponibles, num_rangos=3):
    clases = [fila[0] for fila in instancias]
    clase_mayoritaria = max(set(clases), key=clases.count)
    
    # Casos base
    if len(set(clases)) == 1:
        return NodoID3(es_hoja=True, clase_predicha=clases[0])
    if not atributos_disponibles:
        return NodoID3(es_hoja=True, clase_predicha=clase_mayoritaria)
        
    mejor_ganancia = -1
    mejor_atrib = None
    mejores_grupos = None
    
    # Seleccionar el atributo con máxima Ganancia de Información
    for atrib in atributos_disponibles:
        ganancia, grupos = calcular_ganancia_informacion(instancias, atrib)
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_atrib = atrib
            mejores_grupos = grupos
            
    if mejor_ganancia <= 0.0001 or mejor_atrib is None:
        return NodoID3(es_hoja=True, clase_predicha=clase_mayoritaria)
        
    # Crear nodo de decisión
    nodo = NodoID3(atributo=mejor_atrib)
    nuevos_atributos = [a for a in atributos_disponibles if a != mejor_atrib]
    
    # ID3 crea una rama por cada categoría posible dinámicamente (1 a num_rangos)
    for nivel in range(1, num_rangos + 1):
        categoria = str(nivel)
        sub_instancias = mejores_grupos.get(categoria, [])
        if not sub_instancias:
            # Si una rama se queda vacía, se le asigna la clase mayoritaria del nodo padre
            nodo.hijos[categoria] = NodoID3(es_hoja=True, clase_predicha=clase_mayoritaria)
        else:
            # Pasamos num_rangos a la llamada recursiva
            nodo.hijos[categoria] = construir_arbol_id3(sub_instancias, nuevos_atributos, num_rangos)
            
    return nodo

def clasificar_instancia(arbol, fila):
    if arbol.es_hoja:
        return arbol.clase_predicha
    valor_categoria = fila[arbol.atributo]
    
    # Por seguridad, si la categoría no existe en el árbol entrenado, va por defecto
    if valor_categoria not in arbol.hijos:
        return "1" 
    return clasificar_instancia(arbol.hijos[valor_categoria], fila)

def cargar_csv(ruta):
    with open(ruta, 'r') as f:
        return [fila for fila in csv.reader(f) if fila]

# 1. Cargar sets particionados numéricos
train_crudo = cargar_csv('wine_train.csv')
test_crudo = cargar_csv('wine_test.csv')
indices_atributos = list(range(1, len(train_crudo[0])))

# Puedes cambiar este valor para probar con 4, 5 o más rangos
CANTIDAD_RANGOS = 5 

# 2. Calcular límites basados EN EL ENTRENAMIENTO y discretizar
limites_rangos = calcular_limites_rangos(train_crudo, indices_atributos, CANTIDAD_RANGOS)

train_discreto = [discretizar_instancia(f, limites_rangos, indices_atributos) for f in train_crudo]
test_discreto = [discretizar_instancia(f, limites_rangos, indices_atributos) for f in test_crudo]

print(f"--- Entrenando Modelo ID3 Puro (Datos Discretizados en {CANTIDAD_RANGOS} niveles) ---")
arbol_id3 = construir_arbol_id3(train_discreto, indices_atributos, CANTIDAD_RANGOS)
print("¡Modelo Categórico Generado con éxito!\n")

# 3. Evaluar
clases_reales = []
clases_predichas = []

for fila in test_discreto:
    clases_reales.append(fila[0])
    clases_predichas.append(clasificar_instancia(arbol_id3, fila))

# 4. Matriz de Confusión Manual
etiquetas = sorted(list(set(clases_reales)))
matriz = {r: {p: 0 for p in etiquetas} for r in etiquetas}

for r, p in zip(clases_reales, clases_predichas):
    matriz[r][p] += 1

print("MATRIZ DE CONFUSIÓN RESULTANTE (ID3 DISCRETIZADO):")
print("      Predicho: ", "   ".join(etiquetas))
print("Real:")
total_aciertos = 0
for r in etiquetas:
    fila_str = f"  {r}             "
    for p in etiquetas:
        fila_str += f"{matriz[r][p]}   "
        if r == p:
            total_aciertos += matriz[r][p]
    print(fila_str)

precision = (total_aciertos / len(test_discreto)) * 100
print(f"\nExactitud del modelo discretizado: {precision:.2f}%")