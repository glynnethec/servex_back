import pandas as pd
from typing import TypedDict, List, Dict, Any, Set # Importar Set para el nuevo tipo
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory 
import os
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher
from collections import OrderedDict
import random

# ==================================================
# 0. UTILIDADES (Sin cambios)
# ==================================================

def normalize_text(s: Any) -> str:
    """Normaliza el texto: a minúsculas, elimina caracteres especiales y espacios extra."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\-/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def words_set(s: str) -> set:
    """Devuelve un conjunto de palabras normalizadas."""
    return set(w for w in normalize_text(s).split() if w)

def fuzzy_ratio(a: str, b: str) -> float:
    """Calcula la similitud de Levenshtein entre dos cadenas."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

CATEGORIAS = {
    "tables": [
        "table", "mesa", "worktable", "work station", "lab table",
        "drawing table", "art table", "mobile table", "modular table",
        "STEM table", "drafting table", "adjustable height table",
        "chemistry table", "physics table", "biological lab table",
        "trapezoid table", "A-frame table", "folding table",
        "electric lift table", "hand-crank table", "fabrication prep table"
    ],
    "benches": [
        "bench", "workbench", "angle iron bench", "sheet metal bench",
        "side clamp bench", "glue bench", "stain bench",
        "industrial bench", "apprentice bench", "student bench"
    ],
    "cabinets": [
        "cabinet", "armario", "gabinete", "storage cabinet", "tall storage",
        "wardrobe", "display cabinet", "wall cabinet", "base cabinet",
        "microscope cabinet", "safety cabinet", "flamable cabinet",
        "corrosive cabinet", "first aid cabinet", "eye safety cabinet"
    ],
    "carts": [
        "cart", "carro", "mobile cart", "demo cart", "utility cart",
        "multi-purpose cart", "chemical cart", "lab supply cart",
        "tool cart", "robotics cart"
    ],
    "sinks": [
        "sink", "wash station", "hand-washing station", "hygiene station",
        "eyewash station", "shower station", "stainless steel sink",
        "chemical resistant sink"
    ],
    "stools_chairs": [
        "stool", "chair", "taboret", "seat", "adjustable stool",
        "stackable chair", "cantilever chair", "ergonomic chair",
        "polyurethane stool", "vinyl chair"
    ],
    "tops": [
        "maple top", "maple butcher block", "maple countertop",
        "countertop", "phenolic top", "epoxy top", "stainless steel top",
        "laminate top", "chemguard top", "shoptop", "glass top",
        "dry-erase top", "non-skid top"
    ],
    "robotics": [
        "robot", "robotics", "robot compartment", "robot tote",
        "VEX robotics", "programmable robot station", "robot workbench"
    ],
    "safety": [
        "flammable", "acid", "corrosive", "safety", "first aid",
        "hepa filter", "paint hood", "fire extinguisher", "emergency shower",
        "chemical storage", "germicidal lamp"
    ],
    "lockers": [
        "locker", "steel locker", "mobile locker", "personal locker",
        "equipment locker", "tote locker"
    ],
    "accessories": [
        "clamp", "upright", "crossbar", "casters", "vise",
        "jug", "pump", "fixture", "filter", "electrical outlet",
        "modesty panel", "drawer slides", "shelf brackets"
    ]
}

MATERIALES_MADERA = [
    "oak", "veneer", "hardwood", "maple", "birch", "wood",
    "solid hardwood", "oak veneer", "maple bb", "maple block",
    "maple butcher block", "light hardwood", "ash wood", "plywood top",
    "laminate top with hardwood core", "face-glued hardwood", "MDF with veneer"
]


def generar_keywords_producto(p: Dict[str, Any]) -> List[str]:
    """Genera una lista de palabras clave (tokens) para un producto."""
    posibles_campos = [
        "Modelo", "Model Number", "Model", "Descripcion", "Producto", "Product", "Caracteristicas", "Features"
    ]
    partes = []
    for k in posibles_campos:
        if k in p and p.get(k):
            partes.append(str(p.get(k)))
    texto = normalize_text(" ".join(partes))
    tokens = [t for t in texto.split() if len(t) > 1]
    modelo_raw = p.get("Modelo") or p.get("Model Number") or p.get("Model") or ""
    modelo_norm = normalize_text(modelo_raw).replace(" ", "")
    if modelo_norm:
        tokens.append(modelo_norm)
    return list(OrderedDict.fromkeys(tokens))

def construir_indice_productos(catalogo: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Construye un índice de palabras clave por clave única de producto."""
    indice = {}
    for i, p in enumerate(catalogo):
        modelo_raw = p.get("Modelo") or p.get("Model Number") or p.get("Model") or f"idx{i}"
        clave = f"PROD_{normalize_text(modelo_raw).replace(' ', '_').replace('-', '_')}"
        indice[clave] = generar_keywords_producto(p)
    return indice

def buscar_por_categoria(pregunta: str) -> set:
    """Detecta las categorías presentes en el texto de la pregunta."""
    texto = normalize_text(pregunta)
    encontrados = set()
    for categoria, palabras in CATEGORIAS.items():
        for palabra in palabras:
            # Usar 'palabra' in 'texto' en lugar de un match exacto, permite flexibilidad
            if palabra in texto:
                encontrados.add(categoria)
                break
    return encontrados

def buscar_por_producto(pregunta: str, indice_productos: Dict[str, List[str]]) -> List[str]:
    """Busca productos específicos por palabras clave exactas en la pregunta."""
    texto = normalize_text(pregunta)
    matches = []
    for prod_key, palabras in indice_productos.items():
        if any(k in texto for k in palabras):
            matches.append(prod_key)
    return matches

# ==================================================
# 1. CARGAR CSV (Sin cambios)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "catalogo.csv")
# ASUMIMOS que el archivo 'catalogo.csv' existe en el mismo directorio.
try:
    CATALOGO = pd.read_csv(CSV_PATH, on_bad_lines="skip", dtype=str).fillna("")
except FileNotFoundError:
    print(f"ERROR: Archivo no encontrado en {CSV_PATH}. Creando un DataFrame vacío.")
    CATALOGO = pd.DataFrame()

# ==================================================
# 2. CONFIG LLM (Sin cambios)
# ==================================================
load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

# ==================================================
# 3. ESTADO GLOBAL POR CONVERSACIÓN (Añadiendo Memoria y Rastreo de Productos)
# ==================================================
class State(TypedDict):
    user_id: str
    mensaje: str
    catalogo: List[Dict[str, Any]]
    productos_filtrados: List[Dict[str, Any]]
    respuesta: str
    _indice_productos: Dict[str, List[str]]
    _respuestas_count: int
    memoria: ConversationBufferMemory
    _modelos_mencionados: Set[str] # 💡 MODIFICADO: Rastrea Modelos/IDs ya presentados.

# 💡 Almacenamiento de instancias de memoria por user_id
MEMORIA_SESIONES: Dict[str, ConversationBufferMemory] = {}

def get_or_create_memory(user_id: str) -> ConversationBufferMemory:
    """Devuelve la memoria para el usuario o crea una nueva."""
    if user_id not in MEMORIA_SESIONES:
        # Usamos ConversationBufferMemory para almacenar el historial completo.
        # k=4 almacena los últimos 4 intercambios (8 mensajes).
        MEMORIA_SESIONES[user_id] = ConversationBufferMemory(
            memory_key="historial_conversacion", # La clave que usará para guardar
            input_key="mensaje",
            output_key="respuesta",
            llm=llm,
            k=4,
            return_messages=True # Retorna lista de mensajes, más fácil de usar en el prompt
        )
    return MEMORIA_SESIONES[user_id]


def iniciar_conversacion(user_id: str, mensaje: str) -> State:
    """Inicializa el estado de la conversación y carga el catálogo."""
    memoria_instance = get_or_create_memory(user_id)

    state: State = {
        "user_id": user_id,
        "mensaje": mensaje,
        "catalogo": [],
        "productos_filtrados": [],
        "respuesta": "",
        "_indice_productos": {},
        "_respuestas_count": 0,
        "memoria": memoria_instance,
        "_modelos_mencionados": set(), # 💡 MODIFICADO: Inicializar el set vacío
    }
    state = load_catalog(state)
    return state

# ==================================================
# 4. NODE 1 — CARGAR CATÁLOGO (Sin cambios)
# ==================================================
def load_catalog(state: State) -> State:
    """Carga y preprocesa el catálogo en el estado."""
    # ... (Se mantiene el código original de preprocesamiento)
    rows = CATALOGO.to_dict(orient="records")
    for r in rows:
        r["_norm_categoria"] = normalize_text(r.get("Categoria", "") or r.get("Category", ""))
        r["_norm_subcategoria"] = normalize_text(r.get("Subcategoria", "") or r.get("Subcategory", ""))
        r["_norm_modelo"] = normalize_text(r.get("Modelo", "") or r.get("Model Number", "") or r.get("Model", ""))
        r["_norm_descripcion"] = normalize_text(r.get("Descripcion", "") or r.get("Description", "") or r.get("Producto", ""))
        r["_wordset"] = words_set(
            " ".join([
                r["_norm_categoria"],
                r["_norm_subcategoria"],
                r["_norm_modelo"],
                r["_norm_descripcion"],
                normalize_text(r.get("Caracteristicas", "") or r.get("Features", ""))
            ])
        )
    state["catalogo"] = rows
    state["_indice_productos"] = construir_indice_productos(rows)
    state["_respuestas_count"] = 0
    return state

# ==================================================
# 5. PRODUCT SELECTOR (Sin cambios en la lógica)
# ==================================================
def product_selector(state: State) -> State:
    """Selecciona los productos relevantes. (Lógica original intacta)."""
    pregunta_raw = state["mensaje"] or ""
    pregunta = normalize_text(pregunta_raw)
    tokens = [t for t in pregunta.split() if t]
    catalogo = state["catalogo"]
    indice_productos = state.get("_indice_productos", {})

    filtrados: List[Dict[str, Any]] = []
    candidatos: List[Dict[str, Any]] = []

    # 1. FILTRADO POR CATEGORÍAS DETECTADAS (Máxima prioridad)
    categorias_detectadas = buscar_por_categoria(pregunta)
    
    if categorias_detectadas:
        for p in catalogo:
            texto_producto = normalize_text(" ".join([
                p.get("Modelo","") or p.get("Model Number","") or p.get("Model",""),
                p.get("Descripcion","") or p.get("Description","") or p.get("Producto",""),
                p.get("Caracteristicas","") or p.get("Features","")
            ]))
            
            if any(any(term in texto_producto for term in CATEGORIAS[c]) for c in categorias_detectadas):
                candidatos.append(p)
        
        if candidatos:
            filtrados = candidatos
        
        if filtrados:
            dedup = OrderedDict()
            for p in filtrados:
                key = p.get("Modelo") or p.get("Model Number") or id(p)
                if key not in dedup:
                    dedup[key] = p
            state["productos_filtrados"] = list(dedup.values())
            return state

    # 2. FILTRADO POR MODELO O CÓDIGO ESPECÍFICO
    posible_modelo_token = None
    for t in tokens:
        if re.search(r"[a-zA-Z].*\d|\d.*[a-zA-Z]|-", t):
            posible_modelo_token = t
            break

    if posible_modelo_token:
        norm_token = normalize_text(posible_modelo_token)
        for p in catalogo:
            if p["_norm_modelo"] and (p["_norm_modelo"] == norm_token or p["_norm_modelo"].replace(" ","") == norm_token.replace(" ","")):
                filtrados.append(p)
        if not filtrados:
            for p in catalogo:
                if fuzzy_ratio(norm_token, p["_norm_modelo"]) >= 0.75:
                    filtrados.append(p)

    # 3. FILTRADO POR PRODUCTO INDEXADO
    if not filtrados:
        productos_detectados = buscar_por_producto(pregunta, indice_productos)
        if productos_detectados:
            for p in catalogo:
                modelo_raw = p.get("Modelo") or p.get("Model Number") or p.get("Model") or ""
                clave = f"PROD_{normalize_text(modelo_raw).replace(' ', '_').replace('-', '_')}"
                if clave in productos_detectados:
                    filtrados.append(p)

    # 4. FILTRADO POR PALABRAS CLAVE GENERALES / FUZZY MATCH
    if not filtrados:
        stopwords = {"el","la","los","las","de","del","con","para","y","o","en","un","una","que","qué"}
        query_tokens = [t for t in tokens if t not in stopwords]
        
        if query_tokens:
            for p in catalogo:
                if any(q in p["_wordset"] for q in query_tokens):
                    filtrados.append(p)
            
            if not filtrados:
                for p in catalogo:
                    combined = " ".join([p["_norm_modelo"], p["_norm_descripcion"]])
                    if any(fuzzy_ratio(q, combined) >= 0.68 for q in query_tokens):
                        filtrados.append(p)
        
        if not filtrados:
            filtrados = catalogo[:5]

    # 5. DESDUPLICACIÓN FINAL Y ASIGNACIÓN
    if not filtrados:
        filtrados = catalogo[:5]

    dedup = OrderedDict()
    for p in filtrados:
        key = p.get("Modelo") or p.get("Model Number") or id(p)
        if key not in dedup:
            dedup[key] = p

    state["productos_filtrados"] = list(dedup.values())
    return state


# ==================================================
# 6. ASESOR IA (MODIFICADO para usar Memoria y Evitar Repetición)
# ==================================================
def advisor_agent(state: State) -> State:
    """Invoca al LLM para generar una respuesta, **filtrando productos ya mencionados**."""
    state["_respuestas_count"] = state.get("_respuestas_count", 0) + 1

    productos_iniciales = state["productos_filtrados"]
    mensaje = state["mensaje"]
    memory = state["memoria"]
    modelos_mencionados = state["_modelos_mencionados"] # 💡 NUEVO: Recuperamos el set

    # --- NUEVA LÓGICA: Filtrar productos ya mencionados ---
    productos_para_mostrar = []
    modelos_actuales_a_mencionar = set()
    
    # 1. Filtramos la lista de productos filtrados por el selector.
    for p in productos_iniciales:
        # Usamos el Modelo/ID para identificar un producto único
        modelo_id = p.get("Modelo") or p.get("Model Number") or p.get("Model")
        
        # Si el ID existe y NO ha sido mencionado antes, lo incluimos.
        if modelo_id and modelo_id not in modelos_mencionados:
            productos_para_mostrar.append(p)
            modelos_actuales_a_mencionar.add(modelo_id)
    
    # Manejo del caso donde no hay productos nuevos.
    if not productos_para_mostrar:
        productos_a_resumir = productos_iniciales[:5] # Volvemos a mostrar los 5 primeros como último recurso
        resumen_header = (
            f"La búsqueda encontró **{len(productos_iniciales)}** productos, pero ya te mencioné los nuevos anteriormente. "
            f"Aquí están los primeros **{len(productos_a_resumir)}** para tu referencia."
        )
    else:
        # Usamos los productos filtrados y nuevos
        productos_a_resumir = productos_para_mostrar if len(productos_para_mostrar) <= 50 else random.sample(productos_para_mostrar, 50) 
        resumen_header = f"Se han encontrado **{len(productos_para_mostrar)}** productos relevantes (excluyendo los ya mencionados). Te enviaré información detallada de los primeros **{len(productos_a_resumir)}** para su análisis."
        
    # -----------------------------------------------------

    # 1. Cargamos el historial de la memoria (Sin cambios)
    historial_data = memory.load_memory_variables({})
    historial_lines = []
    for msg in historial_data.get("historial_conversacion", []):
        rol = "USUARIO" if msg.type == "human" else "ASISTENTE"
        historial_lines.append(f"[{rol}]: {msg.content}")
    historial_contexto = "\n".join(historial_lines)

    # 2. Resumen de productos (Usando `productos_a_resumir`)
    resumen_lines = []
    
    for p in productos_a_resumir:
        # ... (Formato de resumen de producto)
        resumen_lines.append(
            f"""
Producto:
  Modelo: {p.get("Modelo","dato no disponible") or p.get("Model Number","dato no disponible")}
  Descripcion: {p.get("Descripcion","dato no disponible") or p.get("Description","dato no disponible")}
  Dimensiones_W: {p.get("Dimensiones_W","dato no disponible")}
  Dimensiones_D: {p.get("Dimensiones_D","dato no disponible")}
  Dimensiones_H: {p.get("Dimensiones_H","dato no disponible")}
  Caracteristicas: {p.get("Caracteristicas","") or p.get("Features","")}
"""
        )

    resumen = resumen_header + "\n" + "\n".join(resumen_lines) if resumen_lines else "No hay productos filtrados."


    # 3. Construcción del Prompt (Añadiendo el historial)
    prompt = f"""ROL: SERVEX AI CONSULTANT. Asesor experto, amigable y conversacional de productos de Diversified Spaces. Guía al usuario como un colega.

TONO: Profesional, natural, cercano (sin exagerar).

HISTORIAL: {historial_contexto} (Usa el historial para mantener la continuidad, evitar repeticiones y retomar referencias).

CATÁLOGO: {resumen} (Datos ÚNICOS válidos. No inventes información no incluida).

PREGUNTA: "{mensaje}"

DIRECTRICES:
1. Responde a la intención del usuario (comparar, filtrar, sugerir, etc.).
2. Usa **SOLO** la información del CATÁLOGO. Si un dato falta, indícalo de forma natural.
3. Explica claro y conversacionalmente. Evita tecnicismos innecesarios.
4. Si hay >10 productos, resume características comunes y recomienda los 3-5 más relevantes para la consulta.
5. NO inventes ningún atributo (nombre, material, dimensión).
6. Objetivo: Respuesta útil, enfocada y profesional para ayudar en la elección.
"""

    resp = llm.invoke(prompt)
    respuesta_generada = getattr(resp, "content", None) or str(resp)

    # 💡 4. Guardamos el nuevo intercambio en la memoria de LangChain
    memory.save_context({"mensaje": mensaje}, {"respuesta": respuesta_generada})
    
    # 💡 5. MODIFICADO: Actualizar el set de modelos mencionados con los que acabamos de mostrar
    state["_modelos_mencionados"].update(modelos_actuales_a_mencionar)

    state["respuesta"] = respuesta_generada
    # Opcional: Reemplazar productos_filtrados con la lista solo de nuevos
    state["productos_filtrados"] = productos_para_mostrar 
    return state


# ==================================================
# 7. WORKFLOW LANGGRAPH (Sin cambios)
# ==================================================
workflow = StateGraph(State)
workflow.add_node("catalog", load_catalog)
workflow.add_node("selector", product_selector)
workflow.add_node("advisor", advisor_agent)

workflow.set_entry_point("catalog")
workflow.add_edge("catalog", "selector")
workflow.add_edge("selector", "advisor")
workflow.add_edge("advisor", END)

workflow_app = workflow.compile()

# ==================================================
# 8. GESTIÓN DE SESIONES (Modificado para usar la memoria)
# ==================================================
SESSIONS: Dict[str, State] = {}

def ejecutar_workflow(user_id: str, mensaje: str) -> str:
    """Función principal para ejecutar el workflow con gestión de sesiones."""
    # Recuperar estado existente o iniciar uno nuevo
    if user_id not in SESSIONS:
        # Inicia la conversación, creando y guardando la instancia de memoria
        SESSIONS[user_id] = iniciar_conversacion(user_id, mensaje)
    else:
        # Si ya existe, actualiza el mensaje y asegura que el catálogo esté cargado
        current_state = SESSIONS[user_id]
        if not current_state.get("catalogo"):
             SESSIONS[user_id] = load_catalog(current_state)
        SESSIONS[user_id]["mensaje"] = mensaje
        # El estado persistente (memoria y modelos_mencionados) se mantiene
        
    # Ejecutar workflow con el estado que contiene la instancia de memoria
    # El estado se actualiza con los resultados de los nodos
    SESSIONS[user_id] = workflow_app.invoke(SESSIONS[user_id])
    
    # La memoria ya fue actualizada dentro de advisor_agent
    return SESSIONS[user_id]["respuesta"]


# Ejemplo de uso/Comprobación (Si se ejecuta directamente el script)
if __name__ == '__main__':
    print("Workflow cargado con ConversationBufferMemory y lógica anti-repetición.")
    
    test_user_id = "test_memoria_123"
    
    # Limpiar sesiones anteriores si existen para la prueba
    if test_user_id in SESSIONS:
        del SESSIONS[test_user_id]
    if test_user_id in MEMORIA_SESIONES:
        del MEMORIA_SESIONES[test_user_id]

    # ----------------------------------------------------------------------
    # PRUEBA 1: Pregunta inicial (Filtra productos, se mencionan los primeros N)
    # ----------------------------------------------------------------------
    pregunta_1 = "Quiero ver todas las mesas que tengan tapa de arce" 
    print(f"\n→ Usuario 1: {pregunta_1}")
    respuesta_1 = ejecutar_workflow(test_user_id, pregunta_1)
    print(f"\n← AI Consultant 1:\n{respuesta_1}")
    
    estado_final_1 = SESSIONS[test_user_id]
    num_filtrados_1 = len(estado_final_1["productos_filtrados"])
    num_mencionados_1 = len(estado_final_1["_modelos_mencionados"])
    print(f"\n[INFO: El selector filtró inicialmente **{num_filtrados_1}** productos. Se han registrado **{num_mencionados_1}** modelos como mencionados.]")
    
    # ----------------------------------------------------------------------
    # PRUEBA 2: Pregunta de seguimiento o refinamiento
    # El selector vuelve a encontrar los mismos productos, PERO el advisor DEBE filtrar los ya mencionados.
    # ----------------------------------------------------------------------
    pregunta_2 = "De esas mesas de arce, ¿tienen alguna con ruedas (casters)?"
    print(f"\n--- Nueva consulta con memoria (buscando nuevas mesas con ruedas) ---")
    print(f"\n→ Usuario 2: {pregunta_2}")
    respuesta_2 = ejecutar_workflow(test_user_id, pregunta_2)
    print(f"\n← AI Consultant 2:\n{respuesta_2}")
    
    # Comprobación de que la memoria y el set de modelos se actualizaron
    estado_final_2 = SESSIONS[test_user_id]
    num_filtrados_2 = len(estado_final_2["productos_filtrados"])
    num_mencionados_2 = len(estado_final_2["_modelos_mencionados"])
    print(f"\n[INFO: El selector/advisor filtró para mostrar **{num_filtrados_2}** productos *nuevos* (con ruedas y sin repetir). El total de modelos mencionados es **{num_mencionados_2}**.]")
    
    historial_actual = estado_final_2["memoria"].load_memory_variables({})
    print("\n[INFO: Historial actual en la memoria:]")
    for msg in historial_actual.get("historial_conversacion", []):
         print(f" - {msg.type.upper()}: {msg.content[:50]}...")