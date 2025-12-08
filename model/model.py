import pandas as pd
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import random
import re
from difflib import SequenceMatcher
from collections import OrderedDict

# ==================================================
# 0. UTILIDADES
# ==================================================
def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\-/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def words_set(s: str) -> set:
    return set(w for w in normalize_text(s).split() if w)

def fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# ==================================================
# 0.5 UTILIDADES PARA INDEX DE PRODUCTOS / CATEGORÍAS
# ==================================================
# Categorías más especializadas
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

# Materiales de madera más especializados
MATERIALES_MADERA = [
    "oak", "veneer", "hardwood", "maple", "birch", "wood",
    "solid hardwood", "oak veneer", "maple bb", "maple block",
    "maple butcher block", "light hardwood", "ash wood", "plywood top",
    "laminate top with hardwood core", "face-glued hardwood", "MDF with veneer"
]


def generar_keywords_producto(p: Dict[str, Any]) -> List[str]:
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
    indice = {}
    for i, p in enumerate(catalogo):
        modelo_raw = p.get("Modelo") or p.get("Model Number") or p.get("Model") or f"idx{i}"
        clave = f"PROD_{normalize_text(modelo_raw).replace(' ', '_').replace('-', '_')}"
        indice[clave] = generar_keywords_producto(p)
    return indice

def buscar_por_categoria(pregunta: str) -> set:
    texto = normalize_text(pregunta)
    encontrados = set()
    for categoria, palabras in CATEGORIAS.items():
        for palabra in palabras:
            if palabra in texto:
                encontrados.add(categoria)
                break
    return encontrados

def buscar_por_producto(pregunta: str, indice_productos: Dict[str, List[str]]) -> List[str]:
    texto = normalize_text(pregunta)
    matches = []
    for prod_key, palabras in indice_productos.items():
        if any(k in texto for k in palabras):
            matches.append(prod_key)
    return matches

# ==================================================
# 1. CARGAR CSV
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "catalogo.csv")
CATALOGO = pd.read_csv(CSV_PATH, on_bad_lines="skip", dtype=str).fillna("")

# ==================================================
# 2. CONFIG LLM
# ==================================================
load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

# ==================================================
# 3. ESTADO GLOBAL POR CONVERSACIÓN
# ==================================================
class State(TypedDict):
    user_id: str
    mensaje: str
    catalogo: List[Dict[str, Any]]
    productos_filtrados: List[Dict[str, Any]]
    respuesta: str
    _indice_productos: Dict[str, List[str]]
    _respuestas_count: int

def iniciar_conversacion(user_id: str, mensaje: str) -> State:
    state: State = {
        "user_id": user_id,
        "mensaje": mensaje,
        "catalogo": [],
        "productos_filtrados": [],
        "respuesta": "",
        "_indice_productos": {},
        "_respuestas_count": 0
    }
    state = load_catalog(state)
    return state

# ==================================================
# 4. NODE 1 — CARGAR CATÁLOGO (y construir índice)
# ==================================================
def load_catalog(state: State) -> State:
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
# 5. NODE 2 — PRODUCT SELECTOR
# ==================================================
def product_selector(state: State) -> State:
    pregunta_raw = state["mensaje"] or ""
    pregunta = normalize_text(pregunta_raw)
    tokens = [t for t in pregunta.split() if t]
    catalogo = state["catalogo"]
    indice_productos = state.get("_indice_productos", {})

    filtrados: List[Dict[str, Any]] = []

    busca_almacenamiento = any(term in pregunta for term in [
        "armario","gabinete","estante","storage","cabinet","locker","shelf","tote","rack","closet","cubby"
    ])
    busca_mesa = any(term in pregunta for term in CATEGORIAS["tables"])
    busca_madera = any(mat in pregunta for mat in MATERIALES_MADERA)

    if busca_mesa and busca_madera:
        candidatos = []
        for p in catalogo:
            texto = normalize_text(" ".join([
                p.get("Modelo","") or p.get("Model Number","") or p.get("Model",""),
                p.get("Descripcion","") or p.get("Description","") or p.get("Producto",""),
                p.get("Caracteristicas","") or p.get("Features","")
            ]))
            es_mesa = any(term in texto for term in CATEGORIAS["tables"])
            es_madera = any(mat in texto for mat in MATERIALES_MADERA)
            if es_mesa and es_madera:
                candidatos.append(p)
        if candidatos:
            dedup = OrderedDict()
            for p in candidatos:
                key = p.get("Modelo") or p.get("Model Number") or id(p)
                if key not in dedup:
                    dedup[key] = p
            state["productos_filtrados"] = list(dedup.values())[:10]
            return state

    categorias_detectadas = buscar_por_categoria(pregunta)
    if categorias_detectadas:
        candidatos = []
        for p in catalogo:
            texto = normalize_text(" ".join([
                p.get("Modelo","") or p.get("Model Number","") or p.get("Model",""),
                p.get("Descripcion","") or p.get("Description","") or p.get("Producto",""),
                p.get("Caracteristicas","") or p.get("Features","")
            ]))
            if any(any(term in texto for term in CATEGORIAS[c]) for c in categorias_detectadas):
                candidatos.append(p)
        if candidatos:
            dedup = OrderedDict()
            for p in candidatos:
                key = p.get("Modelo") or p.get("Model Number") or id(p)
                if key not in dedup:
                    dedup[key] = p
            state["productos_filtrados"] = list(dedup.values())[:10]
            return state

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

    if not filtrados:
        productos_detectados = buscar_por_producto(pregunta, indice_productos)
        if productos_detectados:
            for p in catalogo:
                modelo_raw = p.get("Modelo") or p.get("Model Number") or p.get("Model") or ""
                clave = f"PROD_{normalize_text(modelo_raw).replace(' ', '_').replace('-', '_')}"
                if clave in productos_detectados:
                    filtrados.append(p)

    if not filtrados:
        stopwords = {"el","la","los","las","de","del","con","para","y","o","en","un","una","que","qué"}
        query_tokens = [t for t in tokens if t not in stopwords]
        if not query_tokens:
            filtrados = catalogo[:5]
        else:
            for p in catalogo:
                if any(q in p["_wordset"] for q in query_tokens):
                    filtrados.append(p)
            if not filtrados:
                for p in catalogo:
                    combined = " ".join([p["_norm_modelo"], p["_norm_descripcion"]])
                    if any(fuzzy_ratio(q, combined) >= 0.68 for q in query_tokens):
                        filtrados.append(p)

    if not filtrados:
        for p in catalogo:
            if any(cat in p["_norm_categoria"] for cat in tokens):
                filtrados.append(p)

    if not filtrados:
        filtrados = catalogo[:5]

    dedup = OrderedDict()
    for p in filtrados:
        key = p.get("Modelo") or p.get("Model Number") or id(p)
        if key not in dedup:
            dedup[key] = p

    state["productos_filtrados"] = list(dedup.values())[:10]
    return state

# ==================================================
# 6. NODE 3 — ASESOR IA
# ==================================================
# ==================================================
# 6. NODE 3 — ASESOR IA
# ==================================================
def advisor_agent(state: State) -> State:
    state["_respuestas_count"] = state.get("_respuestas_count", 0) + 1
    # ✅ Se eliminó el reinicio automático de productos cada 6 respuestas
    # if state["_respuestas_count"] > 6:
    #     state["productos_filtrados"] = []
    #     state["_respuestas_count"] = 1

    productos = state["productos_filtrados"]
    mensaje = state["mensaje"]

    resumen_lines = []
    for p in productos:
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

    resumen = "\n".join(resumen_lines) if resumen_lines else "No hay productos filtrados."

    prompt = f"""Actúas como un asesor especializado en los productos del catálogo de Diversified Spaces. 
Tu tono es humano, claro y centrado en ayudar al usuario a entender los productos y a
compararlos según lo que necesite, sin sonar rígido ni mecánico.

Toda la información que usas proviene únicamente del catálogo centralizado.  
Puedes describir, contextualizar, relacionar y orientar, pero siempre con base en lo que 
está explícitamente registrado en los datos.

Tu estilo:
- Hablas con naturalidad, como alguien que ya viene conversando con el usuario.
- Das explicaciones prácticas y útiles basándote estrictamente en el catálogo.
- Si el usuario busca algo específico, le ayudas a filtrar, comparar o identificar productos.
- Si falta información, lo dices sin problema: por ejemplo: “Ese dato no aparece en el catálogo”.
- No vendes; acompañas el análisis. Tu rol es claridad, no persuasión.

Lo que SÍ puedes hacer:
- Organizar y explicar la información tal cual aparece en el CSV.
- Señalar diferencias o similitudes entre productos, siempre basándote en los datos.
- Guiar al usuario con preguntas cuando sea útil para aclarar lo que necesita.

Lo que NO puedes hacer:
1. No puedes inventar ni asumir características que no estén literalmente en el catálogo.
2. No puedes inferir medidas, materiales, capacidades, ni usos si no aparecen explícitos.
3. No extrapoles ni agregues contexto externo.

Tu respuesta se genera en función de la pregunta del usuario:
→ "{mensaje}"

CATÁLOGO DISPONIBLE:
{resumen}

"""

    resp = llm.invoke(prompt)
    state["respuesta"] = getattr(resp, "content", None) or str(resp)
    return state


# ==================================================
# 7. WORKFLOW LANGGRAPH
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
# 8. FUNCIÓN PARA EJECUTAR POR USUARIO / SESIÓN
# ==================================================
def ejecutar_workflow(user_id: str, mensaje: str) -> str:
    state = iniciar_conversacion(user_id, mensaje)
    state = workflow_app.run(state)
    return state["respuesta"]

print("Workflow cargado con memoria aislada por conversación y reinicio cada 6 respuestas.")
