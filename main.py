from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 💡 IMPORTANTE: Asegúrate de que 'iniciar_conversacion' esté en tu model.py
#                y que lo exporte junto con 'workflow_app'
from model.model import workflow_app, iniciar_conversacion 
import uvicorn
import copy # Importar para manejar la copia del estado de manera segura

app = FastAPI(
    title="API Asesor de Mobiliario Educativo",
    description="Servicio que usa LangGraph + Groq para asesorar sobre mobiliario educativo.",
    version="1.0.0"
)

# ----------------------------
# Habilitar CORS (Sin cambios)
# ----------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://servex-demo-v0.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# MEMORY PERSISTENTE EN RAM (Sin cambios)
# ----------------------------
memory_store = {}  # <- AQUI SE GUARDA LA MEMORIA COMPLETA (incluyendo el objeto LangChain Memory)

# ----------------------------
# Modelo para recibir consulta (Sin cambios)
# ----------------------------
class Consulta(BaseModel):
    user_id: str = "frontend_user"
    mensaje: str

@app.post("/asesor")
async def asesor_endpoint(data: Consulta):

    # 1. Recuperar o Inicializar el estado (CORREGIDO)
    if data.user_id not in memory_store:
        # 💡 Si el usuario es nuevo, usamos 'iniciar_conversacion' para crear
        #    el estado **COMPLETO**, incluyendo el objeto 'memoria'.
        estado_prev = iniciar_conversacion(data.user_id, data.mensaje)
    else:
        # 💡 Si ya existe, recuperamos el estado previo.
        #    Usamos copy.copy() para trabajar con una copia superficial y evitar 
        #    problemas si el invoke modificara el objeto antes de guardarlo.
        estado_prev = copy.copy(memory_store[data.user_id])
        
        # 2. Se actualiza SOLO el mensaje
        estado_prev["mensaje"] = data.mensaje

    # 3. Se manda al workflow
    # El estado_prev ahora contiene el objeto de LangChain Memory,
    # asegurando que la memoria se mantenga.
    resultado = workflow_app.invoke(estado_prev)

    # 4. Guardamos memoria actualizada
    memory_store[data.user_id] = resultado

    return {"respuesta": resultado["respuesta"]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)