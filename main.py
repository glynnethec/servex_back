from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import workflow_app
import uvicorn

app = FastAPI(
    title="API Asesor de Mobiliario Educativo",
    description="Servicio que usa LangGraph + Groq para asesorar sobre mobiliario educativo.",
    version="1.0.0"
)

# ----------------------------
# Habilitar CORS
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
# MEMORY PERSISTENTE EN RAM
# ----------------------------
memory_store = {}  # <- AQUI SE GUARDA LA MEMORIA POR user_id

# ----------------------------
# Modelo para recibir consulta
# ----------------------------
class Consulta(BaseModel):
    user_id: str = "frontend_user"
    mensaje: str

@app.post("/asesor")
async def asesor_endpoint(data: Consulta):

    # Si el usuario NO tiene memoria, se crea solo una vez
    if data.user_id not in memory_store:
        memory_store[data.user_id] = {
            "user_id": data.user_id,
            "mensaje": "",
            "catalogo": [],
            "productos_filtrados": [],
            "respuesta": ""
        }

    # Recuperamos su estado previo
    estado_prev = memory_store[data.user_id]

    # Se actualiza SOLO el mensaje
    estado_prev["mensaje"] = data.mensaje

    # Se manda al workflow, conservando el resto del estado previo
    resultado = workflow_app.invoke(estado_prev)

    # Guardamos memoria actualizada
    memory_store[data.user_id] = resultado

    return {"respuesta": resultado["respuesta"]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
