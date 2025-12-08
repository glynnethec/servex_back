from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import workflow_app, iniciar_conversacion, SESSIONS
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
# Modelo para recibir consulta
# ----------------------------
class Consulta(BaseModel):
    user_id: str = "frontend_user"
    mensaje: str


# =====================================================
#  ENDPOINT: ASESOR CON MEMORIA REAL POR USUARIO
# =====================================================
@app.post("/asesor")
async def asesor_endpoint(data: Consulta):

    # -----------------------------
    # 1. Verificar si es un usuario nuevo
    # -----------------------------
    if data.user_id not in SESSIONS:
        print(f"⚡ Nueva sesión creada para {data.user_id}")
        SESSIONS[data.user_id] = iniciar_conversacion(
            data.user_id,
            data.mensaje
        )

    else:
        # Usuario ya existe → actualizar solo el mensaje
        SESSIONS[data.user_id]["mensaje"] = data.mensaje

    # -----------------------------
    # 2. Ejecutar workflow con estado persistente
    # -----------------------------
    estado_actual = workflow_app.run(SESSIONS[data.user_id])

    # Guardar el estado actualizado
    SESSIONS[data.user_id] = estado_actual

    # -----------------------------
    # 3. Respuesta al frontend
    # -----------------------------
    return {
        "respuesta": estado_actual["respuesta"]
    }


# =====================================================
#  RUN SERVER
# =====================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
