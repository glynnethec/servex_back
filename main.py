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
    "http://localhost:3000",  # tu frontend
    "http://127.0.0.1:3000",
    "https://servex-demo-v0.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # permite POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

# ----------------------------
# Modelo para recibir consulta
# ----------------------------
class Consulta(BaseModel):
    user_id: str = "frontend_user"
    mensaje: str

@app.post("/asesor")
async def asesor_endpoint(data: Consulta):
    estado_inicial = {
        "user_id": data.user_id,
        "mensaje": data.mensaje,
        "catalogo": [],
        "productos_filtrados": [],
        "respuesta": ""
    }

    resultado = workflow_app.invoke(estado_inicial)

    return {"respuesta": resultado["respuesta"]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
