# Nuevo CLI.py (Usa la función con gestión de memoria)
# Ya no necesitas importar 'app' ni usar 'estado_inicial' manualmente
# Si la función está definida así: def ejecutar_workflow(user_id: str, mensaje: str) -> str:
from model.model import ejecutar_workflow

def run_cli():
    print("\n=== CLI del Asesor de Mobiliario Educativo ===")
    print("Escribe tu pregunta o 'salir' para terminar.\n")

    user_id = "cli_user" # ID de usuario fijo para la CLI

    while True:
        pregunta = input("Tu pregunta: ")

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Saliendo del asesor. ¡Hasta luego!")
            break

        # Llama a la función que ejecuta todo el workflow, incluyendo la memoria (SESSIONS)
        respuesta = ejecutar_workflow(user_id, pregunta) 

        print("\n--- Respuesta del asesor ---")
        print(respuesta)
        print("\n")


if __name__ == "__main__":
    run_cli()