from model.model import app

def run_cli():
    print("\n=== CLI del Asesor de Mobiliario Educativo ===")
    print("Escribe tu pregunta o 'salir' para terminar.\n")

    while True:
        pregunta = input("Tu pregunta: ")

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Saliendo del asesor. ¡Hasta luego!")
            break

        estado_inicial = {
            "user_id": "cli_user",
            "mensaje": pregunta,
            "catalogo": [],
            "productos_filtrados": [],
            "respuesta": ""
        }

        resultado = app.invoke(estado_inicial)
        print("\n--- Respuesta del asesor ---")
        print(resultado["respuesta"])
        print("\n")


if __name__ == "__main__":
    run_cli()
