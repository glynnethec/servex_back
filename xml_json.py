import xml.etree.ElementTree as ET
import json
import os

def xml_to_dict(element):
    """
    Convierte un elemento XML recursivamente a un diccionario Python.
    Maneja: texto, atributos, hijos repetidos, y estructuras complejas.
    """
    node = {}

    # Si tiene atributos, agrégalos
    if element.attrib:
        node["@attributes"] = element.attrib

    # Texto interno (si existe y no es solo espacios)
    text = element.text.strip() if element.text and element.text.strip() else None
    if text:
        node["#text"] = text

    # Procesar hijos
    for child in element:
        child_dict = xml_to_dict(child)

        if child.tag not in node:
            node[child.tag] = child_dict
        else:
            # Si existe, convertir en lista para almacenar múltiples nodos iguales
            if isinstance(node[child.tag], list):
                node[child.tag].append(child_dict)
            else:
                node[child.tag] = [node[child.tag], child_dict]

    return node


def convert_xml_to_json(xml_filename):
    # Asegurar que existe
    if not os.path.exists(xml_filename):
        raise FileNotFoundError(f"No se encontró el archivo {xml_filename}")

    # Parsear XML
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    # Convertir a dict
    data_dict = {root.tag: xml_to_dict(root)}

    # Nombre del JSON final
    json_filename = xml_filename.replace(".xml", ".json")

    # Guardar
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

    print(f"✔ Conversión completa: {json_filename}")


if __name__ == "__main__":
    convert_xml_to_json("DWT.xml")
