from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Sobrescribir TODO EL DICCIONARIO DE NOMBRES
model.model.names = {
    0: "NOM",
    1: "NOM-CE",
    2: "NOM-EAC",
    3: "NOM-NYCE",
    4: "NOM-UL",
    5: "alto voltaje",
    6: "cont. especial", 
    7: "choque electr",
    8: "doble aislamiento",
}

model.save("best_fixed.pt")

print("NUEVOS NOMBRES:", model.model.names)

