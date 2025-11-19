import io
import os
import fitz
import uvicorn
import numpy as np
import base64 
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "detectorlogos-app-c2ce9c28a4f6.json"

#MODELO YOLO
try:
    model = YOLO("runs/detect/train/weights/best.pt")
    print("Modelo 'best.pt' cargado exitosamente.")
except Exception as e:
    print(f"Error cargando 'best.pt', usando modelo de prueba: {e}")
    model = YOLO("yolov8n.pt") 

#FASTAPI
app = FastAPI(title="API de Detección de Logos (YOLO + Google)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIÓN AUXILIAR GOOGLE VISION ---
# --- FUNCIÓN AUXILIAR GOOGLE VISION (CORREGIDA) ---
def consultar_google_vision(pil_image):
    """
    Recibe una imagen PIL, la convierte a bytes y la envía a Google Cloud.
    Retorna una lista de nombres de marcas detectadas.
    """
    try:
        # 1. Convertir la imagen PIL a bytes en memoria
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()

        # 2. Preparar el cliente y la imagen para Google
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)

        # 3. Llamada a la API (Solo Logos)
        response = client.logo_detection(image=image)
        logos = response.logo_annotations

        #Extraer nombres
        nombres_logos = []
        for logo in logos:
            nombres_logos.append(f"{logo.description} ({logo.score:.2f})")
            
        return nombres_logos 

    except Exception as e:
        print(f"Error conectando con Google Vision: {e}")
        return ["Error en Google Vision"]

#ENDPOINT
@app.post("/analizar-etiqueta-pdf/")
async def analyze_label_pdf(file: UploadFile = File(...)):
    print(f"Procesando archivo: {file.filename}")
    
    try:
        #leer PDF y extraer imagen
        pdf_bytes = await file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_doc.load_page(0) 
        pix = page.get_pixmap(dpi=300) 
        pdf_doc.close()
        
        #convertir a imagen PIL
        if pix.alpha:
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 4))
            pil_image = Image.fromarray(img_data[:, :, :3], 'RGB')
        else:
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            pil_image = Image.fromarray(img_data, 'RGB')
            
        #ejecutar predicción de YOLO
        results = model(pil_image, verbose=False)

        #INICIO DEL FILTRO
        if results and results[0].boxes:
            mejor_indice_por_clase = {}
            
            #Iteramos sobre todas las cajas detectadas
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0]) 
                conf = float(box.conf[0])
                
                #si no hemos visto esta clase, o si la actual tiene MEJOR puntaje que la guardada
                if cls_id not in mejor_indice_por_clase:
                    mejor_indice_por_clase[cls_id] = (conf, i)
                else:
                    #comparamos con la que ya teniamos guardada
                    conf_guardada = mejor_indice_por_clase[cls_id][0]
                    if conf > conf_guardada:
                        mejor_indice_por_clase[cls_id] = (conf, i)
            
            #extraemos solo los índices ganadores
            indices_finales = [item[1] for item in mejor_indice_por_clase.values()]
            results[0].boxes = results[0].boxes[indices_finales]
        # --- FIN DEL FILTRO ---


        #Extraer los datos en JSON (Ahora solo vendrán los filtrados)
        detected_classes_yolo = []
        if results and results[0].boxes:
            unique_class_indices = results[0].boxes.cls.unique()
            detected_classes_yolo = [model.names[int(cls_idx)] for cls_idx in unique_class_indices]
        
        print("Consultando Google Cloud Vision...")
        detected_classes_google = consultar_google_vision(pil_image)

        print(f"YOLO detectó: {detected_classes_yolo}")
        print(f"Google detectó: {detected_classes_google}")

        #generar PDF de respuesta
        image_with_boxes_np = results[0].plot()
        image_with_boxes_pil = Image.fromarray(image_with_boxes_np[..., ::-1], 'RGB')

        pdf_output_bytes_io = io.BytesIO()
        image_with_boxes_pil.save(
            pdf_output_bytes_io, 
            format="PDF", 
            resolution=300.0, 
            save_all=False
        )
        pdf_output_bytes = pdf_output_bytes_io.getvalue()
        pdf_base64_string = base64.b64encode(pdf_output_bytes).decode('utf-8')

        print("Procesamiento completo.")

        return JSONResponse(content={
            "detections_yolo": detected_classes_yolo,    
            "detections_google": detected_classes_google,
            "pdf_file_base64": pdf_base64_string
        })

    except Exception as e:
        print(f"Error grave procesando el PDF: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error: {e}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)