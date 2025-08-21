from ultralytics import YOLO

# Cargar un modelo pre-entrenado (YOLOv8n es un buen punto de partida)
# Puedes descargar otros modelos como 'yolov8s.pt', 'yolov8m.pt', etc.
model = YOLO('yolov8n.pt')

# Ruta a tu archivo de configuración del dataset (data.yaml)
# Asegúrate de que este archivo esté correctamente configurado con las rutas a tus imágenes y etiquetas.
data_yaml_path = 'app/src/main/assets/data.yaml'

# Parámetros de entrenamiento
# epochs: número de épocas de entrenamiento
# imgsz: tamaño de la imagen de entrada (debe ser un múltiplo de 32)
# batch: tamaño del batch (ajusta según la memoria de tu GPU)
# name: nombre de la carpeta donde se guardarán los resultados (ej. runs/detect/train_custom)
results = model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=16, name='train_custom_billetes')

# Exportar el modelo entrenado a formato ONNX
# El modelo se guardará en la carpeta de resultados del entrenamiento (ej. runs/detect/train_custom/weights/best.onnx)
# Asegúrate de que el input_size (imgsz) sea el mismo que usas en tu aplicación Android (640x640)
model.export(format='onnx', imgsz=640)

print("Entrenamiento y exportación a ONNX completados.")
print("El modelo ONNX se encuentra en la carpeta de resultados del entrenamiento, por ejemplo: runs/detect/train_custom_billetes/weights/best.onnx")