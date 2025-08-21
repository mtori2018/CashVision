package com.example.cashvision.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.exp

private fun sigmoid(x: Float): Float {
    return 1.0f / (1.0f + exp(-x))
}

class YoloDetector(private val context: Context) {
    
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null
    
    // Model configuration
    private val inputSize = 640
    private val confidenceThreshold = 0.6f  // Aumentado para reducir falsos positivos
    private val iouThreshold = 0.45f  // Se mantiene igual
    
    // Class names - adjust according to your model (solo 5 clases)
    private val classNames = arrayOf(
        "billete_1000",
        "billete_2000",
        "billete_5000",
        "billete_10000" // Ajustado a 4 clases según la salida del modelo [1, 9, 8400]
    )
    
    companion object {
        private const val TAG = "YoloDetector"
    }
    
    /**
     * Initialize the ONNX model
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            val modelBytes = context.assets.open("yolo_model.onnx").use { inputStream ->
                inputStream.readBytes()
            }
            
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.addCPU(false) // Use CPU
            
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            
            Log.d(TAG, "YOLO model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize YOLO model", e)
            false
        }
    }
    
    /**
     * Run detection on a bitmap
     */
    suspend fun detect(bitmap: Bitmap): List<Detection> = withContext(Dispatchers.Default) {
        val session = ortSession ?: return@withContext emptyList()
        
        try {
            // Preprocess image
            val preprocessedData = preprocessImage(bitmap)
            
            // Create input tensor
            val inputName = session.inputNames.iterator().next()
            val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, preprocessedData, shape)
            
            // Run inference
            val inputs = mapOf(inputName to inputTensor)
            val outputs = session.run(inputs)

            // Debug output info
            Log.d(TAG, "Model outputs count: ${outputs.size()}")
            for (i in 0 until outputs.size()) {
                val output = outputs[i]
                Log.d(TAG, "Output $i info: ${output.info}")
                Log.d(TAG, "Output $i type: ${output.value::class.java.simpleName}")
            }
            
            // Process outputs - El log indica que el formato es float[][][]
            // Esto significa que outputs[0].value es Array<Array<FloatArray>>
            // Y la forma es [1, 9, 8400]
            // Donde outputValue[0] es un Array<FloatArray> de tamaño 9
            // Y cada FloatArray tiene 8400 elementos (para x, y, w, h, obj, class0, ..., class4)
            val outputValue = outputs[0].value
            val detections = if (outputValue is Array<*>) {
                val outputArray = outputValue as Array<Array<FloatArray>>
                postprocessOutput(outputArray, bitmap.width, bitmap.height)
            } else {
                Log.e(TAG, "Unexpected output format: ${outputValue::class.java.simpleName}. Expected Array<Array<FloatArray>>.")
                emptyList()
            }
            
            // Clean up
            inputTensor.close()
            outputs.close()
            
            detections
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
            emptyList()
        }
    }
    
    /**
     * Preprocess image for YOLO input
     */
    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        Log.d(TAG, "Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${inputSize}x${inputSize}")

        // Resize bitmap to model input size manteniendo aspect ratio
        val resizedBitmap = resizeBitmapWithPadding(bitmap, inputSize, inputSize)

        val buffer = FloatBuffer.allocate(3 * inputSize * inputSize)
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to CHW format (channels first) y normalizar correctamente
        // Orden: B channel completo, luego G channel completo, luego R channel completo (común en modelos YOLO)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val b = (pixel and 0xFF) / 255.0f
            buffer.put(b)
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            buffer.put(g)
        }

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            buffer.put(r)
        }

        buffer.rewind()
        resizedBitmap.recycle()
        Log.d(TAG, "Image preprocessing completed")
        return buffer
    }

    /**
     * Resize bitmap - SIMPLIFICADO para evitar problemas de coordenadas
     */
    private fun resizeBitmapWithPadding(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        val scale: Float
        val xOffset: Float
        val yOffset: Float

        if (originalWidth > originalHeight) {
            scale = targetWidth.toFloat() / originalWidth
            xOffset = 0f
            yOffset = (targetHeight - originalHeight * scale) / 2f
        } else {
            scale = targetHeight.toFloat() / originalHeight
            yOffset = 0f
            xOffset = (targetWidth - originalWidth * scale) / 2f
        }

        val matrix = android.graphics.Matrix()
        matrix.postScale(scale, scale)
        matrix.postTranslate(xOffset, yOffset)

        val paddedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(paddedBitmap)
        canvas.drawBitmap(bitmap, matrix, null)

        Log.d(TAG, "Resized bitmap from ${originalWidth}x${originalHeight} to ${targetWidth}x${targetHeight} with padding. Scale: $scale, Offset: ($xOffset, $yOffset)")
        return paddedBitmap
    }
    
    /**
     * Post-process YOLO output para formato [1, 9, 8400]
     */
    private fun postprocessOutput(
        output: Array<Array<FloatArray>>, // Ahora es Array<Array<FloatArray>>
        originalWidth: Int,
        originalHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        Log.d(TAG, "Processing transposed output format")
        // output[0] es el array de 9 FloatArrays, cada uno con 8400 elementos
        val numDetections = output[0][0].size // 8400 detecciones
        val elementsPerDetection = output[0].size // 9 elementos por detección (x,y,w,h,obj,class_scores)

        Log.d(TAG, "Output dimensions: ${elementsPerDetection} x ${numDetections}")

        for (i in 0 until numDetections) {
            val centerX = output[0][0][i]
            val centerY = output[0][1][i]
            val width = output[0][2][i]
            val height = output[0][3][i]
            val objectnessRaw = output[0][4][i] // Raw objectness score

            // Aplicar sigmoide a objectness
            val objectness = sigmoid(objectnessRaw)

            // Log.d(TAG, "Raw detection $i: centerX=$centerX, centerY=$centerY, width=$width, height=$height, objectnessRaw=$objectnessRaw, objectness=$objectness")

            if (objectness < confidenceThreshold) {
                // Log.d(TAG, "Skipping detection $i with low objectness: $objectness")
                continue
            }

            // Find best class and apply sigmoid to class scores
            var bestClassId = -1
            var bestClassScore = 0f

            for (j in classNames.indices) {
                val classScoreRaw = output[0][5 + j][i] // Raw class score
                val activatedClassScore = sigmoid(classScoreRaw)
                if (activatedClassScore > bestClassScore) {
                    bestClassScore = activatedClassScore
                    bestClassId = j
                }
            }

            // Calcular confianza final: objectness * bestClassScore (ambos ya sigmoid)
            val finalConfidence = objectness * bestClassScore

            // Validaciones básicas para reducir falsos positivos (se confía más en confidenceThreshold)

            // Log.d(TAG, "Best class: $bestClassId (${classNames.getOrNull(bestClassId)}), classScore=$bestClassScore, finalConfidence=$finalConfidence")

            if (finalConfidence < confidenceThreshold || bestClassId == -1) {
                // Log.d(TAG, "Skipping detection with low final confidence: $finalConfidence")
                continue
            }

            // Coordenadas están normalizadas (0-1) con respecto al inputSize (640x640)
            // Convertir a píxeles de la imagen original
            val scaleX = originalWidth.toFloat() / inputSize
            val scaleY = originalHeight.toFloat() / inputSize

            val left = (centerX - width / 2) * scaleX
            val top = (centerY - height / 2) * scaleY
            val right = (centerX + width / 2) * scaleX
            val bottom = (centerY + height / 2) * scaleY

            val bbox = RectF(
                max(0f, left),
                max(0f, top),
                min(originalWidth.toFloat(), right),
                min(originalHeight.toFloat(), bottom)
            )

            // Validaciones básicas de tamaño
            val bboxWidth = bbox.width()
            val bboxHeight = bbox.height()

            if (bboxWidth < 5f || bboxHeight < 5f) { // Reducir umbral de tamaño mínimo
                Log.d(TAG, "Skipping detection with too small bbox: ${bboxWidth}x${bboxHeight}")
                continue
            }

            // Validación menos estricta del área
            val imageArea = originalWidth * originalHeight
            val bboxArea = bboxWidth * bboxHeight
            val areaRatio = bboxArea / imageArea

            if (areaRatio > 0.99f) { // Aumentar umbral de área máxima
                Log.d(TAG, "Skipping detection with too large bbox (${(areaRatio * 100).toInt()}% of image)")
                continue
            }

            Log.d(TAG, "Adding detection: bbox=$bbox, confidence=$finalConfidence")

            detections.add(
                Detection(
                    bbox = bbox,
                    confidence = finalConfidence,
                    classId = bestClassId,
                    className = classNames[bestClassId]
                )
            )
        }

        Log.d(TAG, "Found ${detections.size} valid detections before NMS")

        val finalDetections = applyNMS(detections)
        Log.d(TAG, "Final detections after NMS: ${finalDetections.size}")

        return finalDetections
    }
    
    /**
     * Apply Non-Maximum Suppression - Mejorado para reducir detecciones múltiples
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        Log.d(TAG, "Applying NMS to ${detections.size} detections")

        // Ordenar por confianza descendente
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val result = mutableListOf<Detection>()
        val suppressed = BooleanArray(sortedDetections.size) { false }

        for (i in sortedDetections.indices) {
            if (suppressed[i]) continue

            val currentDetection = sortedDetections[i]
            result.add(currentDetection)
            Log.d(TAG, "Keeping detection: ${currentDetection.className} with confidence ${currentDetection.confidence}")

            for (j in (i + 1) until sortedDetections.size) {
                if (suppressed[j]) continue

                val otherDetection = sortedDetections[j]

                // Aplicar NMS solo si son de la misma clase
                if (currentDetection.classId == otherDetection.classId) {
                    val iou = calculateIoU(currentDetection.bbox, otherDetection.bbox)
                    if (iou > iouThreshold) {
                        Log.d(TAG, "Suppressing detection with IoU $iou > $iouThreshold for same class")
                        suppressed[j] = true
                    }
                }
            }
        }

        Log.d(TAG, "NMS result: ${result.size} detections kept")
        return result
    }
    
    /**
     * Calculate Intersection over Union
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)
        
        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0f
        }
        
        val intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
    }
}
