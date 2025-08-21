package com.example.cashvision

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.TextView
import android.view.View
import android.view.animation.AnimationUtils
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.cashvision.camera.ImageAnalyzer
import com.example.cashvision.ml.Detection
import com.example.cashvision.ml.YoloDetector
import com.example.cashvision.ui.DetectionOverlay
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnFlash: Button
    private lateinit var btnDetection: Button
    private lateinit var statusText: TextView
    private lateinit var statusIndicator: View
    private lateinit var detectionOverlay: DetectionOverlay

    private var camera: Camera? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var yoloDetector: YoloDetector? = null

    private var isFlashOn = false
    private var isDetectionActive = false

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else finish()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeViews()
        setupClickListeners()
        initializeYoloDetector()
        requestCameraPermission()
    }

    private fun initializeViews() {
        previewView = findViewById(R.id.previewView)
        btnFlash = findViewById(R.id.btnFlash)
        btnDetection = findViewById(R.id.btnDetection)
        statusText = findViewById(R.id.statusText)
        statusIndicator = findViewById(R.id.statusIndicator)
        detectionOverlay = findViewById(R.id.detectionOverlay)

        // Initialize UI state
        updateFlashButtonState(false)
        updateDetectionButtonState(false)
        updateStatusIndicator(false)
    }

    private fun initializeYoloDetector() {
        statusText.text = getString(R.string.status_model_loading)

        lifecycleScope.launch {
            try {
                yoloDetector = YoloDetector(this@MainActivity)
                val success = yoloDetector?.initialize() ?: false

                if (success) {
                    statusText.text = getString(R.string.status_camera_ready)
                } else {
                    statusText.text = getString(R.string.status_model_error)
                }
            } catch (e: Exception) {
                statusText.text = getString(R.string.status_model_error)
                e.printStackTrace()
            }
        }
    }

    private fun setupClickListeners() {
        btnFlash.setOnClickListener {
            animateButtonPress(btnFlash)
            toggleFlash()
        }

        btnDetection.setOnClickListener {
            animateButtonPress(btnDetection)
            toggleDetection()
        }
    }

    private fun animateButtonPress(button: Button) {
        val pressAnim = AnimationUtils.loadAnimation(this, R.anim.button_press)
        val releaseAnim = AnimationUtils.loadAnimation(this, R.anim.button_release)

        button.startAnimation(pressAnim)
        button.postDelayed({
            button.startAnimation(releaseAnim)
        }, 150)
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                // Setup image analysis for YOLO detection
                setupImageAnalysis()

                val selector = CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, selector, preview, imageAnalyzer
                )

                // Update UI when camera is ready
                updateStatusIndicator(true)
                if (yoloDetector != null) {
                    statusText.text = getString(R.string.status_camera_ready)
                }

            } catch (exc: Exception) {
                statusText.text = getString(R.string.status_camera_error)
                updateStatusIndicator(false)
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun setupImageAnalysis() {
        val detector = yoloDetector ?: return

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9) // Usar relación de aspecto 16:9
            .setTargetRotation(previewView.display.rotation) // Mantener la rotación de la pantalla
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analyzer ->
                analyzer.setAnalyzer(
                    ContextCompat.getMainExecutor(this),
                    ImageAnalyzer(detector) { detections, imageWidth, imageHeight ->
                        onDetectionResult(detections, imageWidth, imageHeight)
                    }
                )
            }
    }

    private fun toggleFlash() {
        val info = camera?.cameraInfo ?: return
        val control = camera?.cameraControl ?: return

        if (info.hasFlashUnit()) {
            isFlashOn = !isFlashOn
            control.enableTorch(isFlashOn)
            updateFlashButtonState(isFlashOn)
            updateStatusText(isFlashOn)
        } else {
            statusText.text = getString(R.string.status_no_flashlight)
        }
    }

    private fun toggleDetection() {
        isDetectionActive = !isDetectionActive
        updateDetectionButtonState(isDetectionActive)

        if (isDetectionActive) {
            statusText.text = getString(R.string.status_detection_on)
        } else {
            statusText.text = getString(R.string.status_detection_off)
            detectionOverlay.clearDetections()
        }
    }

    private fun onDetectionResult(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        if (isDetectionActive) {
            Log.d("MainActivity", "Received ${detections.size} detections from model")

            // Escalar las coordenadas de la imagen al tamaño de la vista
            val scaledDetections = scaleDetectionsToView(detections, imageWidth, imageHeight)

            // Log de todas las detecciones para debugging
            scaledDetections.forEachIndexed { index, detection ->
                Log.d("MainActivity", "Detection $index: ${detection.className} confidence=${detection.confidence} bbox=${detection.bbox}")
            }

            // Filtrar detecciones con confianza consistente con el detector (0.3f)
            val validDetections = scaledDetections.filter { it.confidence >= 0.3f }

            Log.d("MainActivity", "Valid detections (>=30%): ${validDetections.size}")

            // Mostrar todas las detecciones válidas
            val detectionsToShow = validDetections

            detectionOverlay.updateDetections(detectionsToShow)

            if (detectionsToShow.isNotEmpty()) {
                // Mostrar información de la primera detección o un resumen
                val firstDetection = detectionsToShow.first()
                statusText.text = "Detectado: ${firstDetection.getFormattedDenomination()} (${(firstDetection.confidence * 100).toInt()}%)"
                Log.d("MainActivity", "Showing ${detectionsToShow.size} detections. First: ${firstDetection.className} with ${(firstDetection.confidence * 100).toInt()}%")
            } else {
                statusText.text = getString(R.string.status_detecting)
            }
        }
    }

    private fun scaleDetectionsToView(detections: List<Detection>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val viewWidth = previewView.width.toFloat()
        val viewHeight = previewView.height.toFloat()

        if (viewWidth <= 0 || viewHeight <= 0) return detections

        val scaleX = viewWidth / imageWidth
        val scaleY = viewHeight / imageHeight

        return detections.map { detection ->
            val scaledBbox = android.graphics.RectF(
                detection.bbox.left * scaleX,
                detection.bbox.top * scaleY,
                detection.bbox.right * scaleX,
                detection.bbox.bottom * scaleY
            )

            detection.copy(bbox = scaledBbox)
        }
    }

    private fun updateFlashButtonState(isOn: Boolean) {
        btnFlash.isSelected = isOn
        btnFlash.text = getString(if (isOn) R.string.btn_flashlight_off else R.string.btn_flashlight)
    }

    private fun updateDetectionButtonState(isOn: Boolean) {
        btnDetection.isSelected = isOn
        btnDetection.text = getString(if (isOn) R.string.btn_detection_off else R.string.btn_detection)
    }

    private fun updateStatusIndicator(isActive: Boolean) {
        statusIndicator.isSelected = isActive
    }

    private fun updateStatusText(flashOn: Boolean) {
        statusText.text = getString(
            if (flashOn) R.string.status_flashlight_on
            else R.string.status_flashlight_off
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector?.close()
    }
}
