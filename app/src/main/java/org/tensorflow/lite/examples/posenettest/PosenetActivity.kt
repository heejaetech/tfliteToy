/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenettest

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.Context.LAYOUT_INFLATER_SERVICE
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.Rect
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.*
import android.media.ImageReader.OnImageAvailableListener
import android.net.Uri
import android.os.*
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.Fragment
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.view.animation.AlphaAnimation
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat.getSystemService
import kotlinx.android.synthetic.main.tfe_pn_activity_posenet.*
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import org.tensorflow.lite.examples.posenet.lib.BodyPart  // posenet 라이브러리의 클래스 사용
import org.tensorflow.lite.examples.posenet.lib.KeyPoint
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import kotlin.math.acos
import kotlin.math.atan
import kotlin.math.atan2

class PosenetActivity :
  Fragment(),
  ActivityCompat.OnRequestPermissionsResultCallback {
//  /** count 변수 */
//  var count: Int = 0
//  /** count Flag */
//  var countFlag: Int = 0
  /** 스쿼트 Count */
  var squartCnt: Int = 0
  /** 스쿼트 Flag */
  var squart_FLAG: Int = 0

  /** videoView 세팅 */
  private var videoView: VideoView? = null

  /** 음악재생 */
  var mSoundPool: SoundPool? = null
  var mTestStreamId: Int = 0

  /** 애니메이션 */
//  var animAppear: Animation? = AnimationUtils.loadAnimation(activity, R.anim.fade_in)
//  var animDisappear: Animation? = AnimationUtils.loadAnimation(activity, R.anim.fade_out)
//  var startAnimation: Animation? = AlphaAnimation(1.0f, 0.1f)

  /** MediaController 추가 */
  var mediaController: MediaController? = null

  /** List of body joints that should be connected.    */
  private val bodyJoints = listOf(
    Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
    Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
    Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
    Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
    Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
    Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
    Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
    Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
    Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
  )

  /** Threshold for confidence score. */
  private val minConfidence = 0.5

  /** Radius of circle used to draw keypoints.  */
  private val circleRadius = 8.0f

  /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
  private var paint = Paint()
  private var paint2 = Paint()

  /** A shape for extracting frame data.   */
//  private var PREVIEW_WIDTH = 1280
//  private var PREVIEW_HEIGHT = 720
  private var PREVIEW_WIDTH = 640
  private var PREVIEW_HEIGHT = 480

  /** An object for the Posenet library.    */
  private lateinit var posenet: Posenet  // posenet 라이브러리의 클래스 사용

  /** ID of the current [CameraDevice].   */
  private var cameraId: String? = null

  /** A [SurfaceView] for camera preview.   */
  private var surfaceView: SurfaceView? = null

  /** A [CameraCaptureSession] for camera preview.   */
  private var captureSession: CameraCaptureSession? = null

  /** A reference to the opened [CameraDevice].    */
  private var cameraDevice: CameraDevice? = null

  /** The [android.util.Size] of camera preview.  */
  private var previewSize: Size? = null

  /** The [android.util.Size.getWidth] of camera preview. */
  private var previewWidth = 0

  /** The [android.util.Size.getHeight] of camera preview.  */
  private var previewHeight = 0

  /** A counter to keep count of total frames.  */
  private var frameCounter = 0

  /** An IntArray to save image data in ARGB8888 format  */
  private lateinit var rgbBytes: IntArray

  /** A ByteArray to save image data in YUV format  */
  private var yuvBytes = arrayOfNulls<ByteArray>(3)

  /** An additional thread for running tasks that shouldn't block the UI.   */
  private var backgroundThread: HandlerThread? = null

  /** A [Handler] for running tasks in the background.    */
  private var backgroundHandler: Handler? = null

  /** An [ImageReader] that handles preview frame capture.   */
  private var imageReader: ImageReader? = null

  /** [CaptureRequest.Builder] for the camera preview   */
  private var previewRequestBuilder: CaptureRequest.Builder? = null

  /** [CaptureRequest] generated by [.previewRequestBuilder   */
  private var previewRequest: CaptureRequest? = null

  /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
  private val cameraOpenCloseLock = Semaphore(1)

  /** Whether the current camera device supports Flash or not.    */
  private var flashSupported = false

  /** Orientation of the camera sensor.   */
  private var sensorOrientation: Int? = null

  /** Abstract interface to someone holding a display surface.    */
  private var surfaceHolder: SurfaceHolder? = null

  /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
  private val stateCallback = object : CameraDevice.StateCallback() {

    override fun onOpened(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      this@PosenetActivity.cameraDevice = cameraDevice
      createCameraPreviewSession()
    }

    override fun onDisconnected(cameraDevice: CameraDevice) {
      cameraOpenCloseLock.release()
      cameraDevice.close()
      this@PosenetActivity.cameraDevice = null
    }

    override fun onError(cameraDevice: CameraDevice, error: Int) {
      onDisconnected(cameraDevice)
      this@PosenetActivity.activity?.finish()
    }
  }

  /**
   * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
   */
  private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
    override fun onCaptureProgressed(
      session: CameraCaptureSession,
      request: CaptureRequest,
      partialResult: CaptureResult
    ) {
    }

    override fun onCaptureCompleted(
      session: CameraCaptureSession,
      request: CaptureRequest,
      result: TotalCaptureResult
    ) {
    }
  }

  /**
   * Shows a [Toast] on the UI thread.
   *
   * @param text The message to show
   */
  private fun showToast(text: String) {
    val activity = activity
    activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    activity?.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT)
//    activity?.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE)
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?
  ): View? = inflater.inflate(R.layout.tfe_pn_activity_posenet, container, false)

  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    surfaceView = view.findViewById(R.id.surfaceView)
    surfaceHolder = surfaceView!!.holder
    videoView = view.findViewById(R.id.videoView)
//    PosenetActivity.init
//    val rotation: Int ?= activity?.getWindowManager()?.getDefaultDisplay()?.getRotation()
//    var degrees : Int = 0;
//    when (rotation){
//      Surface.ROTATION_0 -> degrees = 0
//      Surface.ROTATION_90 -> degrees = 90
//      Surface.ROTATION_180 -> degrees = 180
//      Surface.ROTATION_270 -> degrees = 270
//    }
//    val result : Int = (90 - degrees +360) % 360
//    cameraDevice.setDisplayOrientation(result)
    // 이미지 붙이기
    val startIv = ImageView(this.activity)
    startIv.layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    startIv.setImageResource(R.drawable.start_img)
    val imgFrameLayout: FrameLayout = imgFrame
    imgFrameLayout.addView(startIv)
    if (startIv.visibility == View.INVISIBLE || startIv.visibility == View.GONE){
      startIv.visibility = View.VISIBLE
    }
    // fade out 애니메이션
    val startOutAnim = AnimationUtils.loadAnimation(this.activity,
      R.anim.fade_out
    )
    imgFrameLayout.animation = startOutAnim
    // fade out (3초) 후 이미지 사라지게
    Handler().postDelayed({
      startIv.visibility = View.GONE
    }, 3000)

    mediaController = MediaController(this.activity)
    mediaController!!.setAnchorView(videoView)
    val videouri: Uri = Uri.parse("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")
    videoView!!.setMediaController(mediaController)
    videoView!!.setVideoURI(videouri)
    videoView!!.start()
  }

  override fun onResume() {
    super.onResume()
    startBackgroundThread()
    mSoundPool!!.resume(mTestStreamId)

    // 추가
    // When the screen is turned off and turned back on, the SurfaceTexture is already
    // available, and "onSurfaceTextureAvailable" will not be called. In that case, we can open
    // a camera and start preview from here (otherwise, we wait until the surface is ready in
    // the SurfaceTextureListener).
//    if (textureView.isAvailable) {
//      openCamera(textureView.width, textureView.height)
//    } else {
//      textureView.surfaceTextureListener = surfaceTextureListener
//    }
  }

  override fun onStart() {
    super.onStart()
    openCamera()
    posenet = Posenet(this.context!!) // posenet 라이브러리의 클래스 사용

    /** Sound 세팅 */
    // 객체 생성
    mSoundPool = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
      val mAudioAttributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_GAME)
        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
        .build()
      SoundPool.Builder()
        .setAudioAttributes(mAudioAttributes)
        .setMaxStreams(2).build()
    } else {
      SoundPool(2, AudioManager.STREAM_MUSIC, 0)
    }
    // 파일 재생
    val mTestSoundId = mSoundPool!!.load(this.context!!,
      R.raw.test, 1)
    // 리스너 (파일로딩 후 재생 위함)
    mSoundPool!!.setOnLoadCompleteListener { soundPool, i, status ->
      mTestStreamId = mSoundPool!!.play(mTestSoundId, 0.1f, 0.1f, 1, -1, 1f)
    }
  }

//  override fun onConfigurationChanged(newConfig: Configuration) {
//    super.onConfigurationChanged(newConfig)
//    Log.d("onConfigurationChanged" , "ConConfigurationChanged");
//    if (newConfig.orientation == Configuration.ORIENTATION_PORTRAIT){
//      Log.d("onConfigurationChanged" , "Configuration.ORIENTATION_PORTRAIT");
////      Toast.makeText(activity, "세로모드", Toast.LENGTH_SHORT).show();
//    }
//    else if (newConfig.orientation == Configuration.ORIENTATION_LANDSCAPE){
//      Log.d("onConfigurationChanged" , "Configuration.ORIENTATION_LANDSCAPE");
////      Toast.makeText(activity, "가로모드", Toast.LENGTH_SHORT).show();
//    }
//  }

  override fun onPause() {
    closeCamera()
    stopBackgroundThread()
    super.onPause()
    mSoundPool!!.pause(mTestStreamId)
  }

  override fun onStop() {
    super.onStop()
    mSoundPool!!.stop(mTestStreamId)
  }

  override fun onDestroy() {
    super.onDestroy()
    posenet.close()
    mSoundPool!!.release()
  }

  private fun requestCameraPermission() {
    if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
      ConfirmationDialog().show(childFragmentManager,
        FRAGMENT_DIALOG
      )
    } else {
      requestPermissions(arrayOf(Manifest.permission.CAMERA),
        REQUEST_CAMERA_PERMISSION
      )
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ) {
    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (allPermissionsGranted(grantResults)) {
        ErrorDialog.newInstance(
          getString(R.string.tfe_pn_request_permission)
        )
          .show(childFragmentManager,
            FRAGMENT_DIALOG
          )
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
  }

  private fun allPermissionsGranted(grantResults: IntArray) = grantResults.all {
    it == PackageManager.PERMISSION_GRANTED
  }


  /**
   * Sets up member variables related to camera.
   */
  private fun setUpCameraOutputs() {
    val activity = activity
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      for (cameraId in manager.cameraIdList) {
        val characteristics = manager.getCameraCharacteristics(cameraId)

        // 전면 카메라 / 후면 카메라
        // We don't use a front( BACK -> front 으로 변경 ) facing camera in this sample.
        val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
        if (cameraDirection != null &&
          cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
//          cameraDirection == CameraCharacteristics.LENS_FACING_BACK
        ) {
          continue
        }

        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) as StreamConfigurationMap
        var largestPreviewSize: Size = map!!.getOutputSizes(ImageFormat.JPEG)[0]
        previewWidth = largestPreviewSize.width
        previewHeight = largestPreviewSize.height
//        Log.d("w,h", "$PREVIEW_WIDTH $PREVIEW_HEIGHT")

//        setAspectRatioTextureView(largestPreviewSize.height, largestPreviewSize.width)

        previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)
//        previewSize = Size(previewWidth, previewHeight)

        previewWidth = previewSize!!.width
        previewHeight = previewSize!!.height

        Log.d("w,h", "$PREVIEW_WIDTH $PREVIEW_HEIGHT")
        Log.d("newwh", "$previewWidth $previewHeight")
        imageReader = ImageReader.newInstance(
          PREVIEW_WIDTH, PREVIEW_HEIGHT,
          ImageFormat.JPEG, /*maxImages*/ 1
        )

        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

        // Initialize the storage bitmaps once when the resolution is known.
        rgbBytes = IntArray(PREVIEW_WIDTH * PREVIEW_HEIGHT)

        // Check if the flash is supported.
        flashSupported =
          characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

        this.cameraId = cameraId

        // We've found a viable camera and finished setting up member variables,
        // so we don't need to iterate through other available cameras.
        return
      }
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: NullPointerException) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      ErrorDialog.newInstance(
        getString(R.string.tfe_pn_camera_error)
      )
        .show(childFragmentManager,
          FRAGMENT_DIALOG
        )
    }
  }

  /**
   * Opens the camera specified by [PosenetActivity.cameraId].
   */
  private fun openCamera() {
    val permissionCamera = getContext()!!.checkPermission(
      Manifest.permission.CAMERA, Process.myPid(), Process.myUid()
    )
    if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
      requestCameraPermission()
    }
    setUpCameraOutputs()
    val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    try {
      // Wait for camera to open - 2.5 seconds is sufficient
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw RuntimeException("Time out waiting to lock camera opening.")
      }
      if (cameraId == null){
        val intent = Intent(activity, CameraActivity::class.java)
        startActivity(intent);  // 카메라 권한 없어 재시작
      }
      manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera opening.", e)
    }
  }

  /**
   * Closes the current [CameraDevice].
   */
  private fun closeCamera() {
    if (captureSession == null) {
      return
    }

    try {
      cameraOpenCloseLock.acquire()
      captureSession!!.close()
      captureSession = null
      cameraDevice!!.close()
      cameraDevice = null
      imageReader!!.close()
      imageReader = null
    } catch (e: InterruptedException) {
      throw RuntimeException("Interrupted while trying to lock camera closing.", e)
    } finally {
      cameraOpenCloseLock.release()
    }
  }

  /**
   * Starts a background thread and its [Handler].
   */
  private fun startBackgroundThread() {
    backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
    backgroundHandler = Handler(backgroundThread!!.looper)
  }

  /**
   * Stops the background thread and its [Handler].
   */
  private fun stopBackgroundThread() {
    backgroundThread?.quitSafely()
    try {
      backgroundThread?.join()
      backgroundThread = null
      backgroundHandler = null
    } catch (e: InterruptedException) {
      Log.e(TAG, e.toString())
    }
  }

  /** Fill the yuvBytes with data from image planes.   */
  private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
    // Row stride is the total number of bytes occupied in memory by a row of an image.
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (i in planes.indices) {
      val buffer = planes[i].buffer
      if (yuvBytes[i] == null) {
        yuvBytes[i] = ByteArray(buffer.capacity())
      }
      buffer.get(yuvBytes[i]!!)
    }
  }

  /** A [OnImageAvailableListener] to receive frames as they are available.  */
  private var imageAvailableListener = object : OnImageAvailableListener {
    override fun onImageAvailable(imageReader: ImageReader) {
      // We need wait until we have some size from onPreviewSizeChosen
      if (previewWidth == 0 || previewHeight == 0) {
        return
      }

      val image = imageReader.acquireLatestImage() ?: return
      fillBytes(image.planes, yuvBytes)
      Log.d("imyuv", yuvBytes[0].toString())
      ImageUtils.convertYUV420ToARGB8888(
        yuvBytes[0]!!,
        yuvBytes[1]!!,
        yuvBytes[2]!!,
        previewWidth,
        previewHeight,
        /*yRowStride=*/ image.planes[0].rowStride,
        /*uvRowStride=*/ image.planes[1].rowStride,
        /*uvPixelStride=*/ image.planes[1].pixelStride,
        rgbBytes
      )

      // Create bitmap from int array
      val imageBitmap = Bitmap.createBitmap(
        rgbBytes, previewWidth, previewHeight,
        Bitmap.Config.ARGB_8888
      )

      // Create rotated version for portrait display -> landscape로 변경
      val rotateMatrix = Matrix()
//      rotateMatrix.postRotate(ORIENTATIONS.get(Surface.ROTATION_90).toFloat()) // landscape
//      rotateMatrix.setScale(-1.0f, 1.0f)  // (전면) 좌우반전

      rotateMatrix.postRotate(90.0f)

      val rotatedBitmap = Bitmap.createBitmap(
        imageBitmap, 0, 0, previewWidth, previewHeight,
        rotateMatrix, true
      )
      image.close()

      processImage(rotatedBitmap)
    }
  }

  /** Crop Bitmap to maintain aspect ratio of model input.   */
  private fun cropBitmap(bitmap: Bitmap): Bitmap {
    val bitmapRatio = bitmap.height.toFloat() / bitmap.width
    val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
    var croppedBitmap = bitmap

    // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
    val maxDifference = 1e-5

    // Checks if the bitmap has similar aspect ratio as the required model input.
    when {
      abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
      modelInputRatio < bitmapRatio -> {
        // New image is taller so we are height constrained.
        val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          0,
          (cropHeight / 2).toInt(),
          bitmap.width,
          (bitmap.height - cropHeight).toInt()
        )
      }
      else -> {
        val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
        croppedBitmap = Bitmap.createBitmap(
          bitmap,
          (cropWidth / 2).toInt(),
          0,
          (bitmap.width - cropWidth).toInt(),
          bitmap.height
        )
      }
    }
    return croppedBitmap
  }

  /** Set the paint color and size.    */
  private fun setPaint() {
    paint.color = Color.MAGENTA
    paint.textSize = 60.0f
    paint.strokeWidth = 8.0f

    paint2.color = Color.GREEN
    paint2.textSize = 120.0f
    paint2.strokeWidth = 10.0f
  }

  /** Draw bitmap on Canvas.   */
  private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
    canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
    // Draw `bitmap` and `person` in square canvas.
    val screenWidth: Int
    val screenHeight: Int
    val left: Int
    val right: Int
    val top: Int
    val bottom: Int
    if (canvas.height > canvas.width) {
      screenWidth = canvas.width
      screenHeight = canvas.width
      left = 0
      top = (canvas.height - canvas.width) / 2
    } else {
      screenWidth = canvas.height
      screenHeight = canvas.height
      left = (canvas.width - canvas.height) / 2
      top = 0
    }
    right = left + screenWidth
    bottom = top + screenHeight

    setPaint()
    canvas.drawBitmap(
      bitmap,
      Rect(0, 0, bitmap.width, bitmap.height),
      Rect(left, top, right, bottom),
      paint
    )

    val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
    val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

    val squartCntObj =
      SquartObj(
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
      )
    // Draw key points over the image.
    for (keyPoint in person.keyPoints) {
      if (keyPoint.score > minConfidence) {
        val position = keyPoint.position
        val adjustedX: Float = position.x.toFloat() * widthRatio + left
        val adjustedY: Float = position.y.toFloat() * heightRatio + top
//        Log.d("partXXX", position.x.toString() + ", " + adjustedY.toString() + " / " + keyPoint.bodyPart.toString())
        squartCntObj.setAdjustedparts(keyPoint, adjustedX, adjustedY) // 좌표 세팅 for 스쿼트
        canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
      }

      // 스쿼트
//      count = AnalExercise().squartCnt
//      countFlag = AnalExercise().squart_FLAG
      val cnt = AnalExercise()
        .squartCount(squartCntObj, squart_FLAG)
      if (cnt == 1){
        squart_FLAG = 1
      } else if (cnt == 0){
        squartCnt++
        squart_FLAG = 0
      }
    }

    for (line in bodyJoints) {
      if (
        (person.keyPoints[line.first.ordinal].score > minConfidence) and
        (person.keyPoints[line.second.ordinal].score > minConfidence)
      ) {
        Log.d("person", person.keyPoints[line.first.ordinal].position.x.toFloat().toString())
        canvas.drawLine(
          person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
          person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
          person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
          paint2
        )
      }
    }

    canvas.drawText(
      "Score: %.2f".format(person.score),
      (15.0f * widthRatio),
      (30.0f * heightRatio + bottom*0.7f),
      paint
    )
    canvas.drawText(
      "Device: %s".format(posenet.device),
      (15.0f * widthRatio),
      (50.0f * heightRatio + bottom*0.7f),
      paint
    )
    canvas.drawText(
      "Time: %.2f ms".format(posenet.lastInferenceTimeNanos * 1.0f / 1_000_000),
      (15.0f * widthRatio),
      (70.0f * heightRatio + bottom*0.7f),
      paint
    )

//    "Squart Count: %d".format(squartCnt),
//    (0.6f * right),
    canvas.drawText(
      "Squart Count: %d".format(squartCnt),
      (0.1f * right),
      (top + 80.0f),
      paint
    )

    // Draw!
    surfaceHolder!!.unlockCanvasAndPost(canvas)
  }

  /** Process image using Posenet library.   */
  private fun processImage(bitmap: Bitmap) {
    // Crop bitmap.
    val croppedBitmap = cropBitmap(bitmap)

    // Created scaled version of bitmap for model input.
    val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap,
      MODEL_WIDTH,
      MODEL_HEIGHT, true)

    // Perform inference.
    val person = posenet.estimateSinglePose(scaledBitmap)
    val canvas: Canvas = surfaceHolder!!.lockCanvas()
    draw(canvas, person, scaledBitmap)
  }

  /**
   * Creates a new [CameraCaptureSession] for camera preview.
   */
  private fun createCameraPreviewSession() {
    try {
      // We capture images from preview in YUV format.
      imageReader = ImageReader.newInstance(
        previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
      )
      imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

      // This is the surface we need to record images for processing.
      val recordingSurface = imageReader!!.surface

      // We set up a CaptureRequest.Builder with the output Surface.
      previewRequestBuilder = cameraDevice!!.createCaptureRequest(
        CameraDevice.TEMPLATE_PREVIEW
      )
      previewRequestBuilder!!.addTarget(recordingSurface)

      // Here, we create a CameraCaptureSession for camera preview.
      cameraDevice!!.createCaptureSession(
        listOf(recordingSurface),
        object : CameraCaptureSession.StateCallback() {
          override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
            // The camera is already closed
            if (cameraDevice == null) return

            // When the session is ready, we start displaying the preview.
            captureSession = cameraCaptureSession
            try {
              // Auto focus should be continuous for camera preview.
              previewRequestBuilder!!.set(
                CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
              )
              // Flash is automatically enabled when necessary.
              setAutoFlash(previewRequestBuilder!!)

              // Finally, we start displaying the camera preview.
              previewRequest = previewRequestBuilder!!.build()
              captureSession!!.setRepeatingRequest(
                previewRequest!!,
                captureCallback, backgroundHandler
              )
            } catch (e: CameraAccessException) {
              Log.e(TAG, e.toString())
            }
          }

          override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
            showToast("Failed")
          }
        },
        null
      )
    } catch (e: CameraAccessException) {
      Log.e(TAG, e.toString())
    }
  }

  private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
    if (flashSupported) {
      requestBuilder.set(
        CaptureRequest.CONTROL_AE_MODE,
        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
      )
    }
  }

  /** squartclass */
  class SquartObj (
    var squartLHIPx: Float,
    var squartLHIPy: Float,
    var squartRHIPx: Float,
    var squartRHIPy: Float,
    var squartLKNEEx: Float,
    var squartLKNEEy: Float,
    var squartRKNEEx: Float,
    var squartRKNEEy: Float,
    var squartLANKx: Float,
    var squartLANKy: Float,
    var squartRANKx: Float,
    var squartRANKy: Float) {
    var legLANG: Double = 0.0
    var legRANG: Double = 0.0
    var absLegVal: Double = 0.0
    var aLegLANG: Double = 0.0
    var aLegRANG: Double = 0.0
    fun setAdjustedparts(keyPoint: KeyPoint, adjustedX: Float, adjustedY: Float){
      when (keyPoint.bodyPart) {
        BodyPart.LEFT_HIP -> {
          squartLHIPx = adjustedX
          squartLHIPy = adjustedY
        }
        BodyPart.RIGHT_HIP -> {
          squartRHIPx = adjustedX
          squartRHIPy = adjustedY
        }
        BodyPart.LEFT_KNEE -> {
          squartLKNEEx = adjustedX
          squartLKNEEy = adjustedY
        }
        BodyPart.RIGHT_KNEE -> {
          squartRKNEEx = adjustedX
          squartRKNEEy = adjustedY
        }
        BodyPart.LEFT_ANKLE -> {
          squartLANKx = adjustedX
          squartLANKy = adjustedY
        }
        BodyPart.RIGHT_ANKLE -> {
          squartRANKx = adjustedX
          squartRANKy = adjustedY
        }
        else -> {
          // nothing for now
        }
      }
    }
    fun setLegAngle(){
      if ((squartLHIPx * squartLHIPy * squartLKNEEx * squartLKNEEy * squartLANKx * squartLANKy) > 0){
        legLANG = AnalExercise()
          .calLegAngle(squartLHIPx, squartLHIPy, squartLKNEEx, squartLKNEEy, squartLANKx, squartLANKy)
      }
      if ((squartRHIPx * squartRHIPy * squartRKNEEx * squartRKNEEy * squartRANKx * squartRANKy) > 0){
        legRANG = AnalExercise()
          .calLegAngle(squartRHIPx, squartRHIPy, squartRKNEEx, squartRKNEEy, squartRANKx, squartRANKy)
      }
      absLegVal = Math.abs(legLANG)
      aLegLANG = if (absLegVal>180) 360 - absLegVal else absLegVal
      absLegVal = Math.abs(legRANG)
      aLegRANG = if (absLegVal>180) 360 - absLegVal else absLegVal
    }
  }

  class AnalExercise {
//    /** 스쿼트 Count */
//    var squartCnt: Int = 0
//    /** 스쿼트 Flag */
//    var squart_FLAG: Int = 0

    fun calLegAngle(x1: Float, y1: Float, ox: Float, oy: Float, x2: Float, y2: Float): Double {
//      val numerator: Float = ((x1-ox)*(y2-oy) + (y1-oy)*(x2-ox))
//      val denominator: Float = ((x1-ox)*(x2-ox) - (y1-oy)*(y2-oy))
//      val ratio: Float = numerator/denominator
//      val angleRed: Float = atan(ratio)

      val angleRed: Float = atan2(y1-oy, x1-ox) - atan2(y2-oy, x2-ox)

//      val numerator: Float = x1*x2+y1*y2  // 내적
//      val denominator: Double = Math.sqrt(Math.pow(x1.toDouble(), 2.0)+Math.pow(y1.toDouble(), 2.0)) * Math.sqrt(Math.pow(x2.toDouble(), 2.0)*Math.pow(y2.toDouble(), 2.0))  // 크기 곱
//      val ratio: Double = numerator/denominator
//      val angleRed: Double = acos(ratio)

      Log.d("atan2-atan2", (atan2(y1-oy, x1-ox)*180 / Math.PI).toString() +" / " +(atan2(y2-oy, x2-ox)*180 / Math.PI).toString() + " // " + (angleRed*180 / Math.PI))
      return angleRed*180 / Math.PI
    }

    fun squartRecog(squartLHIPy: Float, squartLKNEEy: Float, squartRHIPy: Float, squartRKNEEy: Float, aLegLANG: Double, aLegRANG: Double, squart_FLAG: Int): Int {
      var changingFLAG: Int = -1
      Log.d("herere", squartLHIPy.toString()+"/"+squartLKNEEy.toString()+" "+squart_FLAG.toString())

      if (squartLHIPy != 0.0f && squartLKNEEy != 0.0f) {
        val diffL = squartLHIPy - squartLKNEEy
        val diffR = squartRHIPy - squartRKNEEy
        if ((diffL > -40 || diffR > -40) && (aLegLANG <= 110 || aLegRANG <= 110) && squart_FLAG == 0) { // 스쿼트
//        if ((result > -20) && squart_FLAG == 0) { // 스쿼트
          Log.d("cal", diffL.toString())
          changingFLAG = 1
          Log.d("result", "yes!");
          return changingFLAG
        } else if ((diffL < -40 && diffR < -40) && (aLegLANG > 135 && aLegRANG > 135) && squart_FLAG == 1) {  // 일어나면 카운트
//        } else if ((result < -20) && squart_FLAG == 1) {  // 일어나면 카운트
          changingFLAG = 0
          Log.d("result2", squart_FLAG.toString()+" you did it");
          return changingFLAG
        }
      }
      return changingFLAG // -1
    }

    fun squartCount(squartCntObj: SquartObj, squart_FLAG: Int): Int {
      Log.d("partX", squartCntObj.squartLHIPy.toString() + " / " + squartCntObj.squartLHIPy.toString())
      squartCntObj.setLegAngle()
//      Log.d("posparts", (squartCntObj.squartLHIPx * squartCntObj.squartLHIPy * squartCntObj.squartLKNEEx * squartCntObj.squartLKNEEy * squartCntObj.squartLANKx * squartCntObj.squartLANKy).toString()+ "   "+ legLANG.toString() + " ,, " + legRANG.toString())

//      Log.d("pospartsNew", (squartLHIPx * squartLHIPy * squartLKNEEx * squartLKNEEy * squartLANKx * squartLANKy).toString()+ "   "+ aLegLANG.toString() + " ,, " + aLegRANG.toString())

      if (squartCntObj.aLegLANG > 0 && squartCntObj.aLegRANG > 0){
//        Log.d("angles",  squartLHIPx.toString() +" "+squartLHIPy.toString() +" , " + squartLANKx + " "+squartLANKy +" /ang "+ aLegLANG.toString() + " // "
//                + squartRHIPx + " " + squartRHIPy + " , " + squartRANKx + " " + squartRANKy + " /ang " + aLegRANG.toString())
        Log.d("pospartsNew", (squartCntObj.squartLHIPx * squartCntObj.squartLHIPy * squartCntObj.squartLKNEEx * squartCntObj.squartLKNEEy * squartCntObj.squartLANKx * squartCntObj.squartLANKy).toString()+ "   "+ squartCntObj.aLegLANG.toString() + " ,, " + squartCntObj.aLegRANG.toString())

        val countResult = squartRecog(
          squartCntObj.squartLHIPy,
          squartCntObj.squartLKNEEy,
          squartCntObj.squartRHIPy,
          squartCntObj.squartRKNEEy,
          squartCntObj.aLegLANG,
          squartCntObj.aLegRANG,
          squart_FLAG
        )  // 분석함수

        Log.d("countRes", countResult.toString())
        if (squart_FLAG == 0 && countResult == 1) {
//          squart_FLAG = 1
          return 1;
        } else if (squart_FLAG == 1 && countResult == 0){
//          squartCnt++;
//          Log.d("squartCnt", squartCnt.toString())
//          squart_FLAG = 0
          return 0;
        }
      }
      return -1; // default return
    }
  }

  /**
   * Shows an error message dialog.
   */
  class ErrorDialog : DialogFragment() {

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
      AlertDialog.Builder(activity)
        .setMessage(arguments!!.getString(ARG_MESSAGE))
        .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
        .create()

    companion object {

      @JvmStatic
      private val ARG_MESSAGE = "message"

      @JvmStatic
      fun newInstance(message: String): ErrorDialog = ErrorDialog()
        .apply {
        arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
      }
    }
  }

  companion object {
    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private val ORIENTATIONS = SparseIntArray()
    private val FRAGMENT_DIALOG = "dialog"
    // 코틀린의 생성자(초기화 프로그램 블록) - 속성 초기화
    init {
      ORIENTATIONS.append(Surface.ROTATION_0, 90)
      ORIENTATIONS.append(Surface.ROTATION_90, 0)
      ORIENTATIONS.append(Surface.ROTATION_180, 270)
      ORIENTATIONS.append(Surface.ROTATION_270, 180)
    }

    /**
     * Tag for the [Log].
     */
    private const val TAG = "PosenetActivity"
  }
}
