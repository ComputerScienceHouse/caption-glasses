package com.example.myapplication

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.LinkProperties
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.annotation.RequiresApi
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import androidx.core.app.ActivityCompat
import com.pedro.library.rtsp.RtspOnlyAudio
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import android.os.Handler
import android.os.Looper
import androidx.annotation.RequiresPermission
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothGattCharacteristic
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioRecord.RECORDSTATE_RECORDING
import android.media.MediaRecorder
import android.view.View.VISIBLE
import androidx.core.view.isInvisible
import androidx.core.view.isVisible
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.Response
import okio.ByteString.Companion.toByteString
import java.util.UUID

const val REQUEST_CODE = 200

val connectChecker = RtspConnectChecker()

const val SoundUUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
const val CaptionUUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
const val RECORDER_SAMPLERATE = 16000
const val RECORDER_CHANNELS: Int = 1
@RequiresApi(Build.VERSION_CODES.S)
const val RECORDER_AUDIO_ENCODING: Int = AudioFormat.ENCODING_PCM_16BIT

class MainActivity : ComponentActivity() {

    val rtspStream = RtspOnlyAudio(connectChecker)
    private var audioPermissions = arrayOf(Manifest.permission.RECORD_AUDIO)
    @RequiresApi(Build.VERSION_CODES.S)
    private var bluetoothPermissions = arrayOf(Manifest.permission.BLUETOOTH_SCAN)
    @RequiresApi(Build.VERSION_CODES.S)
    private var bluetoothConnectionPermissions = arrayOf(Manifest.permission.BLUETOOTH_CONNECT)
    private var locationPermissions = arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
    private var micPermissionGranted = false
    private var bluetoothPermissionsGranted = false
    private var bluetoothConnectionPermissionsGranted = false
    private var locationPermissionsGranted = false
    lateinit var bluetoothGatt: BluetoothGatt

    lateinit var device: BluetoothDevice

    private var scanning = false

    @SuppressLint("UnsafeOptInUsageError")
    @Serializable
    data class TextType(val type: String, val text: String, val speaker: String?)
    private val handler = Handler(Looper.getMainLooper())
    private val SCAN_PERIOD: Long = 10000


    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)

        val bluetoothManager = getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
        val bluetoothAdapter: BluetoothAdapter = bluetoothManager.adapter
        val bluetoothLeScanner = bluetoothAdapter.bluetoothLeScanner
        LeDeviceListAdapter(applicationContext)
        val textView: TextView = findViewById(R.id.textView)
        val leScanCallback: ScanCallback = object : ScanCallback() {
            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                super.onScanResult(callbackType, result)
                if (result.device != null) {
                    if (result.device.name != null) {
                        if (result.device.name.equals("CGPI")) {
                            textView.text = "PI Connected"
                            device = result.device
                            bluetoothGatt = device.connectGatt(applicationContext, false, gattCallback)

                        }
                    }
                }

            }
        }
        println(audioPermissions[0])
        bluetoothPermissionsGranted = ActivityCompat.checkSelfPermission(
            applicationContext,
            bluetoothPermissions[0]
        ) == PackageManager.PERMISSION_GRANTED
        if (!bluetoothPermissionsGranted) {
            ActivityCompat.requestPermissions(this, bluetoothPermissions, REQUEST_CODE)
        }

        bluetoothConnectionPermissionsGranted = ActivityCompat.checkSelfPermission(
            applicationContext,
            bluetoothConnectionPermissions[0]
        ) == PackageManager.PERMISSION_GRANTED
        if (!bluetoothConnectionPermissionsGranted) {
            ActivityCompat.requestPermissions(this, bluetoothConnectionPermissions, REQUEST_CODE)
        }

        locationPermissionsGranted = ActivityCompat.checkSelfPermission(
            applicationContext,
            locationPermissions[0]
        ) == PackageManager.PERMISSION_GRANTED
        if (!locationPermissionsGranted) {
            ActivityCompat.requestPermissions(this, locationPermissions, REQUEST_CODE)
        }

        scanLeDevice(leScanCallback, bluetoothLeScanner)

        val listen_url = "wss://typing-phrase-particular-survival.trycloudflare.com/ws"

        val startButton: Button = findViewById(R.id.button2)
        val stopButton: Button = findViewById(R.id.button)
        val scanButton: Button = findViewById(R.id.button3)
        val reconnectButton: Button = findViewById(R.id.button4)

        val client = OkHttpClient()
        val request = Request.Builder().url(listen_url).build()
        val listener = object: com.example.myapplication.WebSocketListener(listen_url) {
            @RequiresApi(Build.VERSION_CODES.TIRAMISU)
            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            override fun onMessage(webSocket: WebSocket, text: String) {
                var message = ""
                var deserializedText = Json.decodeFromString<TextType>(text)
                if (deserializedText.type.equals("sound")){
                    message = "[" + deserializedText.text + "]"
                    sendString(bluetoothGatt, message, SoundUUID)
                    println("Websocket Received: $message")
                } else if (deserializedText.type.equals("final") || deserializedText.type.equals("partial")) {
                    message = "[" + deserializedText.speaker + "]: " + deserializedText.text
                    sendString(bluetoothGatt, message, CaptionUUID)
                    println("Websocket Received: $message")
                }


            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                println("Websocket Failure: $response")
                runOnUiThread {
                    if (reconnectButton.isInvisible) {
                        reconnectButton.visibility = VISIBLE
                    }
                }
            }
        }

       println("Connecting to the server...")
        micPermissionGranted = ActivityCompat.checkSelfPermission(
            applicationContext,
            audioPermissions[0]
        ) == PackageManager.PERMISSION_GRANTED
        if (!micPermissionGranted) {
            ActivityCompat.requestPermissions(this, audioPermissions, REQUEST_CODE)
        }

        val audioRecord = initAudioRecorder()

        var websocket = listen(client, request, listener)

        var recordRoutine: Job? = null

        startButton.setOnClickListener { v ->
            recordRoutine = MainScope().launch {
                record(websocket, audioRecord)
            }
        }
        stopButton.setOnClickListener { v ->
            println("Stopping...")
            stop(audioRecord)
            recordRoutine?.cancel()

        }
        scanButton.setOnClickListener { v ->
            scanLeDevice(leScanCallback, bluetoothLeScanner)
        }

        reconnectButton.setOnClickListener { v ->
            websocket = listen(client, request, listener)
        }

    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    private fun scanLeDevice(leScanCallback: ScanCallback, bluetoothLeScanner: BluetoothLeScanner) {
        val bluetoothStatusText: TextView = findViewById(R.id.textView)
        if (!scanning) { // Stops scanning after a pre-defined scan period.
            handler.postDelayed(@RequiresPermission(Manifest.permission.BLUETOOTH_SCAN) {
                scanning = false
                //likely will override pi connected text. Needs testing
                bluetoothStatusText.text = "Scan Timed Out"
                bluetoothLeScanner.stopScan(leScanCallback)
            }, SCAN_PERIOD)
            scanning = true
            bluetoothStatusText.text =("Scanning...")
            bluetoothLeScanner.startScan(leScanCallback)
        } else {
            bluetoothStatusText.text =("Scan Failed")
            scanning = false
            bluetoothLeScanner.stopScan(leScanCallback)
        }
    }

    private suspend fun record(webSocket: WebSocket, audioRecord: AudioRecord) {
        val buf = ByteArray(4096)
        try {
            audioRecord.startRecording()
            while (true) {
                val byteRead = audioRecord.read(buf, 0, buf.size)
//                println(buf.toByteString(0, 4096))
                if (byteRead <= 0)
                    break
                webSocket.send(buf.toByteString(0, byteRead))
                delay(100L)
            }
        } catch (e: Exception) {
            println("Recording Error: $e")
            stop(audioRecord)
        }
    }


    fun stop(audioRecord: AudioRecord) {
        if (audioRecord.recordingState == RECORDSTATE_RECORDING){
            audioRecord.stop()
        }
    }

        @Throws(InterruptedException::class)
        fun listen(client: OkHttpClient, request: Request, listener: WebSocketListener): WebSocket {
            val websocket = client.newWebSocket(request, listener)
            return websocket
    }
}

private val gattCallback = object : BluetoothGattCallback() {

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    override fun onConnectionStateChange(
        gatt: BluetoothGatt,
        status: Int,
        newState: Int
    ) {
        if (newState == BluetoothProfile.STATE_CONNECTED) {
            println("Connected!")
            gatt.discoverServices()
        } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
            println("Disconnected")
        }
    }

    override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
    }
}

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
fun sendString(gatt: BluetoothGatt, message: String, uuid: String) {
    val service = gatt.getService(
        UUID.fromString("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
    ) ?: return

    val characteristic = service.getCharacteristic(
        UUID.fromString(uuid)
    ) ?: return

    val data = message.toByteArray(Charsets.UTF_8)

    gatt.writeCharacteristic(
        characteristic,
        data,
        BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
    )
}

@RequiresApi(Build.VERSION_CODES.S)
@RequiresPermission(Manifest.permission.RECORD_AUDIO)
fun initAudioRecorder(): AudioRecord{

    var audioRecord: AudioRecord? = null

    audioRecord = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        RECORDER_SAMPLERATE, RECORDER_CHANNELS,
        RECORDER_AUDIO_ENCODING, 4096
    )

    return audioRecord
}

//@Composable
//fun LongBasicDropdownMenu() {
//    var expanded by remember { mutableStateOf(false) }
//    // Placeholder list of 100 strings for demonstration
//    val menuItemData = List(10) { "Option ${it + 1}" }
//
//    Box(
//        modifier = Modifier
//            .padding(16.dp)
//    ) {
//        IconButton(onClick = { expanded = !expanded }) {
//
//        }
//        DropdownMenu(
//            expanded = expanded,
//            onDismissRequest = { expanded = false }
//        ) {
//            menuItemData.forEach { option ->
//                DropdownMenuItem(
//                    text = { Text(option) },
//                    onClick = { /* Do something... */ }
//                )
//            }
//        }
//    }
//}