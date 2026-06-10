package com.example.myapplication

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.logging.Logger

abstract class WebSocketListener(val url: String) : WebSocketListener() {
    /**
     * Invoked when a web socket has been accepted by the remote peer and may begin transmitting
     * messages.
     */

    override fun onOpen(webSocket: WebSocket, response: Response) {
        println("WebSocket opened: " + response.message)
    }

    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
        println("WebSocket closing: $reason (code: $code)")
    }

    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
        println("WebSocket closed: reason=$reason (code: $code)")
    }

    abstract override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?)
    abstract override fun onMessage(webSocket: WebSocket, text: String)


 //   fun onMessage(webSocket: WebSocket?, bytes: ByteString) {
//        println("Received bytes: " + bytes.hex())
//    }
}