package com.example.myapplication

import com.pedro.common.ConnectChecker

class RtspConnectChecker : ConnectChecker {
    override fun onAuthError(){
        println("Auth Error")
    }

    override fun onConnectionStarted(url: String) {
        println("Connection Started at: $url")
    }

    override fun onConnectionSuccess() {
        println("Connection Success")
    }

    override fun onConnectionFailed(reason: String) {
        println("Connection Failed: $reason")
    }

    override fun onDisconnect() {
        println("Disconnected")
    }

    override fun onAuthSuccess(){
        println("Auth Success")
    }
}