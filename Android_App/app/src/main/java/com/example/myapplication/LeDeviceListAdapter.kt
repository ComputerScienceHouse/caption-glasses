package com.example.myapplication

import android.Manifest
import android.R
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.TextView
import androidx.annotation.RequiresPermission

class LeDeviceListAdapter(private val context: Context) : BaseAdapter() {

    private val devices = ArrayList<BluetoothDevice>()
    private val inflater: LayoutInflater = LayoutInflater.from(context)

    fun addDevice(device: BluetoothDevice) {
        if (!devices.contains(device)) {
            devices.add(device)
        }
    }

    fun getDevice(position: Int): BluetoothDevice {
        return devices[position]
    }

    override fun getCount(): Int = devices.size

    override fun getItem(position: Int): Any = devices[position]

    override fun getItemId(position: Int): Long = position.toLong()

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
        val view = convertView ?: inflater.inflate(R.layout.simple_list_item_2, parent, false)

        val device = devices[position]

        val nameText = view.findViewById<TextView>(R.id.text1)
        val addressText = view.findViewById<TextView>(R.id.text2)

        nameText.text = device.name ?: "Unknown Device"
        addressText.text = device.address

        return view
    }
}