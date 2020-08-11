package org.tensorflow.lite.examples.posenet

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.KeyEvent
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.tfe_main_activity_wv.*

class MainActivity : AppCompatActivity() {
    var mContext: Context? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.tfe_main_activity_wv)
        mContext = this.applicationContext

        var mWebSettings = webView.settings
        mWebSettings.javaScriptEnabled = true
        mWebSettings.loadWithOverviewMode = true
        mWebSettings.useWideViewPort = true
//        mWebSettings.layoutAlgorithm = WebSettings.LayoutAlgorithm.TEXT_AUTOSIZING
        mWebSettings.cacheMode = WebSettings.LOAD_NO_CACHE
        mWebSettings.domStorageEnabled = true

        webView.loadUrl("http://hij.dothome.co.kr/")
        webView.isVerticalScrollBarEnabled = true
        webView.webViewClient = WebViewClientClass()

//        savedInstanceState ?: supportFragmentManager.beginTransaction()
//            .replace(R.id.container, PosenetActivity())
//            .commit()
    }

    private inner class WebViewClientClass : WebViewClient() {
        override fun shouldOverrideUrlLoading(view: WebView?, url: String?): Boolean {
            if (url!!.startsWith("app://")){
//            if (url!!.startsWith("app://tf_test_application")){
//                val intent = Intent(mContext.applicationContext, MainActivity::class.java)
                val intent = Intent(this@MainActivity.mContext, CameraActivity::class.java)
                startActivity(intent)
                return true
            } else{
                view!!.loadUrl(url)
                return true
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if ((keyCode == KeyEvent.KEYCODE_BACK) && webView.canGoBack()){
            webView.goBack()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
}
