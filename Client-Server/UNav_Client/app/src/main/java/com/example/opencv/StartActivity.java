package com.example.opencv;

import android.content.Intent;
import android.graphics.Typeface;
import android.graphics.drawable.AnimationDrawable;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.LinearInterpolator;
import android.view.animation.RotateAnimation;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RelativeLayout;

import androidx.appcompat.app.AppCompatActivity;
import java.util.Locale;

public class StartActivity extends AppCompatActivity {
    Handler mHandler = new Handler();
    private TextToSpeech mTTS;
    private EditText server_id, port_id,localization_interval;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
        RelativeLayout relativeLayout=findViewById(R.id.startactivity);
        AnimationDrawable animationDrawable= (AnimationDrawable) relativeLayout.getBackground();
        animationDrawable.setEnterFadeDuration(2500);
        animationDrawable.setExitFadeDuration(2500);
        animationDrawable.start();
        Button start_button = (Button) findViewById(R.id.start_navigation) ;
        setTitle("UNav");

        View logo=findViewById(R.id.logo);

        Runnable runnable0=new Runnable() {
            @Override
            public void run() {
                logo.animate().rotationYBy(360f).setDuration(2000).withEndAction(this).setInterpolator(new LinearInterpolator()).start();
            }
        };

        Runnable runnable=new Runnable() {
            @Override
            public void run() {
                logo.animate().rotationYBy(360f).alpha(1f).scaleXBy(-.5f).scaleYBy(-.5f).translationYBy(1200.5f).translationXBy(278.5f).withEndAction(runnable0).setDuration(750).setInterpolator(new LinearInterpolator()).start();
            }
        };
        logo.setOnClickListener((position) -> {
            logo.animate().rotationXBy(360f).alpha(.5f).scaleXBy(.5f).scaleYBy(.5f).translationYBy(-1200.5f).translationXBy(-278.5f).withEndAction(runnable).setDuration(1000).setInterpolator(new LinearInterpolator()).start();
            });

        mTTS=new TextToSpeech(this,new TextToSpeech.OnInitListener(){

            @Override
            public void onInit(int status) {
                if (status==TextToSpeech.SUCCESS){
                    int result=mTTS.setLanguage(Locale.US);
                    float pitch=(float) 1.0,speed=(float) 2.0;
                    mTTS.setPitch(pitch);
                    mTTS.setSpeechRate(speed);
                    String message="Please enter server i,d, port i,d, localization interval and press connect button at bottom";
                    mTTS.speak(message,TextToSpeech.QUEUE_FLUSH,null);
                    if (result==TextToSpeech.LANG_MISSING_DATA || result==TextToSpeech.LANG_NOT_SUPPORTED){
                        Log.e("TTS","Language not Supported");
                    }
                } else{
                    Log.e("TTS","Initialization failed");
                }
            }
        });

        start_button.setOnClickListener((position) -> {
            server_id = (EditText) findViewById(R.id.server_id) ;
            port_id = (EditText) findViewById(R.id.port_id) ;
            localization_interval = (EditText) findViewById(R.id.localization_interval) ;
//                mHandler.post(() -> Toast.makeText(getBaseContext(), selectedItem, Toast.LENGTH_SHORT).show());
            Intent switchActivityIntent = new Intent(this, PlaceActivity.class);
            switchActivityIntent.putExtra("server_id",server_id.getText().toString());
            switchActivityIntent.putExtra("port_id",port_id.getText().toString());
            switchActivityIntent.putExtra("localization_interval", localization_interval.getText().toString());
            startActivity(switchActivityIntent);
        });

    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if (mTTS!=null){
            mTTS.stop();
            mTTS.shutdown();
        }
    }
}