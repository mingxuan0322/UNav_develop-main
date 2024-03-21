package com.example.opencv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import org.json.JSONException;
import org.json.JSONObject;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.AnimationDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.LinearInterpolator;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.res.ResourcesCompat;

public class BuildingActivity extends AppCompatActivity {
    Handler mHandler = new Handler();
    private TextToSpeech mTTS;
    public String server_id, port_id, localization_interval;
    private Typeface mTypeface;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_buildings);
        RelativeLayout relativeLayout=findViewById(R.id.buildingactivity);
        AnimationDrawable animationDrawable= (AnimationDrawable) relativeLayout.getBackground();
        animationDrawable.setEnterFadeDuration(2500);
        animationDrawable.setExitFadeDuration(2500);
        animationDrawable.start();
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
        ListView place_list = (ListView) findViewById(R.id.list_place);
        ListView building_list = (ListView) findViewById(R.id.list_building);
        Bundle extras = getIntent().getExtras();
        server_id = extras.getString("server_id");
        port_id = extras.getString("port_id");
        localization_interval = extras.getString("localization_interval");
        mTypeface = ResourcesCompat.getFont(this, R.font.lato_medium);
        String Place = extras.getString("Place");
        setTitle("UNav ("+Place+")");
        String Buildingobject=extras.getString("Building_list");
        mTTS=new TextToSpeech(this,new TextToSpeech.OnInitListener(){

            @Override
            public void onInit(int status) {
                if (status==TextToSpeech.SUCCESS){
                    int result=mTTS.setLanguage(Locale.US);
                    float pitch=(float) 1.0,speed=(float) 2.0;
                    mTTS.setPitch(pitch);
                    mTTS.setSpeechRate(speed);
                    String message="Choose the building of "+Place;
                    mTTS.speak(message,TextToSpeech.QUEUE_FLUSH,null);
                    if (result==TextToSpeech.LANG_MISSING_DATA || result==TextToSpeech.LANG_NOT_SUPPORTED){
                        Log.e("TTS","Language not Supported");
                    }
                } else{
                    Log.e("TTS","Initialization failed");
                }
            }
        });

        try {
            JSONObject Building_object=new JSONObject(Buildingobject);
            String[] buildingstring = new String[]{};
            List<String> Building_list = new ArrayList<>(Arrays.asList(buildingstring));

            Iterator<String> listKEY = Building_object.keys();
            do {
                String newkey = listKEY.next();
                Building_list.add(newkey);
            } while (listKEY.hasNext());
            ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, Building_list){
                @Override
                public View getView(int position, View convertView, ViewGroup parent){
                    TextView item = (TextView) super.getView(position,convertView,parent);
                    item.setTypeface(mTypeface);
                    item.setTextColor(Color.parseColor("#ffffff"));
                    item.setTypeface(item.getTypeface(), Typeface.BOLD);
                    item.setTextSize(TypedValue.COMPLEX_UNIT_DIP,18);
                    item.setOutlineAmbientShadowColor(Color.parseColor("#ff0000"));
                    item.setShadowLayer(10,0,0,Color.parseColor("#FFFF00"));
                    return item;
                }
            };
            building_list.setAdapter(arrayAdapter);

            building_list.setOnItemClickListener((parent, view, position, id) -> {
                String selectedItem = (String) parent.getItemAtPosition(position);
//                mHandler.post(() -> Toast.makeText(getBaseContext(), selectedItem, Toast.LENGTH_SHORT).show());
                try {
                    Intent switchActivityIntent = new Intent(this, FloorActivity.class);
                    switchActivityIntent.putExtra("server_id", server_id);
                    switchActivityIntent.putExtra("port_id", port_id);
                    switchActivityIntent.putExtra("localization_interval", localization_interval);
                    switchActivityIntent.putExtra("Place",Place);
                    switchActivityIntent.putExtra("Building",selectedItem);
                    JSONObject floor_list=Building_object.getJSONObject(selectedItem);
                    switchActivityIntent.putExtra("Floor_list", floor_list.toString());
                    startActivity(switchActivityIntent);
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            });
        } catch (JSONException e) {
            e.printStackTrace();
        }


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