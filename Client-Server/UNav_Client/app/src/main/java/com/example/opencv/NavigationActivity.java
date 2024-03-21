package com.example.opencv;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.SensorPrivacyManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Handler;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NavigationActivity extends Activity implements CvCameraViewListener2, OnTouchListener, SensorEventListener {

    private static String Tag="MainActivity",Tag1="IMU",Tag2 = "Camera";
    private SensorManager sensorManager;
    private Sensor linear_accelerator,gravity,gyroscope_vector;
    private TextToSpeech mTTS;
    private static int Time_interval;
    private int current_time;
    private boolean isGravitySensorPresent;
    private int past_time=(int) System.currentTimeMillis()/1000;
    public String server_id, port_id, localization_interval;
    private Vibrator v;
    private float[] acceleration= new float[3],gyroscope= new float[3];
    private static final float NS2S = 1.0f / 1000000000.0f;
    private final float[] deltaRotationVector = new float[4];
    private float thetax,thetay,thetaz;
    private float rotationx,deltarot,deltadis;
    private float timestamp;
    private ToneGenerator toneGen_rotation,toneGen_distance;
    private String finalMessage;
    private boolean waitfeedback,ishorizontal,isbeep;
    private Thread beep_rot,gravity_vib;
    private ArrayList<Action_list> action_list = new ArrayList<Action_list>();

    Socket s;
    String Place,Building,Floor,Destination;

    JavaCameraView javaCameraView;
    Mat mRgba;
    boolean touched = false;
    Handler mHandler = new Handler();

    BaseLoaderCallback mLoaderCallBack=new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Bundle extras = getIntent().getExtras();
//        server_id = extras.getString("server_id");
//        port_id = extras.getString("port_id");
        localization_interval = extras.getString("localization_interval");
        Time_interval=Integer.parseInt(localization_interval);
        Place = extras.getString("Place");
        Building = extras.getString("Building");
        Floor = extras.getString("Floor");
        Destination = extras.getString("Destination");

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);
        javaCameraView=(JavaCameraView)findViewById(R.id.java_camera_view);
        javaCameraView.enableView();
        javaCameraView.setOnTouchListener(NavigationActivity.this);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        v=(Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        toneGen_rotation=new ToneGenerator(AudioManager.STREAM_MUSIC, 100);
        toneGen_distance=new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);

        gravity_vib=new Thread(starthorizontalvibrate);

        sensorManager=(SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)!=null)
        {
            sensorManager.registerListener(this,gravity,SensorManager.SENSOR_DELAY_NORMAL);
            isGravitySensorPresent=true;
        }else {
            isGravitySensorPresent=false;
        }

        linear_accelerator=sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gravity=sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        gyroscope_vector=sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        sensorManager.registerListener(NavigationActivity.this,linear_accelerator,SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(NavigationActivity.this,gravity,SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(NavigationActivity.this,gyroscope_vector,SensorManager.SENSOR_DELAY_NORMAL);

        waitfeedback=false;
        ishorizontal=false;
        isbeep=false;
        rotationx=0;

        mTTS=new TextToSpeech(this, status -> {
            if (status==TextToSpeech.SUCCESS){
                int result=mTTS.setLanguage(Locale.US);
                float pitch=(float) 1.1,speed=(float) 1.2;
                mTTS.setPitch(pitch);
                mTTS.setSpeechRate(speed);
                mTTS.speak("Please place the phone horizontally",TextToSpeech.QUEUE_FLUSH,null);
                if (result==TextToSpeech.LANG_MISSING_DATA || result==TextToSpeech.LANG_NOT_SUPPORTED){
                    Log.e("TTS","Language not Supported");
                }
            } else{
                Log.e("TTS","Initialization failed");
            }
        });

    }
    @Override
    protected void onPause(){
        super.onPause();
        if (sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)!=null)
            sensorManager.unregisterListener(this,gravity);
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if (javaCameraView!=null) {
            javaCameraView.disableView();
        }
        if (mTTS!=null){
            mTTS.stop();
            mTTS.shutdown();
        }
        if (!s.isClosed()){
            try {
                s.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    @Override
    protected void onResume(){
        super.onResume();
        if (sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)!=null)
            sensorManager.registerListener(this,gravity,SensorManager.SENSOR_DELAY_NORMAL);
        if (OpenCVLoader.initDebug()){
            Log.i(Tag2,"Opencv loaded successfully");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            Log.i(Tag2,"Opencv not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9,this,mLoaderCallBack);
        }
    }
    @Override
    public void onCameraViewStarted(int width, int height) {

        mRgba=new Mat(height,width, CvType.CV_8UC4);
    }
    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba=inputFrame.rgba();
        current_time=(int) System.currentTimeMillis()/1000;
        if(touched|| ((current_time-past_time)%Time_interval==0)&&(!waitfeedback)&&ishorizontal) {
            waitfeedback=true;
            past_time = current_time;
            rotationx=0;
            StartSocket(mRgba);
            touched = false;
        }
        return mRgba;
    }

    static class Action_list {
        float rotation;
        float distance;
    }

    private void StartSocket(final Mat img) {
        Thread send = new Thread(() -> {
            try {
                Mat fit=new Mat(360,640,CvType.CV_8UC4);
                Size newSize = new Size(640, 360);
                Imgproc.resize(img,fit,newSize);
                MatOfByte bytemat = new MatOfByte();
                Imgcodecs.imencode(".jpg", fit, bytemat);
                byte[] bytes = bytemat.toArray();

                //--- SEND IMAGE TO SERVER ---//
                //                    s = new Socket (server_id, Integer.parseInt(port_id));
                s=PlaceActivity.getSocket();
                InputStreamReader isr=new InputStreamReader(s.getInputStream());
                BufferedReader br=new BufferedReader(isr);
                String message="";
                DataOutputStream dout = new DataOutputStream(s.getOutputStream());
                Integer bytes_len=(Integer) bytes.length;

                dout.writeInt(1);
                dout.writeInt(bytes_len);
                Log.d(Tag, String.valueOf(bytes.length));

                dout.write(bytes);
                dout.writeUTF(Place+','+Building+','+Floor+','+Destination);

                mHandler.post(() -> Toast.makeText(getBaseContext(), "Sent an image to server", Toast.LENGTH_SHORT).show());
                mTTS.speak("Start localization",TextToSpeech.QUEUE_FLUSH,null);
                message=br.readLine();
                finalMessage = message;
                char firstChar = finalMessage.charAt(0);
                if (firstChar=='[') {

                    Pattern p = Pattern.compile("[-]?[0-9]*\\.?[0-9]+");
                    Matcher m = p.matcher(finalMessage);
                    int ind;
                    ind=0;
                    while (m.find()) {
                        System.out.println(m.group());
                        if (ind % 2==0){
                            action_list.add(new Action_list());
                            action_list.get(ind/2).rotation=Float.parseFloat(m.group());}
                                else action_list.get(ind/2).distance=Float.parseFloat(m.group());
                        ind++;
                    }
                    finalMessage="Instruction updated!";
                }
                rotationx=action_list.get(0).rotation;

                mHandler.post(() -> Toast.makeText(getBaseContext(), finalMessage, Toast.LENGTH_SHORT).show());
                mTTS.speak(finalMessage,TextToSpeech.QUEUE_FLUSH,null);
                waitfeedback=false;
//                isr.close();
//                br.close();
//
//                dout.flush();
//                dout.close();
//                s.close();
            } catch (UnknownHostException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        send.start();
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.i(Tag,"onTouch event");
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touched = true;
                break;
            case MotionEvent.ACTION_UP:
                break;
            default:
                break;
        }
        return true;
    }

    private final Runnable startSoundRunnable = new Runnable() {
        @Override
        public void run() {
            toneGen_rotation.startTone(ToneGenerator.TONE_CDMA_ALERT_AUTOREDIAL_LITE, 1000);
            try {
                System.out.println("8888888888888");
                Thread.sleep(1000);
                System.out.println("777777777777");
            } catch (InterruptedException e) {
                System.out.println("999999999999999");
                e.printStackTrace();
            }

        }
    };

    private final Runnable starthorizontalvibrate = new Runnable() {
        @Override
        public void run() {
            v.vibrate(VibrationEffect.createOneShot(100,10));
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    };
    @Override
    public void onSensorChanged(SensorEvent event) {
        Sensor sensor = event.sensor;
        float x=event.values[0],y=event.values[1],z=event.values[2];
        if (sensor.getType() == Sensor.TYPE_GRAVITY) {
//            Log.d(Tag2,"Gravity Changed: X: "+x+" Y: "+y+" Z: "+z);
            if (event.values[0]<=9.6){
                ishorizontal=false;
//                if (!gravity_vib.isAlive()) {
//                    gravity_vib.start();
////            beep_rot.
//                }
            } else {
                ishorizontal=true;
            }
        }
        else if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
//            Log.d(Tag2,"Acceleration Changed: X: "+x+" Y: "+y+" Z: "+z);
            acceleration=event.values;
        }else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
//            Log.d(Tag2,"Rotation Changed: X: "+x+" Y: "+y+" Z: "+z);
            gyroscope=event.values;
            if (timestamp != 0) {
                final float dT = (event.timestamp - timestamp) * NS2S;
                // Axis of the rotation sample, not normalized yet.
                float axisX = gyroscope[0];
                float axisY = gyroscope[1];
                float axisZ = gyroscope[2];

                // Calculate the angular speed of the sample
                float omegaMagnitude = (float) Math.sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);

                // Normalize the rotation vector if it's big enough to get the axis
                // (that is, EPSILON should represent your maximum allowable margin of error)
                if (omegaMagnitude > 0) {
                    axisX /= omegaMagnitude;
                    axisY /= omegaMagnitude;
                    axisZ /= omegaMagnitude;
                }

                // Integrate around this axis with the angular speed by the timestep
                // in order to get a delta rotation from this sample over the timestep
                // We will convert this axis-angle representation of the delta rotation
                // into a quaternion before turning it into the rotation matrix.
                float thetaOverTwo = omegaMagnitude * dT / 2.0f;
                float sinThetaOverTwo = (float) Math.sin(thetaOverTwo);
                float cosThetaOverTwo = (float) Math.cos(thetaOverTwo);
                deltaRotationVector[0] = sinThetaOverTwo * axisX;
                deltaRotationVector[1] = sinThetaOverTwo * axisY;
                deltaRotationVector[2] = sinThetaOverTwo * axisZ;
                deltaRotationVector[3] = cosThetaOverTwo;
            }
            timestamp = event.timestamp;
            float[] deltaRotationMatrix = new float[9];
            SensorManager.getRotationMatrixFromVector(deltaRotationMatrix, deltaRotationVector);
            thetax=(float) Math.toDegrees(Math.atan2(deltaRotationMatrix[7],deltaRotationMatrix[8]));
            thetay=(float) Math.toDegrees(Math.atan2(-deltaRotationMatrix[6],Math.sqrt(deltaRotationMatrix[7]*deltaRotationMatrix[7]+deltaRotationMatrix[8]*deltaRotationMatrix[8])));
            thetaz=(float) Math.toDegrees(Math.atan2(deltaRotationMatrix[3],deltaRotationMatrix[0]));
//            Log.d(Tag2,"Acceleration Changed: X: "+acceleration[0]+" Y: "+acceleration[1]+" Z: "+acceleration[2]);
            rotationx+=thetax;
            Log.d(Tag,"11111111111111111111");
            beep_rot=new Thread(startSoundRunnable);
            if (!beep_rot.isAlive())
            {
                Log.d(Tag,"2222222222222222222");
                beep_rot.start();
                try {
                    beep_rot.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

//            toneGen_rotation.startTone(ToneGenerator.TONE_PROP_BEEP,(int) Math.abs(rotationx));
//            toneGen_rotation.stopTone();
//            toneGen_rotation.release();

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }
}


