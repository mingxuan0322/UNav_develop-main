import cv2
import socket
import pyttsx3
import argparse
import threading
import json
import speech_recognition as sr
import time
import sys
import tty
import termios
import jpysocket
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import inflect

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_id', default=None,type=str, required=True,
                        help='host id')
    parser.add_argument('--port_id', default=None,type=int, required=True,
                        help='port id')
    parser.add_argument('--camera_index', default=None, type=int, required=True,
                        help='camera index')
    parser.add_argument('--capture_interval', default=None, type=int, required=True,
                        help='image capture time interval')
    parser.add_argument('--brightness', default=None, type=int, required=True,
                        help='camera brightness')
    parser.add_argument('--contrast', default=None, type=int, required=True,
                        help='camera contrast')
    parser.add_argument('--saturation', default=None, type=int, required=True,
                        help='camera saturation')
    parser.add_argument('--speak_rate', default=None, type=int, required=True,
                        help='speaker rate')
    parser.add_argument('--volume', default=None, type=float, required=True,
                        help='speaker volume')
    parser.add_argument('--voices', default=None, type=int, required=True,
                        help='speaker voices')
    opt = parser.parse_args()
    return opt

class Audio():
    def __init__(self,speak_rate,speak_volume,voice):
        self.listner = sr.Recognizer()
        self.stopword=stopwords.words('english')

        self.engine = pyttsx3.init(driverName='espeak')
        self.engine.setProperty('rate', speak_rate)  # setting up new voice rate
        self.engine.setProperty('volume', speak_volume)  # setting up volume level  between 0 and 1
        voices = self.engine.getProperty('voices')  # getting details of current voice
        self.engine.setProperty('voice', voices[voice].id)  # changing index, changes voices. o for male

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.number2word=inflect.engine().number_to_words
    # def
    def clean_string(self,text):
        text = ''.join([word.replace('_', ' ') for word in text if word not in string.punctuation or word == '_'])
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in self.stopword])
        return text

    def cosine_sim_vectors(self,v1, v2):
        v1 = v1.reshape(1, -1)
        v2 = v2.reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]

    def retrieve_key(self,text,List):
        try:
            if text[:6]=='number':
                i=0
                while True:
                    i+=1
                    if text=='number '+self.number2word(i):
                        return List[i-1]
                    if i>100000:
                        return False
            text=self.clean_string(text)
            list=[self.clean_string(words) for words in List]
            list=[text]+list
            vectorizer = CountVectorizer().fit_transform(list)
            vector = vectorizer.toarray()
            score_max=0.9
            ind=-1
            for i in range(len(vector)-1):
                score = self.cosine_sim_vectors(vector[0], vector[i+1])
                if score>score_max:
                    score_max=score
                    ind=i
            if ind==-1:
                responds=False
            else:
                responds=List[ind]
            return responds
        except:
            return False

    def recognize_speech_from_mic(self):
        with self.microphone as source:
            self.listner.adjust_for_ambient_noise(source)
            audio_data = self.listner.listen(source)

        print("Recognizing...")
        response = {"success": True,
                    "error": None,
                    "transcription": None}
        try:
            response["transcription"] = self.listner.recognize_google(audio_data)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
        return response

    def go_back_detection(self,Type):
        self.engine.say('or go back to'+Type+ 'manual')
        self.engine.runAndWait()
        self.engine.say('say')
        self.engine.runAndWait()
        self.engine.say('go back')
        self.engine.runAndWait()
        response = self.recognize_speech_from_mic()
        pattern = response['transcription']
        return pattern

    def speak(self,i,j):
        self.engine.say(j)
        self.engine.runAndWait()
        self.engine.say('or')
        self.engine.runAndWait()
        self.engine.say('number ' + self.number2word(i + 1))
        self.engine.runAndWait()

    def get_destination(self,List):
        Places_list=list(List.keys())
        Place = False
        while not Place:
            # self.engine.runAndWait()
            s='Choose the place say'
            self.engine.say(s)
            self.engine.runAndWait()
            for i,j in enumerate(Places_list):
                self.speak(i,j)
            response = self.recognize_speech_from_mic()
            pattern = response['transcription']
            Place=self.retrieve_key(pattern,Places_list)
            if Place:
                print('Place:',Place)
                Building_List=List[Place]
                Building_list = list(Building_List.keys())
                Building=False
                while not Building:
                    self.engine.say('Choose the building say')
                    self.engine.runAndWait()
                    for i, j in enumerate(Building_list):
                        self.speak(i,j)
                    pattern=self.go_back_detection('Place')
                    if pattern=='go back':
                        break
                    else:
                        Building = self.retrieve_key(pattern, Building_list)
                        if Building:
                            print('Building:', Building)
                            Floor_List=Building_List[Building]
                            Floor_list = list(Floor_List.keys())
                            Floor = False
                            while not Floor:
                                self.engine.say('Choose the floor say')
                                self.engine.runAndWait()
                                for i, j in enumerate(Floor_list):
                                    self.speak(i,j)
                                pattern = self.go_back_detection('Building')
                                if pattern == 'go back':
                                    break
                                else:
                                    Floor = self.retrieve_key(pattern, Floor_list)
                                    if Floor:
                                        print('Floor:', Floor)
                                        Destination_list=Floor_List[Floor]
                                        Destination = False
                                        while not Destination:
                                            self.engine.say('Choose the destination say')
                                            self.engine.runAndWait()
                                            for i, j in enumerate(Destination_list):
                                                self.speak(i,j)
                                            pattern = self.go_back_detection('Floor')
                                            if pattern == 'go back':
                                                break
                                            else:
                                                Destination = self.retrieve_key(pattern, Destination_list)
                                                if Destination:
                                                    print('Destination:', Destination)
                                                    return Place+','+Building+','+Floor+','+Destination
                                                else:
                                                    self.engine.say("Didn't catch it, repeat please")
                                                    self.engine.runAndWait()
                                    else:
                                        self.engine.say("Didn't catch it, repeat please")
                                        self.engine.runAndWait()
                        else:
                            self.engine.say("Didn't catch it, repeat please")
                            self.engine.runAndWait()
            else:
                self.engine.say("Didn't catch it, repeat please")
                self.engine.runAndWait()

    def Listen(self,List):
        listed_cleaned = list(map(self.clean_string, List))
        while True:
            self.engine.runAndWait()

        # with sr.Microphone() as source:
        #     # read the audio data from the default microphone
        #     text=False
        #     speaker.speak_instructions('Choose the place you are in from the following list')
        #     for i in List:
        #         speaker.speak_instructions(i)
        #     while not text:
        #         response=self.listen(source,5)
        #         print(text)
        #         if text:
        #             estimation = List[self.find_result(text,listed_cleaned)]
        #             speaker.speak_instructions('Are you in'+estimation+'? Yes or no.')
        #             text0=False
        #             while not text0:
        #                 text0 = self.listen(source, 3)
        #                 if text0=='yes':
        #                     return estimation
        #                 elif text0=='no':
        #                     text=False
        #                     break

class Keypressed():
    def __init__(self):
        self.breakNow=False

    def getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def waitForKeyPress(self):
        while True:
            ch = self.getch()
            if ch == "q":  # Or skip this check and just break
                self.breakNow = True
                break

class Camera():
    def __init__(self,camera_index,capture_interval,brightness,contrast,saturation,communicator,keypressed,destination,audio):
        self.capture_interval=capture_interval
        self.camera_id=camera_index
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation

        self.communicator=communicator
        self.keypressed=keypressed

        self.destination=destination
        self.speaker=audio
        self.socket=None

    def camera(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
        t=time.time()
        while not self.keypressed.breakNow:
            ret, frame = cap.read()
            if time.time()-t>self.capture_interval and ret:
                message,self.socket=self.communicator.send_image(frame,self.destination)
                print(message)
                self.speaker.engine.say(message)
                self.speaker.engine.runAndWait()
                t = time.time()
        self.socket.close()
        cap.release()
        cv2.destroyAllWindows()

class Server_Communicator():
    def __init__(self,host,port):
        self.socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    def get_List(self):
        self.socket.sendall(int(0).to_bytes(4,'big'),4)
        data=self.socket.recv(1024).decode()
        data = data.replace("\'", "\"")
        return json.loads(data)
    def send_image(self,image,destination):
        self.socket.sendall(int(1).to_bytes(4, 'big'), 4)
        height,width,_=image.shape
        newsize=(640,int(height/width*640))
        image=cv2.resize(image,newsize)
        _, bytemat = cv2.imencode('.jpg', image)
        Bytes = bytemat.tostring()
        self.socket.sendall(len(Bytes).to_bytes(4, 'big'), 4)
        self.socket.sendall(Bytes)
        destination=jpysocket.jpyencode(destination)
        self.socket.send(destination)
        message = self.socket.recv(4096).decode("utf-8")
        return message,self.socket

def main():
    opt=options()
    communicator=Server_Communicator(opt.host_id,opt.port_id)
    audio = Audio(opt.speak_rate,opt.volume,opt.voices)

    Destination_list=communicator.get_List()
    dest=audio.get_destination(Destination_list)
    # Place='New_York_University'
    # Building='NYU_Langone'
    # Floor='17_DENSE_LOW'
    # Destination='JR'
    # dest=Place+','+Building+','+Floor+','+Destination


    # speaker.speak_Building(Building_list)
    keypress=Keypressed()
    keypressed=threading.Thread(target=keypress.waitForKeyPress)
    keypressed.start()

    camera = Camera(opt.camera_index, opt.capture_interval,opt.brightness, opt.contrast, opt.saturation,communicator,keypress,dest,audio)
    camera_thread=threading.Thread(target=camera.camera)
    camera_thread.start()

if __name__=='__main__':
    main()