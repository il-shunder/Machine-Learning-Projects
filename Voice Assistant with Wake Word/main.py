import sys
from datetime import datetime

import pyttsx3 as tts
import speech_recognition

from basic_assistant import BasicAssistant


class VoiceAssistant:
    INTENTS_PATH = "intents.json"
    TODO_PATH = "todo.txt"

    def __init__(self):
        self.recognizer = speech_recognition.Recognizer()
        self.speaker = tts.init()
        self.speaker.setProperty("rate", 150)
        self.speaker.setProperty("volume", 1)

        self.mappings = {
            "get_time": self.get_time,
            "add_todo": self.add_todo,
            "speak_todo": self.speak_todo,
            "exit": self.exit_app,
        }

        self.basic_assistant = BasicAssistant(self.INTENTS_PATH, self.mappings)

    def run_assistant(self):
        is_awake = False
        while True:
            try:
                with speech_recognition.Microphone() as microphone:
                    self.recognizer.adjust_for_ambient_noise(microphone, duration=0.2)
                    audio = self.recognizer.listen(microphone)
                    sentence = self.recognizer.recognize_google(audio)
                    sentence = sentence.lower()
                    response = None

                    if sentence is not None:
                        if not is_awake:
                            if "hey assistant" in sentence:
                                is_awake = True
                                response = self.basic_assistant.get_response(sentence)
                        else:
                            response = self.basic_assistant.get_response(sentence)

                        if isinstance(response, str):
                            self.speaker.say(response)
                            self.speaker.runAndWait()
            except:
                continue

    def get_time(self):
        current_time = datetime.now().strftime("%I:%M %p")
        self.speaker.say(f"It is {current_time}")
        self.speaker.runAndWait()

    def add_todo(self):
        added = False
        self.speaker.say("What do you want to add to the to do list?")
        self.speaker.runAndWait()

        while not added:
            try:
                with speech_recognition.Microphone() as microphone:
                    self.recognizer.adjust_for_ambient_noise(microphone, duration=0.2)
                    audio = self.recognizer.listen(microphone)
                    todo_text = self.recognizer.recognize_google(audio)

                    with open(self.TODO_PATH, "a") as file:
                        file.write(todo_text + "\n")

                    added = True
                    self.speaker.say(f"I have added {todo_text} to the to do list")
                    self.speaker.runAndWait()
            except speech_recognition.UnknownValueError:
                self.speaker.say("I did not understand. Please try again")
                self.speaker.runAndWait()

    def speak_todo(self):
        todo = open(self.TODO_PATH, "r").read()
        todo_list = todo.split("\n")

        self.speaker.say("Here are the items in your list:")
        for item in todo_list:
            self.peaker.say(item)
        self.speaker.runAndWait()

    def exit_app(self):
        self.speaker.say("Bye")
        self.speaker.runAndWait()
        sys.exit(0)


voice_assistant = VoiceAssistant()
voice_assistant.run_assistant()
