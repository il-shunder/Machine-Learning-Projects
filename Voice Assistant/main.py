import sys
from datetime import datetime

import pyttsx3 as tts
import speech_recognition
from basic_assistant import BasicAssistant

INTENTS_PATH = "intents.json"
TODO_PATH = "todo.txt"

recognizer = speech_recognition.Recognizer()

speaker = tts.init()
speaker.setProperty("rate", 150)
speaker.setProperty("volume", 1)


def get_time():
    current_time = datetime.now().strftime("%I:%M %p")
    speaker.say(f"It is {current_time}")
    speaker.runAndWait()


def add_todo():
    global recognizer

    added = False
    speaker.say("What do you want to add to the to do list?")
    speaker.runAndWait()

    while not added:
        try:
            with speech_recognition.Microphone() as microphone:
                recognizer.adjust_for_ambient_noise(microphone, duration=0.2)
                audio = recognizer.listen(microphone)
                todo_text = recognizer.recognize_google(audio)

                with open(TODO_PATH, "a") as file:
                    file.write(todo_text + "\n")

                added = True
                speaker.say(f"I have added {todo_text} to the to do list")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            speaker.say("I did not understand. Please try again")
            speaker.runAndWait()


def speak_todo():
    todo = open(TODO_PATH, "r").read()
    todo_list = todo.split("\n")

    speaker.say("Here are the items in your list:")
    for item in todo_list:
        speaker.say(item)
    speaker.runAndWait()


def exit_app():
    speaker.say("Bye")
    speaker.runAndWait()
    sys.exit(0)


mappings = {"get_time": get_time, "add_todo": add_todo, "speak_todo": speak_todo, "exit": exit_app}
basic_assistant = BasicAssistant(INTENTS_PATH, mappings)
is_first = True

while True:
    try:
        with speech_recognition.Microphone() as microphone:
            recognizer.adjust_for_ambient_noise(microphone, duration=0.2)
            if is_first:
                speaker.say("Please, speak something!")
                speaker.runAndWait()
            audio = recognizer.listen(microphone)
            sentence = recognizer.recognize_google(audio)
            sentence = sentence.lower()

        response = basic_assistant.get_response(sentence)
        if isinstance(response, str):
            speaker.say(response)
            speaker.runAndWait()
        is_first = False
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
