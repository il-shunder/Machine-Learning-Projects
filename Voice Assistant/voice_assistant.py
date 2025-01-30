import pyttsx3
import speech_recognition as sr

recognizer = sr.Recognizer()


with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source)
    print("Say something...")
    audio = recognizer.listen(source)


try:
    text = recognizer.recognize_google(audio)
    engine = pyttsx3.init()

    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1)

    engine.say(text)
    engine.runAndWait()

    print("You said: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
