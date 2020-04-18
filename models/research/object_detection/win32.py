from win32com.client import constants
import win32com.client
import pythoncom
def voice(Strin):
    speaker = win32com.client.Dispatch("SAPI.SpVoice")

    speaker.Speak(Strin+"is detected")
    
#voice("Welcome Jury members")