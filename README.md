# Object-Detection-by-Voice-Command
Object Detection is a field of Computer Vision that detects instances of semantic objects in images/videos (by creating bounding boxes around them in our case). We can then convert the annotated text into voice responses and give the name of the objects in the person/camera’s view.


The Project has the following modules:
a)	Object Detection
For object detection, we created a large dataset consisting of all the common real-time objects with their respective labels. After creating the dataset we created a neural network  model with the help of the algorithm that runs through a variation of an extremely complex Convolutional Neural Network architecture. The convolutional  neural network is then trained using supervised learning method. Thus after training the neural network model, we have used webcam to test images in this trained model and it processes one frame at a time. Thus the model detects all the objects in the frame and label them accordingly.
b)	Text-to-Speech
After the object has been detected and their respective label have been obtained, the label is used for obtaining voice assistant. For that we have send the text description to the Google Text-to-Speech using the gTTS package. Thus in this module we obtain a voice feedback for the objects that have the highest accuracy in the frame, for example :”Person is detected”.

## To run the code 
Move to directory models\research\object_detection


Run python bot.py
