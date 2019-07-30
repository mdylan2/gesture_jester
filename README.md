# Hand Gesture Recognition App
## Description
This repository contains the Python code to develop your own hand gesture recognition system. 

The app consists of 3 different modes:
1. __**Data Collection Mode:**__ Allows the user to collect train, test, or validation data on a variety of hand gestures
2. __**Model Testing Mode:**__ Test the model's ability to discern between different gestures through real-time visualizations
3. __**Music-Player/Gesture Mode:**__ Use gestures to play music, pause music, and change the volume of music

The application was designed using OpenCV, Keras, PyGame and Numpy.

## Files
Here's a list of files in the directories:
### `src`
- `demo.py`: Contains all the functions to start and run the app
- `music`: Contains the song that will be played during 'gesture mode'

## Usage
In order to start the application, do the following:
1) Clone the repo
```
git clone https://github.com/mdylan2/hand_gesture_recognition.git
```
2) Navigate into the folder, set up a virtual environment and activate it
3) Once you've activated the virtual environment, install the requirements
```
pip install -r requirements.txt
```
4) Download `my_model_weights.h5` from [this Kaggle link] and store the file in the src folder. The model was trained on data 
that I collected through the 'data collection mode' of the app. If you find that my trained model doesn't work well for you or you feel like you need more gestures,
you can train the model using images of your own hand. Please refer to the Kaggle link for more information on building or training the model.
5) Navigate into the src folder and run the following command:
```
python demo.py
```
6) More information on using the features can be seen in the application interface

## Application Interface
### `Capture Background`
![Capture Background](images/2.PNG)

### `Data Collection Mode`
![Collect Data](images/3.PNG)

### `Testing Model Mode`
![Test Model](images/2.PNG)

### `Gesture Mode (Music Player)`
![Gesture Mode](images/2.PNG)

## Questions Or Contributions
I have tried to include as much instruction for use on the app. Please let me know if you have any further questions.
And, as always I'm very open to any recommendations or contributions! Please reach out to me on Github if you would like to chat.
