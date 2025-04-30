import csv
from collections import Counter
import os

"""
ALLOWED RETURN VALUES:
    'Anger', 'Happy', 'Neutral', 'Sad', 'Surprise', None (Files marked as None will be skipped)
"""

# TEMPLATE
def decoder_template(filepath, filename):
    ... # YOUR CODE HERE
    return 'Happy'



def decode_ASVP_ESD(filepath, filename):
    '''Given the path and name of one wav file from the dataset, and returns the emotion of that file (for the ASVP-ESD dataset)'''
    # Split the filename by "-"
    parts = filename.split("-")

    if len(parts) < 10:
        return None  # If the filename doesn't follow the convention, skip it


    # Extract emotion from the filename based on the description of the dataset
    emotion_map = {
        '01': 'boredom, sigh', '02': 'neutral, calm', '03': 'happy, laugh, gaggle',
        '04': 'sad, cry', '05': 'angry, grunt, frustration', '06': 'fearful, scream, panic',
        '07': 'disgust, dislike, contempt', '08': 'surprised, gasp, amazed', '09': 'excited',
        '10': 'pleasure', '11': 'pain, groan', '12': 'disappointment, disapproval', '13': 'breath'
    }
    emotion = emotion_map.get(parts[2], 'unknown')

    # Merge emotions into defined types
    merge_mapping = {
        'angry, grunt, frustration': 'Anger',
        'disgust, dislike, contempt': None,
        'fearful, scream, panic': None,
        'happy, laugh, gaggle': 'Happy',
        'neutral, calm': 'Neutral',
        'sad, cry': 'Sad',
        'surprised, gasp, amazed': 'Surprise',
        'pain, groan': None,
        'boredom, sigh': None,
        'pleasure': None,
        'breath': None,
        'excited': None,
        'disappointment, disapproval': None,
        'unknown': None
    }
    merged_emotion = merge_mapping[emotion]

    return merged_emotion


def decode_ESD(filepath, filename):
    folder_path = os.path.dirname(filepath)
    folder_name = os.path.basename(folder_path)

    emotion_mapping = {
        "Angry": "Anger",
        "Happy": "Happy",
        "Neutral": "Neutral",
        "Sad": "Sad",
        "Surprise": "Surprise",
    }
    return emotion_mapping[folder_name]
