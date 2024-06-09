from flask import Flask, render_template, request, url_for
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os
app = Flask(__name__)

# Load your trained model
model_path = 'dogs_model_vgg16.h5'
model = load_model(model_path)

# Ensure the images directory exists within static
os.makedirs("./static/images", exist_ok=True)

# Dog breed labels
dogs_labels = {
    0: "toy_terrier",
    1: "vizsla",
    2: "tibetan_terrier",
    3: "whippet",
    4: "tibetan_mastiff",
    5: "weimaraner",
    6: "wire-haired_fox_terrier",
    7: "sussex_spaniel",
    8: "yorkshire_terrier",
    9: "standard_schnauzer",
    10: "rottweiler",
    11: "scotch_terrier",
    12: "schipperke",
    13: "shetland_sheepdog",
    14: "standard_poodle",
    15: "scottish_deerhound",
    16: "samoyed",
    17: "saint_bernard",
    18: "saluki",
    19: "sealyham_terrier",
    20: "norwich_terrier",
    21: "old_english_sheepdog",
    22: "rhodesian_ridgeback",
    23: "papillon",
    24: "pembroke",
    25: "pekinese",
    26: "otterhound",
    27: "pug",
    28: "redbone",
    29: "pomeranian",
    30: "norfolk_terrier",
    31: "maltese_dog",
    32: "norwegian_elkhound",
    33: "malamute",
    34: "mexican_hairless",
    35: "miniature_schnauzer",
    36: "miniature_pinscher",
    37: "lhasa",
    38: "newfoundland",
    39: "malinois",
    40: "leonberg",
    41: "irish_wolfhound",
    42: "kuvasz",
    43: "irish_terrier",
    44: "komondor",
    45: "keeshond",
    46: "labrador_retriever",
    47: "japanese_spaniel",
    48: "kerry_blue_terrier",
    49: "irish_water_spaniel",
    50: "german_short-haired_pointer",
    51: "groenendael",
    52: "great_dane",
    53: "irish_setter",
    54: "gordon_setter",
    55: "golden_retriever",
    56: "german_shepherd",
    57: "flat-coated_retriever",
    58: "great_pyrenees",
    59: "ibizan_hound",
    60: "dingo",
    61: "clumber",
    62: "cocker_spaniel",
    63: "entlebucher",
    64: "dhole",
    65: "english_setter",
    66: "doberman",
    67: "curly-coated_retriever",
    68: "collie",
    69: "english_foxhound",
    70: "borzoi",
    71: "cardigan",
    72: "brabancon_griffon",
    73: "boston_bull",
    74: "chesapeake_bay_retriever",
    75: "bouvier_des_flandres",
    76: "bull_mastiff",
    77: "chow",
    78: "cairn",
    79: "border_terrier",
    80: "bloodhound",
    81: "basset",
    82: "bluetick",
    83: "black-and-tan_coonhound",
    84: "blenheim_spaniel",
    85: "bedlington_terrier",
    86: "beagle",
    87: "basenji",
    88: "bernese_mountain_dog",
    89: "border_collie",
    90: "african_hunting_dog",
    91: "afghan_hound",
    92: "airedale"
}

# Route for home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('./static/images', imagefile.filename)
    imagefile.save(image_path)

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Adjust target_size to 224x224
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))  # Since target_size is fixed
    image = image / 255.0  # Normalize the image data

    # Make a prediction using your trained model
    yhat = model.predict(image)
    predicted_class = yhat.argmax(axis=-1)[0]
    confidence = np.max(yhat) * 100  # Confidence percentage

    # Get the predicted label from the dictionary
    predicted_genus = dogs_labels[predicted_class]

    # Generate the URL for the image
    image_url = url_for('static', filename='images/' + imagefile.filename)

    return render_template('index.html', prediction=predicted_genus, confidence=confidence, image_path=image_url)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=3000)
