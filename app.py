from flask import Flask, request, render_template, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('saved_model/weights_best_vanilla.keras')

# Ensure the uploads directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Mapping of class indices to human-readable labels
class_labels = {
    0: "Safe Driving",
    1: "Texting - Right",
    2: "Talking on the Phone - Right",
    3: "Texting - Left",
    4: "Talking on the Phone - Left",
    5: "Operating the Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Distracted",
    9: "Talking to Passenger"
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            img = image.load_img(file_path, target_size=(64, 64), color_mode='grayscale')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            prediction_text = class_labels[predicted_class]
            return render_template('index.html', filename=file.filename, predicted_class=predicted_class, prediction_text=prediction_text)
    return render_template('upload.html')
'''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
