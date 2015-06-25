import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from code.mvpNeuralNet import ArtistClassifier, PortraitClassifier
from code.uploadImagePipeline import preProcessUploadedImage

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.artistNN = ArtistClassifier()
app.portraitNN = PortraitClassifier()

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def classifyPainting(classifier, f_name):
    if classifier == 'Genre':
        imagePipeline = preProcessUploadedImage(f_name, app.portraitNN, isArtist=False)
    else:
        imagePipeline = preProcessUploadedImage(f_name, app.artistNN, isArtist=True)
    imagePipeline.preprocessing()
    imagePipeline.vectorize()
    return imagePipeline.predict()

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('videotron.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Route that will process the file upload
@app.route('/predict', methods=['POST'])
def viewImage():
    f_name = request.form['image_path']
    classifier = request.form['classifier']
    prediction = classifyPainting(classifier, f_name)
    return render_template('upload.html', prediction=prediction, img_path=f_name)


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    classifier = request.form['classifier']
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        f_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(f_name))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
    view_predict = request.form['vp']
    if view_predict == 'View Image':
        return render_template('view.html', img_path=f_name)
    else:
        prediction = classifyPainting(classifier, f_name)

        return render_template('upload.html', prediction=prediction, img_path=f_name)

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6969, debug=True)
