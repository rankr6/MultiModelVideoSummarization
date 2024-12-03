from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
from summarize import summarize_video  # Assuming this function returns both summary and recommendations
from werkzeug.utils import secure_filename

# Create Flask app with correct folder paths
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Summarize video and get recommendations (if applicable)
            summary = summarize_video(filepath)

            # Return the video filename along with the summary
            return jsonify({
                'summary': summary.get('summary'),
                'filename': unique_filename,
            })
        
        except Exception as e:
            # Log the error for debugging and return an error response
            app.logger.error(f"Error during video summarization: {e}")
            return jsonify({'error': str(e)}), 500

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
