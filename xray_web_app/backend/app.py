"""
X-Ray Analysis Web Application - Flask Backend
Main application entry point with API endpoints
"""

from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

from xray_analyzer import XRayAnalyzer
from gradcam_analyzer import get_gradcam_analyzer, GRADCAM_AVAILABLE
from pdf_generator import get_pdf_generator
from database import init_db, save_analysis, get_analysis, get_all_analyses, delete_analysis

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
init_db()

# Initialize ML model (loaded once at startup for efficiency)
print("Loading X-Ray Analysis Model...")
analyzer = XRayAnalyzer()
print("Model loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'X-Ray Analysis API is running',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_and_analyze():
    """
    Handle X-ray image upload and perform analysis
    Returns analysis results with pathology predictions
    """
    # Check if file is present in request
    if 'xray_image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400

    file = request.files['xray_image']

    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Please upload DICOM, PNG, or JPEG images.'
        }), 400

    try:
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"xray_{timestamp}_{unique_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        file.save(filepath)

        # Run X-ray analysis
        results = analyzer.analyze(filepath)

        # Save to database
        analysis_id = save_analysis(
            image_path=filepath,
            image_filename=filename,
            results=results
        )

        # Return results
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'filename': filename,
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/api/analysis/<int:analysis_id>', methods=['GET'])
def get_analysis_by_id(analysis_id):
    """Retrieve a specific analysis by ID"""
    try:
        analysis = get_analysis(analysis_id)
        if analysis is None:
            return jsonify({
                'success': False,
                'error': 'Analysis not found'
            }), 404

        return jsonify({
            'success': True,
            'analysis': analysis
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all previous analyses"""
    try:
        analyses = get_all_analyses()
        return jsonify({
            'success': True,
            'analyses': analyses,
            'count': len(analyses)
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analysis/<int:analysis_id>', methods=['DELETE'])
def delete_analysis_by_id(analysis_id):
    """Delete a specific analysis"""
    try:
        delete_analysis(analysis_id)
        return jsonify({
            'success': True,
            'message': 'Analysis deleted successfully'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/images/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/annotated/<filename>', methods=['GET'])
def serve_annotated_image(filename):
    """Serve annotated images from GradCAM analysis"""
    annotated_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated')
    return send_from_directory(annotated_folder, filename)


@app.route('/api/deep-analysis', methods=['POST'])
def deep_analysis():
    """
    Perform GradCAM-based deep analysis on an uploaded image.
    Returns heatmap visualization showing regions that influenced predictions.
    """
    # Check if GradCAM is available
    if not GRADCAM_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'GradCAM analysis not available.'
        }), 503

    # Check if filename is provided
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({
            'success': False,
            'error': 'No filename provided'
        }), 400

    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if file exists
    if not os.path.exists(filepath):
        return jsonify({
            'success': False,
            'error': 'Image file not found'
        }), 404

    try:
        # Get GradCAM analyzer and run analysis
        # Share the model with the main analyzer for efficiency
        gradcam = get_gradcam_analyzer(model=analyzer.model, device=analyzer.device)
        results = gradcam.analyze(filepath)

        return jsonify({
            'success': True,
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Deep analysis failed: {str(e)}'
        }), 500


@app.route('/api/gradcam-status', methods=['GET'])
def gradcam_status():
    """Check if GradCAM analysis is available"""
    return jsonify({
        'available': GRADCAM_AVAILABLE,
        'message': 'GradCAM heatmap analysis is ready' if GRADCAM_AVAILABLE else 'GradCAM not available'
    })


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf_report():
    """
    Generate a PDF report for the X-ray analysis.
    Requires analysis results and optionally heatmap results.
    """
    data = request.get_json()

    if not data:
        return jsonify({
            'success': False,
            'error': 'No data provided'
        }), 400

    # Get required data
    filename = data.get('filename')
    analysis_results = data.get('analysis_results')
    heatmap_results = data.get('heatmap_results')
    heatmap_filename = data.get('heatmap_filename')

    if not filename or not analysis_results:
        return jsonify({
            'success': False,
            'error': 'Missing required data: filename and analysis_results'
        }), 400

    # Build file paths
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(original_image_path):
        return jsonify({
            'success': False,
            'error': 'Original image file not found'
        }), 404

    # Get heatmap image path if available
    heatmap_image_path = None
    if heatmap_filename:
        annotated_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotated')
        heatmap_image_path = os.path.join(annotated_folder, heatmap_filename)
        if not os.path.exists(heatmap_image_path):
            heatmap_image_path = None

    try:
        # Generate PDF
        pdf_generator = get_pdf_generator()
        pdf_path = pdf_generator.generate_report(
            analysis_results=analysis_results,
            original_image_path=original_image_path,
            heatmap_image_path=heatmap_image_path,
            heatmap_results=heatmap_results
        )

        # Get just the filename for the response
        pdf_filename = os.path.basename(pdf_path)

        return jsonify({
            'success': True,
            'pdf_filename': pdf_filename,
            'message': 'PDF report generated successfully'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'PDF generation failed: {str(e)}'
        }), 500


@app.route('/api/reports/<filename>', methods=['GET'])
def download_pdf_report(filename):
    """Download a generated PDF report"""
    reports_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    pdf_path = os.path.join(reports_folder, filename)

    if not os.path.exists(pdf_path):
        return jsonify({
            'success': False,
            'error': 'PDF report not found'
        }), 404

    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  X-Ray Analysis Web Application")
    print("  Server Starting...")
    print("="*50)
    print(f"\n  Open in browser: http://localhost:8080")
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print("\n  API Endpoints:")
    print("  - GET  / - Main application")
    print("  - POST /api/upload - Upload and analyze X-ray")
    print("  - POST /api/deep-analysis - GradCAM heatmap analysis")
    print("  - POST /api/generate-pdf - Generate PDF report")
    print("  - GET  /api/reports/<filename> - Download PDF report")
    print("  - GET  /api/analysis/<id> - Get analysis by ID")
    print("  - GET  /api/history - Get all analyses")
    print("  - DELETE /api/analysis/<id> - Delete analysis")
    print(f"\n  GradCAM Status: {'Available' if GRADCAM_AVAILABLE else 'Not available'}")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=8080)
