/**
 * X-Ray Analyzer - Main Application Component
 * Professional medical UI for chest X-ray analysis
 */

import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import AnalysisResults from './components/AnalysisResults';
import LoadingOverlay from './components/LoadingOverlay';
import Footer from './components/Footer';
import { uploadAndAnalyze, getImageUrl } from './services/api';
import './App.css';

function App() {
  // Application state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);
  const [imageFilename, setImageFilename] = useState(null);

  /**
   * Handle file selection
   */
  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    // Clear previous results when new file is selected
    setAnalysisResults(null);
    setAnalysisId(null);
    setImageFilename(null);
  }, []);

  /**
   * Handle analysis submission
   */
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select an X-ray image first.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setUploadProgress(0);

    try {
      const response = await uploadAndAnalyze(selectedFile, (progress) => {
        setUploadProgress(progress);
      });

      if (response.success) {
        setAnalysisResults(response.results);
        setAnalysisId(response.analysis_id);
        setImageFilename(response.filename);
      } else {
        setError(response.error || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  }, [selectedFile]);

  /**
   * Clear all state and reset for new analysis
   */
  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResults(null);
    setIsAnalyzing(false);
    setUploadProgress(0);
    setError(null);
    setAnalysisId(null);
    setImageFilename(null);
  }, []);

  /**
   * Get the display image URL
   */
  const getDisplayImageUrl = () => {
    if (imageFilename) {
      return getImageUrl(imageFilename);
    }
    return previewUrl;
  };

  return (
    <div className="app">
      {/* Loading overlay during analysis */}
      {isAnalyzing && (
        <LoadingOverlay progress={uploadProgress} />
      )}

      {/* Header */}
      <Header />

      {/* Main content */}
      <main className="main-content">
        <div className="container">
          {/* Error display */}
          {error && (
            <div className="error-banner">
              <span className="error-icon">!</span>
              <span className="error-message">{error}</span>
              <button className="error-dismiss" onClick={() => setError(null)}>
                &times;
              </button>
            </div>
          )}

          {/* Two-column layout */}
          <div className="content-grid">
            {/* Left column - Upload section */}
            <section className="upload-section">
              <div className="section-card">
                <div className="section-header">
                  <h2 className="section-title">
                    <span className="section-icon">1</span>
                    Upload X-Ray Image
                  </h2>
                </div>

                <ImageUpload
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                  previewUrl={previewUrl}
                  isDisabled={isAnalyzing}
                />

                {/* Action buttons */}
                <div className="action-buttons">
                  <button
                    className="btn btn-primary"
                    onClick={handleAnalyze}
                    disabled={!selectedFile || isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <span className="btn-icon">&#9881;</span>
                        Analyze X-Ray
                      </>
                    )}
                  </button>

                  <button
                    className="btn btn-secondary"
                    onClick={handleClear}
                    disabled={isAnalyzing || (!selectedFile && !analysisResults)}
                  >
                    <span className="btn-icon">&#8634;</span>
                    Clear
                  </button>
                </div>
              </div>

              {/* Image preview panel */}
              {(previewUrl || imageFilename) && (
                <div className="section-card image-preview-card">
                  <div className="section-header">
                    <h2 className="section-title">
                      <span className="section-icon">&#128444;</span>
                      X-Ray Image
                    </h2>
                  </div>
                  <div className="image-preview-container">
                    <img
                      src={getDisplayImageUrl()}
                      alt="X-Ray Preview"
                      className="xray-preview-image"
                    />
                  </div>
                </div>
              )}
            </section>

            {/* Right column - Results section */}
            <section className="results-section">
              <div className="section-card results-card">
                <div className="section-header">
                  <h2 className="section-title">
                    <span className="section-icon">2</span>
                    Analysis Results
                  </h2>
                  {analysisId && (
                    <span className="analysis-id">ID: {analysisId}</span>
                  )}
                </div>

                {analysisResults ? (
                  <AnalysisResults results={analysisResults} />
                ) : (
                  <div className="results-placeholder">
                    <div className="placeholder-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M9 17H5a2 2 0 01-2-2V5a2 2 0 012-2h14a2 2 0 012 2v10a2 2 0 01-2 2h-4" />
                        <path d="M12 15l-3 3m0 0l3 3m-3-3h12" />
                      </svg>
                    </div>
                    <h3 className="placeholder-title">No Analysis Yet</h3>
                    <p className="placeholder-text">
                      Upload a chest X-ray image and click "Analyze" to receive
                      AI-powered diagnostic insights.
                    </p>
                    <div className="placeholder-steps">
                      <div className="step">
                        <span className="step-num">1</span>
                        <span>Upload JPEG/PNG image</span>
                      </div>
                      <div className="step">
                        <span className="step-num">2</span>
                        <span>Click Analyze button</span>
                      </div>
                      <div className="step">
                        <span className="step-num">3</span>
                        <span>Review AI findings</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        </div>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default App;
