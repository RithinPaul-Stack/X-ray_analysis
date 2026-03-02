/**
 * LoadingOverlay Component
 * Full-screen loading indicator during analysis
 */

import React from 'react';
import './LoadingOverlay.css';

function LoadingOverlay({ progress }) {
  return (
    <div className="loading-overlay">
      <div className="loading-content">
        {/* Animated scanner effect */}
        <div className="scanner-container">
          <div className="scanner-xray">
            <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
              {/* Simplified chest outline */}
              <ellipse cx="50" cy="40" rx="35" ry="30" stroke="currentColor" strokeWidth="1.5" opacity="0.3" />
              <ellipse cx="35" cy="40" rx="12" ry="18" stroke="currentColor" strokeWidth="1" opacity="0.5" />
              <ellipse cx="65" cy="40" rx="12" ry="18" stroke="currentColor" strokeWidth="1" opacity="0.5" />
              <path d="M50 20 L50 70" stroke="currentColor" strokeWidth="1" opacity="0.3" />
              <ellipse cx="50" cy="75" rx="8" ry="5" stroke="currentColor" strokeWidth="1" opacity="0.4" />
            </svg>
            <div className="scan-line" />
          </div>
        </div>

        {/* Loading text */}
        <div className="loading-text">
          <h3 className="loading-title">Analyzing X-Ray</h3>
          <p className="loading-subtitle">
            AI model is processing the radiograph...
          </p>
        </div>

        {/* Progress indicator */}
        <div className="loading-progress">
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: progress > 0 ? `${progress}%` : '30%' }}
            />
          </div>
          <span className="progress-text">
            {progress > 0 ? `Uploading... ${progress}%` : 'Processing...'}
          </span>
        </div>

        {/* Analysis steps */}
        <div className="loading-steps">
          <div className={`step ${progress < 100 ? 'active' : 'done'}`}>
            <span className="step-icon">&#128196;</span>
            <span>Preprocessing image</span>
          </div>
          <div className={`step ${progress === 100 ? 'active' : ''}`}>
            <span className="step-icon">&#129504;</span>
            <span>Running DenseNet-121</span>
          </div>
          <div className="step">
            <span className="step-icon">&#128202;</span>
            <span>Generating report</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LoadingOverlay;
