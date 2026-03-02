/**
 * AnalysisResults Component
 * Displays comprehensive X-ray analysis results with visualizations
 */

import React, { useState } from 'react';
import './AnalysisResults.css';

function AnalysisResults({ results }) {
  const [activeTab, setActiveTab] = useState('overview');

  if (!results) return null;

  const {
    status,
    status_message,
    pathologies,
    top_findings,
    recommendations,
    abnormal_count,
    total_pathologies,
    threshold,
    disclaimer
  } = results;

  const isAbnormal = status === 'ABNORMAL';

  /**
   * Get severity color class
   */
  const getSeverityClass = (severity) => {
    switch (severity) {
      case 'high': return 'severity-high';
      case 'moderate': return 'severity-moderate';
      case 'low': return 'severity-low';
      default: return 'severity-normal';
    }
  };

  /**
   * Get progress bar width
   */
  const getProgressWidth = (percentage) => {
    return Math.min(percentage, 100);
  };

  /**
   * Render overview tab
   */
  const renderOverview = () => (
    <div className="tab-content overview-content">
      {/* Status banner */}
      <div className={`status-banner ${isAbnormal ? 'status-abnormal' : 'status-normal'}`}>
        <div className="status-icon">
          {isAbnormal ? (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>
        <div className="status-content">
          <h3 className="status-title">
            {isAbnormal ? 'Abnormal Findings Detected' : 'Normal Study'}
          </h3>
          <p className="status-message">{status_message}</p>
        </div>
        <div className="status-stats">
          <div className="stat">
            <span className="stat-value">{abnormal_count}</span>
            <span className="stat-label">Findings</span>
          </div>
          <div className="stat">
            <span className="stat-value">{total_pathologies}</span>
            <span className="stat-label">Screened</span>
          </div>
        </div>
      </div>

      {/* Top findings */}
      {top_findings && top_findings.length > 0 && (
        <div className="findings-section">
          <h4 className="section-subtitle">
            <span className="subtitle-icon">&#9888;</span>
            Key Findings Requiring Attention
          </h4>
          <div className="findings-list">
            {top_findings.map((finding, index) => (
              <div key={index} className="finding-card">
                <div className="finding-header">
                  <span className={`severity-badge ${getSeverityClass(finding.severity)}`}>
                    {finding.severity.toUpperCase()}
                  </span>
                  <span className="finding-name">{finding.name}</span>
                  <span className="finding-score">{finding.percentage}%</span>
                </div>
                <div className="finding-progress">
                  <div
                    className={`progress-bar ${getSeverityClass(finding.severity)}`}
                    style={{ width: `${getProgressWidth(finding.percentage)}%` }}
                  />
                </div>
                <p className="finding-description">{finding.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="recommendations-section">
          <h4 className="section-subtitle">
            <span className="subtitle-icon">&#128203;</span>
            Clinical Recommendations
          </h4>
          <ul className="recommendations-list">
            {recommendations.map((rec, index) => (
              <li key={index} className="recommendation-item">
                <span className="rec-bullet">&#8227;</span>
                <span className="rec-text">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  /**
   * Render all pathologies tab
   */
  const renderPathologies = () => (
    <div className="tab-content pathologies-content">
      <div className="pathologies-header">
        <span className="pathologies-count">
          {total_pathologies} pathologies screened | Threshold: {(threshold * 100).toFixed(0)}%
        </span>
      </div>
      <div className="pathologies-grid">
        {pathologies.map((path, index) => (
          <div
            key={index}
            className={`pathology-item ${path.is_abnormal ? 'pathology-abnormal' : ''}`}
          >
            <div className="pathology-header">
              <span className="pathology-name">{path.name}</span>
              <span className={`pathology-score ${path.is_abnormal ? 'score-abnormal' : ''}`}>
                {path.percentage}%
              </span>
            </div>
            <div className="pathology-bar-container">
              <div
                className={`pathology-bar ${path.is_abnormal ? getSeverityClass(path.severity) : ''}`}
                style={{ width: `${getProgressWidth(path.percentage)}%` }}
              />
              <div
                className="threshold-marker"
                style={{ left: `${threshold * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  /**
   * Render disclaimer tab
   */
  const renderDisclaimer = () => (
    <div className="tab-content disclaimer-content">
      <div className="disclaimer-card">
        <div className="disclaimer-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h4 className="disclaimer-title">Medical Disclaimer</h4>
        <p className="disclaimer-text">{disclaimer}</p>
        <div className="disclaimer-points">
          <div className="disclaimer-point">
            <span className="point-icon">&#10003;</span>
            <span>This tool is designed to assist, not replace, clinical judgment</span>
          </div>
          <div className="disclaimer-point">
            <span className="point-icon">&#10003;</span>
            <span>All findings should be verified by a qualified radiologist</span>
          </div>
          <div className="disclaimer-point">
            <span className="point-icon">&#10003;</span>
            <span>Clinical correlation with patient history is essential</span>
          </div>
          <div className="disclaimer-point">
            <span className="point-icon">&#10003;</span>
            <span>AI predictions may have false positives and false negatives</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="analysis-results">
      {/* Tab navigation */}
      <div className="results-tabs">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <span className="tab-icon">&#128200;</span>
          Overview
        </button>
        <button
          className={`tab-btn ${activeTab === 'pathologies' ? 'active' : ''}`}
          onClick={() => setActiveTab('pathologies')}
        >
          <span className="tab-icon">&#128202;</span>
          All Pathologies
        </button>
        <button
          className={`tab-btn ${activeTab === 'disclaimer' ? 'active' : ''}`}
          onClick={() => setActiveTab('disclaimer')}
        >
          <span className="tab-icon">&#9432;</span>
          Disclaimer
        </button>
      </div>

      {/* Tab content */}
      <div className="results-content">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'pathologies' && renderPathologies()}
        {activeTab === 'disclaimer' && renderDisclaimer()}
      </div>
    </div>
  );
}

export default AnalysisResults;
