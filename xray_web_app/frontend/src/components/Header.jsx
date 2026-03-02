/**
 * Header Component
 * Professional header with branding and navigation
 */

import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-container">
        {/* Logo and branding */}
        <div className="header-brand">
          <div className="logo">
            <div className="logo-icon">
              <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                {/* Stylized medical cross with radiograph elements */}
                <rect x="4" y="4" width="32" height="32" rx="8" fill="url(#logoGradient)" />
                <path
                  d="M20 10V30M10 20H30"
                  stroke="white"
                  strokeWidth="4"
                  strokeLinecap="round"
                />
                <circle cx="20" cy="20" r="6" stroke="white" strokeWidth="2" fill="none" />
                <defs>
                  <linearGradient id="logoGradient" x1="4" y1="4" x2="36" y2="36">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div className="logo-text">
              <h1 className="logo-title">X-Ray Analyzer</h1>
              <span className="logo-subtitle">AI-Powered Diagnostic Assistant</span>
            </div>
          </div>
        </div>

        {/* Status indicator */}
        <div className="header-status">
          <div className="status-indicator">
            <span className="status-dot"></span>
            <span className="status-text">System Ready</span>
          </div>
          <div className="header-info">
            <span className="info-badge">DenseNet-121</span>
            <span className="info-badge">18 Pathologies</span>
          </div>
        </div>
      </div>

      {/* Decorative gradient line */}
      <div className="header-gradient-line"></div>
    </header>
  );
}

export default Header;
