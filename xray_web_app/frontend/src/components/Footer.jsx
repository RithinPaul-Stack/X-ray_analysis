/**
 * Footer Component
 * Application footer with disclaimer and version info
 */

import React from 'react';
import './Footer.css';

function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          {/* Disclaimer */}
          <div className="footer-disclaimer">
            <span className="disclaimer-icon">&#9432;</span>
            <span className="disclaimer-text">
              For research and educational purposes only. Not intended for clinical diagnosis.
            </span>
          </div>

          {/* Version and credits */}
          <div className="footer-info">
            <span className="footer-version">v1.0.0</span>
            <span className="footer-separator">|</span>
            <span className="footer-credits">
              Powered by torchxrayvision DenseNet-121
            </span>
            <span className="footer-separator">|</span>
            <span className="footer-copyright">
              &copy; {currentYear} X-Ray Analyzer
            </span>
          </div>
        </div>
      </div>

      {/* Decorative gradient line */}
      <div className="footer-gradient-line"></div>
    </footer>
  );
}

export default Footer;
