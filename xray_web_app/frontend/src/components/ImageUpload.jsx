/**
 * ImageUpload Component
 * Drag-and-drop file upload with preview functionality
 */

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './ImageUpload.css';

function ImageUpload({ onFileSelect, selectedFile, previewUrl, isDisabled }) {
  /**
   * Handle file drop/selection
   */
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      const error = rejectedFiles[0].errors[0];
      if (error.code === 'file-too-large') {
        alert('File is too large. Maximum size is 16MB.');
      } else if (error.code === 'file-invalid-type') {
        alert('Invalid file type. Please upload DICOM, PNG, or JPEG images.');
      }
      return;
    }

    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'application/dicom': ['.dcm', '.dicom'],
    },
    maxSize: 16 * 1024 * 1024, // 16MB
    multiple: false,
    disabled: isDisabled,
  });

  /**
   * Format file size for display
   */
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  /**
   * Get dropzone class based on state
   */
  const getDropzoneClass = () => {
    let className = 'dropzone';
    if (isDragActive) className += ' dropzone-active';
    if (isDragAccept) className += ' dropzone-accept';
    if (isDragReject) className += ' dropzone-reject';
    if (isDisabled) className += ' dropzone-disabled';
    if (selectedFile) className += ' dropzone-has-file';
    return className;
  };

  return (
    <div className="image-upload">
      <div {...getRootProps({ className: getDropzoneClass() })}>
        <input {...getInputProps()} />

        {selectedFile ? (
          // File selected state
          <div className="dropzone-file-info">
            <div className="file-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div className="file-details">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
            <div className="file-status">
              <span className="status-check">&#10003;</span>
              Ready for analysis
            </div>
          </div>
        ) : (
          // Empty state
          <div className="dropzone-content">
            <div className="dropzone-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="dropzone-text">
              {isDragActive ? (
                isDragAccept ? (
                  <span className="text-accept">Drop the X-ray image here...</span>
                ) : (
                  <span className="text-reject">This file type is not supported</span>
                )
              ) : (
                <>
                  <span className="text-primary">Drag & drop your X-ray image here</span>
                  <span className="text-secondary">or click to browse files</span>
                </>
              )}
            </div>
            <div className="dropzone-formats">
              <span className="format-badge">DICOM</span>
              <span className="format-badge">JPEG</span>
              <span className="format-badge">PNG</span>
              <span className="format-info">Max 16MB</span>
            </div>
          </div>
        )}
      </div>

      {/* Upload tips */}
      <div className="upload-tips">
        <div className="tip">
          <span className="tip-icon">&#9432;</span>
          <span>For best results, use high-quality PA (posterior-anterior) chest radiographs</span>
        </div>
      </div>
    </div>
  );
}

export default ImageUpload;
