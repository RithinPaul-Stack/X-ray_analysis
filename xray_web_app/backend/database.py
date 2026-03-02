"""
SQLite Database Handler
Manages local storage of X-ray analysis results
"""

import sqlite3
import os
import json
from datetime import datetime

# Database file path (in the same directory as this script)
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xray_analysis.db')


def get_connection():
    """Get database connection with row factory for dict-like access"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initialize the database with required tables"""
    conn = get_connection()
    cursor = conn.cursor()

    # Create analyses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            status TEXT NOT NULL,
            status_message TEXT,
            threshold REAL DEFAULT 0.5,
            abnormal_count INTEGER DEFAULT 0,
            total_pathologies INTEGER DEFAULT 0,
            recommendations TEXT,
            disclaimer TEXT
        )
    ''')

    # Create pathology_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pathology_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            pathology_name TEXT NOT NULL,
            score REAL NOT NULL,
            percentage REAL NOT NULL,
            is_abnormal INTEGER NOT NULL DEFAULT 0,
            severity TEXT,
            description TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        )
    ''')

    # Create indexes for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_analyses_created_at
        ON analyses(created_at DESC)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_pathology_analysis_id
        ON pathology_results(analysis_id)
    ''')

    conn.commit()
    conn.close()
    print(f"  Database initialized: {DATABASE_PATH}")


def save_analysis(image_path, image_filename, results):
    """
    Save analysis results to the database

    Args:
        image_path: Full path to the saved image
        image_filename: Filename of the image
        results: Dictionary containing analysis results

    Returns:
        The ID of the newly created analysis record
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Insert main analysis record
        cursor.execute('''
            INSERT INTO analyses (
                image_path, image_filename, status, status_message,
                threshold, abnormal_count, total_pathologies,
                recommendations, disclaimer
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_path,
            image_filename,
            results['status'],
            results['status_message'],
            results['threshold'],
            results['abnormal_count'],
            results['total_pathologies'],
            json.dumps(results['recommendations']),
            results['disclaimer']
        ))

        analysis_id = cursor.lastrowid

        # Insert pathology results
        for pathology in results['pathologies']:
            cursor.execute('''
                INSERT INTO pathology_results (
                    analysis_id, pathology_name, score, percentage,
                    is_abnormal, severity, description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                pathology['name'],
                pathology['score'],
                pathology['percentage'],
                1 if pathology['is_abnormal'] else 0,
                pathology['severity'],
                pathology['description']
            ))

        conn.commit()
        return analysis_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_analysis(analysis_id):
    """
    Retrieve a specific analysis by ID

    Args:
        analysis_id: The ID of the analysis to retrieve

    Returns:
        Dictionary containing analysis data or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Get main analysis record
        cursor.execute('''
            SELECT * FROM analyses WHERE id = ?
        ''', (analysis_id,))
        analysis_row = cursor.fetchone()

        if analysis_row is None:
            return None

        # Get pathology results
        cursor.execute('''
            SELECT * FROM pathology_results
            WHERE analysis_id = ?
            ORDER BY score DESC
        ''', (analysis_id,))
        pathology_rows = cursor.fetchall()

        # Build response dictionary
        analysis = dict(analysis_row)
        analysis['recommendations'] = json.loads(analysis['recommendations']) if analysis['recommendations'] else []
        analysis['pathologies'] = [
            {
                'name': row['pathology_name'],
                'score': row['score'],
                'percentage': row['percentage'],
                'is_abnormal': bool(row['is_abnormal']),
                'severity': row['severity'],
                'description': row['description']
            }
            for row in pathology_rows
        ]

        # Get top findings (abnormal only)
        analysis['top_findings'] = [p for p in analysis['pathologies'] if p['is_abnormal']][:5]

        return analysis

    finally:
        conn.close()


def get_all_analyses():
    """
    Get all analyses ordered by date (newest first)

    Returns:
        List of analysis summaries (without full pathology details)
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT id, created_at, image_filename, status,
                   abnormal_count, total_pathologies
            FROM analyses
            ORDER BY created_at DESC
            LIMIT 100
        ''')
        rows = cursor.fetchall()

        return [
            {
                'id': row['id'],
                'created_at': row['created_at'],
                'image_filename': row['image_filename'],
                'status': row['status'],
                'abnormal_count': row['abnormal_count'],
                'total_pathologies': row['total_pathologies']
            }
            for row in rows
        ]

    finally:
        conn.close()


def delete_analysis(analysis_id):
    """
    Delete an analysis and its associated image file

    Args:
        analysis_id: The ID of the analysis to delete
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Get image path before deleting
        cursor.execute('''
            SELECT image_path FROM analyses WHERE id = ?
        ''', (analysis_id,))
        row = cursor.fetchone()

        if row and row['image_path']:
            # Delete image file if it exists
            if os.path.exists(row['image_path']):
                os.remove(row['image_path'])

        # Delete analysis (CASCADE will delete pathology_results)
        cursor.execute('''
            DELETE FROM analyses WHERE id = ?
        ''', (analysis_id,))

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def clear_all_analyses():
    """
    Delete all analyses and their images (for cleanup/reset)
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Get all image paths
        cursor.execute('SELECT image_path FROM analyses')
        rows = cursor.fetchall()

        # Delete image files
        for row in rows:
            if row['image_path'] and os.path.exists(row['image_path']):
                os.remove(row['image_path'])

        # Delete all records
        cursor.execute('DELETE FROM pathology_results')
        cursor.execute('DELETE FROM analyses')

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
