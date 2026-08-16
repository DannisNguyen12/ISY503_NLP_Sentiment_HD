
"""Thin wrapper for the inference-only Flask app.

Vercel can import this file and find the Flask application object without
pulling in any training code.
"""

from inference.inference import app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
