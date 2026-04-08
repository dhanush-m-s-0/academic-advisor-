"""
Flask web server for Traffic Analyzer Agent
Mirrors app.py architecture; runs on port 5001
"""
from flask import Flask, render_template, request, jsonify, redirect, session
from flask_cors import CORS
from traffic_analyzer_agent import TrafficAnalyzerAgent
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = 'traffic-analyzer-secret-key'
CORS(app)

# Initialize the agent once at startup
agent = TrafficAnalyzerAgent()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect('/dashboard')
    return redirect('/login')


@app.route('/login')
def login():
    return render_template('traffic_login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('traffic_index.html')


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages from frontend"""
    try:
        data    = request.json
        user_id = session.get('user_id', 'guest')
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Empty message'}), 400

        response = agent.analyse(message, user_id=user_id)

        return jsonify({
            'success'    : True,
            'answer'     : response['answer'],
            'role'       : response['role'],
            'role_emoji' : response['role_emoji'],
            'role_color' : response['role_color'],
            'sources'    : response['sources'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile"""
    try:
        user_id = session.get('user_id', 'guest')
        if user_id in agent.profiles:
            profile = agent.profiles[user_id]
            return jsonify({
                'name'        : profile.name,
                'context'     : profile.context_string(),
                'interactions': len(profile.session_log),
            })
        return jsonify({'error': 'Profile not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report', methods=['GET'])
@login_required
def get_report():
    """Generate session report"""
    try:
        user_id = session.get('user_id', 'guest')
        report  = agent.generate_session_report(user_id)
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login from frontend"""
    try:
        data      = request.json
        user_id   = data.get('user_id', '').strip()
        user_name = data.get('user_name', '').strip()

        if not user_id:
            return jsonify({'error': 'User ID required'}), 400

        session['user_id']   = user_id
        session['user_name'] = user_name or f'User {user_id}'

        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle logout"""
    session.clear()
    return jsonify({'success': True, 'redirect': '/login'})


@app.route('/api/user-info', methods=['GET'])
@login_required
def get_user_info():
    """Get current user info"""
    return jsonify({
        'id'  : session.get('user_id'),
        'name': session.get('user_name', 'Guest'),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
