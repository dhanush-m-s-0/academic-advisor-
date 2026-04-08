/* traffic_script.js — Traffic Analyzer frontend */

// ── State ─────────────────────────────────────────────────────
let currentUser    = { id: 'guest', name: 'Guest' };
let queryCount     = 0;
let isWaiting      = false;

const ROLE_META = {
    traffic_flow    : { emoji: '🚦', name: 'Traffic Flow Analysis',   color: '#e74c3c' },
    route_planning  : { emoji: '🛣️', name: 'Route Planning',          color: '#2ecc71' },
    incident_safety : { emoji: '🚧', name: 'Incident & Safety',       color: '#f39c12' },
    traffic_data    : { emoji: '📊', name: 'Data & Statistics',       color: '#3498db' },
    infrastructure  : { emoji: '🏗️', name: 'Infrastructure',         color: '#9b59b6' },
    environmental   : { emoji: '🌿', name: 'Environmental Impact',    color: '#27ae60' },
};

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadUserInfo();
    setupEventListeners();
});

function setupEventListeners() {
    // Logout
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) logoutBtn.addEventListener('click', handleLogout);

    // Send button
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    // Textarea — Enter to send, Shift+Enter for newline
    const input = document.getElementById('messageInput');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        // Auto-resize
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        });
    }

    // Suggestion chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const msg = chip.dataset.msg;
            if (msg) {
                const input = document.getElementById('messageInput');
                if (input) {
                    input.value = msg;
                    sendMessage();
                }
            }
        });
    });

    // Role cards — clicking highlights them (role still auto-detected per query)
    document.querySelectorAll('.role-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.role-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        });
    });

    // Report button
    const reportBtn = document.getElementById('reportBtn');
    if (reportBtn) reportBtn.addEventListener('click', fetchReport);

    // Close report modal
    const closeReport = document.getElementById('closeReport');
    if (closeReport) closeReport.addEventListener('click', closeReportModal);

    // Close modal on overlay click
    const modal = document.getElementById('reportModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeReportModal();
        });
    }
}

// ── User info ─────────────────────────────────────────────────
function loadUserInfo() {
    fetch('/api/user-info')
        .then(res => res.json())
        .then(data => {
            currentUser.id   = data.id;
            currentUser.name = data.name;
            const el = document.getElementById('userInfo');
            if (el) el.textContent = data.name;
        })
        .catch(err => console.error('Error loading user info:', err));
}

// ── Logout ────────────────────────────────────────────────────
function handleLogout() {
    fetch('/api/logout', { method: 'POST' })
        .then(() => { window.location.href = '/login'; })
        .catch(err => console.error('Logout error:', err));
}

// ── Send message ──────────────────────────────────────────────
function sendMessage() {
    if (isWaiting) return;

    const input   = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    // Clear welcome message on first real question
    const welcome = document.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    // Hide suggestion chips after first send
    const chips = document.getElementById('suggestionChips');
    if (chips) chips.style.display = 'none';

    appendUserMessage(message);
    input.value = '';
    input.style.height = 'auto';

    const typingId = appendTypingIndicator();
    setWaiting(true);

    fetch('/api/chat', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ message }),
    })
    .then(res => res.json())
    .then(data => {
        removeTypingIndicator(typingId);
        setWaiting(false);

        if (data.error) {
            appendBotMessage(`⚠️ Error: ${data.error}`, null, null);
            return;
        }

        appendBotMessage(data.answer, data.role_id || 'traffic_flow', data.role, data.role_color);
        updateRoleUI(data.role_id || 'traffic_flow', data.role, data.role_emoji, data.role_color);
        queryCount++;
        const qc = document.getElementById('queryCount');
        if (qc) qc.textContent = queryCount;

        // Refresh profile display
        fetchProfile();
    })
    .catch(err => {
        removeTypingIndicator(typingId);
        setWaiting(false);
        appendBotMessage(`⚠️ Connection error: ${err.message}`, null, null);
    });
}

// ── DOM helpers ───────────────────────────────────────────────
function appendUserMessage(text) {
    const chatArea = document.getElementById('chatArea');
    const div = document.createElement('div');
    div.className = 'msg-user';
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatArea.appendChild(div);
    scrollToBottom();
}

function appendBotMessage(markdown, roleId, roleName, roleColor) {
    const chatArea = document.getElementById('chatArea');
    const group    = document.createElement('div');
    group.className = 'msg-group';

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let roleTagHtml = '';
    if (roleId && ROLE_META[roleId]) {
        const rm = ROLE_META[roleId];
        const color = roleColor || rm.color;
        roleTagHtml = `<span class="role-tag" style="background:${hexToRgba(color,0.15)};color:${color};border:1px solid ${hexToRgba(color,0.4)}">${rm.emoji} ${roleName || rm.name}</span>`;
    }

    group.innerHTML = `
        ${roleTagHtml}
        <div class="msg-bot">
            <div class="bubble">${renderMarkdown(markdown)}</div>
            <span class="msg-timestamp">${now}</span>
        </div>`;

    chatArea.appendChild(group);
    scrollToBottom();
}

function appendTypingIndicator() {
    const chatArea = document.getElementById('chatArea');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'msg-bot';
    div.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    chatArea.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const chatArea = document.getElementById('chatArea');
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Role UI update ─────────────────────────────────────────────
function updateRoleUI(roleId, roleName, roleEmoji, roleColor) {
    // Update header badge
    const badge = document.getElementById('currentRoleBadge');
    if (badge) {
        const color = roleColor || '#2ecc71';
        badge.style.background = hexToRgba(color, 0.15);
        badge.style.color      = color;
        badge.style.borderColor = hexToRgba(color, 0.3);
        badge.textContent       = `${roleEmoji || ''} ${roleName || roleId}`;
    }

    // Highlight matching role card in sidebar
    document.querySelectorAll('.role-card').forEach(card => {
        card.classList.toggle('active', card.dataset.role === roleId);
    });
}

// ── Profile ───────────────────────────────────────────────────
function fetchProfile() {
    fetch('/api/profile')
        .then(res => res.json())
        .then(data => {
            if (data.context) {
                const el = document.getElementById('profileInfo');
                if (el) {
                    el.innerHTML = data.context
                        .split('\n')
                        .filter(l => l.trim())
                        .map(l => `<div class="profile-line">${escapeHtml(l)}</div>`)
                        .join('');
                }
            }
        })
        .catch(() => {});
}

// ── Report ────────────────────────────────────────────────────
function fetchReport() {
    const modal   = document.getElementById('reportModal');
    const content = document.getElementById('reportContent');
    if (!modal || !content) return;

    content.innerHTML = '<div class="loading-spinner">⏳ Generating session report…</div>';
    modal.style.display = 'flex';

    fetch('/api/report')
        .then(res => res.json())
        .then(data => {
            content.innerHTML = data.report
                ? renderMarkdown(data.report)
                : '<p>No report data available.</p>';
        })
        .catch(err => {
            content.innerHTML = `<p style="color:#e74c3c;">Error: ${err.message}</p>`;
        });
}

function closeReportModal() {
    const modal = document.getElementById('reportModal');
    if (modal) modal.style.display = 'none';
}

// ── Waiting state ─────────────────────────────────────────────
function setWaiting(state) {
    isWaiting = state;
    const sendBtn = document.getElementById('sendBtn');
    const input   = document.getElementById('messageInput');
    if (sendBtn) sendBtn.disabled = state;
    if (input)   input.disabled   = state;
}

// ── Markdown renderer (minimal) ───────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';
    return text
        // Escape HTML first to prevent XSS
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/__(.+?)__/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/_(.+?)_/g, '<em>$1</em>')
        // Inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
        .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
        // Unordered list items
        .replace(/^[\-\*\•] (.+)$/gm, '<li>$1</li>')
        // Ordered list items
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Wrap consecutive <li> in <ul>
        .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        // Line breaks → paragraphs
        .replace(/\n{2,}/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

// ── Utilities ─────────────────────────────────────────────────
function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function hexToRgba(hex, alpha) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(46,204,113,${alpha})`;
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r},${g},${b},${alpha})`;
}
