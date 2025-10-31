// API Base URL
const API_BASE = '/api';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    setupEventListeners();
    loadSubjects();
    loadEvents();
});

// Tab Management
function openTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');
    
    tabs.forEach(tab => tab.classList.remove('active'));
    buttons.forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    // Refresh data when opening certain tabs
    if (tabName === 'viewSubjects') loadSubjects();
    if (tabName === 'viewEvents') loadEvents();
}

// Setup Event Listeners
function setupEventListeners() {
    // Recognize Form
    document.getElementById('recognizeForm').addEventListener('submit', handleRecognize);
    document.getElementById('recognizeImage').addEventListener('change', function(e) {
        previewImage(e, 'recognizePreview');
    });
    
    // Threshold Slider
    document.getElementById('threshold').addEventListener('input', function(e) {
        document.getElementById('thresholdValue').textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    // Add Subject Form
    document.getElementById('addSubjectForm').addEventListener('submit', handleAddSubject);
    document.getElementById('subjectImage').addEventListener('change', function(e) {
        previewImage(e, 'subjectPreview');
    });
    
    // Search
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') searchSubjects();
    });
}

// Load Statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('totalSubjects').textContent = data.data.total_subjects || 0;
            document.getElementById('totalEvents').textContent = data.data.total_events || 0;
            document.getElementById('recentMatches').textContent = data.data.recent_matches || 0;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Preview Image
function previewImage(event, previewId) {
    const file = event.target.files[0];
    const preview = document.getElementById(previewId);
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
}

// Show Loader
function showLoader() {
    document.getElementById('loader').style.display = 'flex';
}

// Hide Loader
function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

// Handle Recognition
async function handleRecognize(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const resultsDiv = document.getElementById('recognizeResults');
    
    showLoader();
    resultsDiv.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/recognize`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayRecognitionResults(data.data);
            loadStats(); // Refresh stats
        } else {
            resultsDiv.innerHTML = `<div class="alert alert-error">Error: ${data.error}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-error">Error: ${error.message}</div>`;
    } finally {
        hideLoader();
    }
}

// Display Recognition Results
function displayRecognitionResults(data) {
    const resultsDiv = document.getElementById('recognizeResults');
    
    if (data.faces_detected === 0) {
        resultsDiv.innerHTML = `
            <div class="result-card warning">
                <h3>⚠️ No Faces Detected</h3>
                <p>No faces were found in the uploaded image.</p>
            </div>
        `;
        return;
    }
    
    let html = `<h3>Detected ${data.faces_detected} Face(s)</h3>`;
    
    data.matches.forEach((match, index) => {
        if (match.status === 'MATCH') {
            const m = match.best_match;
            html += `
                <div class="match-result">
                    <div class="match-header">
                        <h4>Face ${match.face_id}</h4>
                        <span class="match-status match">✓ MATCH FOUND</span>
                    </div>
                    <div class="match-details">
                        <div class="detail-item">
                            <div class="detail-label">Subject ID</div>
                            <div class="detail-value">${m.subject_id}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Name</div>
                            <div class="detail-value">${m.name}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Crime</div>
                            <div class="detail-value">${m.crime}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Similarity Score</div>
                            <div class="detail-value">${(m.similarity_score * 100).toFixed(2)}%</div>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${m.similarity_score * 100}%"></div>
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Confidence Level</div>
                            <div class="detail-value">
                                <span class="confidence-badge confidence-${m.confidence_level.toLowerCase()}">
                                    ${m.confidence_level}
                                </span>
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Detection Confidence</div>
                            <div class="detail-value">${(match.detection_confidence * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                    ${renderTopMatches(match.top_matches)}
                </div>
            `;
        } else {
            html += `
                <div class="match-result">
                    <div class="match-header">
                        <h4>Face ${match.face_id}</h4>
                        <span class="match-status no-match">✗ NO MATCH</span>
                    </div>
                    <p>${match.message}</p>
                    <p><small>Detection Confidence: ${(match.detection_confidence * 100).toFixed(2)}%</small></p>
                </div>
            `;
        }
    });
    
    resultsDiv.innerHTML = html;
}

// Render Top Matches
function renderTopMatches(matches) {
    if (!matches || matches.length === 0) return '';
    
    let html = `
        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #e9ecef;">
            <h5>Top Alternative Matches:</h5>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-top: 10px;">
    `;
    
    matches.slice(0, 3).forEach(match => {
        html += `
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-weight: bold; font-size: 0.9em;">${match.name}</div>
                <div style="font-size: 0.8em; color: #6c757d; margin: 5px 0;">${match.subject_id}</div>
                <div style="font-size: 0.85em; color: #667eea; font-weight: bold;">
                    ${(match.similarity_score * 100).toFixed(1)}%
                </div>
            </div>
        `;
    });
    
    html += `</div></div>`;
    return html;
}

// Handle Add Subject
async function handleAddSubject(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const resultsDiv = document.getElementById('addSubjectResults');
    
    showLoader();
    resultsDiv.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/add_subject`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            resultsDiv.innerHTML = `
                <div class="alert alert-success">
                    <h3>✓ Subject Added Successfully</h3>
                    <p><strong>ID:</strong> ${data.data.subject_id}</p>
                    <p><strong>Name:</strong> ${data.data.name}</p>
                    <p><strong>Detection Confidence:</strong> ${(data.data.confidence * 100).toFixed(2)}%</p>
                </div>
            `;
            event.target.reset();
            document.getElementById('subjectPreview').innerHTML = '';
            loadStats(); // Refresh stats
        } else {
            resultsDiv.innerHTML = `<div class="alert alert-error">Error: ${data.error}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="alert alert-error">Error: ${error.message}</div>`;
    } finally {
        hideLoader();
    }
}

// Load Subjects
async function loadSubjects() {
    const listDiv = document.getElementById('subjectsList');
    listDiv.innerHTML = '<p>Loading subjects...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/subjects`);
        const data = await response.json();
        
        if (data.success && data.data.length > 0) {
            displaySubjects(data.data);
        } else {
            listDiv.innerHTML = `
                <div class="empty-state">
                    <p>No subjects in database</p>
                </div>
            `;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="alert alert-error">Error loading subjects: ${error.message}</div>`;
    }
}

// Search Subjects
async function searchSubjects() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        loadSubjects();
        return;
    }
    
    const listDiv = document.getElementById('subjectsList');
    listDiv.innerHTML = '<p>Searching...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/subjects?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.success && data.data.length > 0) {
            displaySubjects(data.data);
        } else {
            listDiv.innerHTML = `
                <div class="empty-state">
                    <p>No subjects found matching "${query}"</p>
                </div>
            `;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="alert alert-error">Error: ${error.message}</div>`;
    }
}

// Display Subjects
function displaySubjects(subjects) {
    const listDiv = document.getElementById('subjectsList');
    
    let html = `<p style="margin-bottom: 15px; color: #6c757d;">Found ${subjects.length} subject(s)</p>`;
    
    subjects.forEach(subject => {
        html += `
            <div class="subject-card">
                <div class="subject-info">
                    <h3>${subject.name} <span class="subject-id">${subject.subject_id}</span></h3>
                    <p><strong>Crime:</strong> ${subject.crime}</p>
                    <p><small>Added: ${new Date(subject.added_on).toLocaleString()}</small></p>
                </div>
                <button class="btn btn-danger" onclick="deleteSubject('${subject.subject_id}')">Delete</button>
            </div>
        `;
    });
    
    listDiv.innerHTML = html;
}

// Delete Subject
async function deleteSubject(subjectId) {
    if (!confirm(`Are you sure you want to delete subject ${subjectId}?`)) {
        return;
    }
    
    showLoader();
    
    try {
        const response = await fetch(`${API_BASE}/subjects/${subjectId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadSubjects();
            loadStats();
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoader();
    }
}

// Load Events
async function loadEvents() {
    const listDiv = document.getElementById('eventsList');
    const eventType = document.getElementById('eventTypeFilter').value;
    
    listDiv.innerHTML = '<p>Loading events...</p>';
    
    try {
        let url = `${API_BASE}/events?limit=50`;
        if (eventType) {
            url += `&type=${eventType}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.data.length > 0) {
            displayEvents(data.data);
        } else {
            listDiv.innerHTML = `
                <div class="empty-state">
                    <p>No events found</p>
                </div>
            `;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="alert alert-error">Error loading events: ${error.message}</div>`;
    }
}

// Display Events
function displayEvents(events) {
    const listDiv = document.getElementById('eventsList');
    
    let html = `<p style="margin-bottom: 15px; color: #6c757d;">Showing ${events.length} event(s)</p>`;
    
    events.forEach(event => {
        const typeClass = event.event_type.toLowerCase().replace('_', '-');
        const eventTypeDisplay = event.event_type.replace('_', ' ');
        
        html += `
            <div class="event-card">
                <div>
                    <span class="event-type ${typeClass}">${eventTypeDisplay}</span>
                    ${event.subject_id ? `<strong>${event.subject_id}</strong>` : ''}
                    ${event.score ? ` - Score: ${(event.score * 100).toFixed(2)}%` : ''}
                </div>
                <div style="text-align: right; color: #6c757d; font-size: 0.9em;">
                    ${new Date(event.timestamp).toLocaleString()}
                </div>
            </div>
        `;
    });
    
    listDiv.innerHTML = html;
}

// Utility: Format Date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}