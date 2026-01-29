// API endpoint
const API_URL = 'https://plasmaltp-rag.onrender.com';

// DOM elements
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const responseSection = document.getElementById('responseSection');
const answerContent = document.getElementById('answerContent');
const evidenceSection = document.getElementById('evidenceSection');
const evidenceList = document.getElementById('evidenceList');
const errorSection = document.getElementById('errorSection');
const errorContent = document.getElementById('errorContent');

// Submit handler
askBtn.addEventListener('click', handleAsk);
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        handleAsk();
    }
});

async function handleAsk() {
    const question = questionInput.value.trim();
    
    if (!question) {
        showError('Please enter a question.');
        return;
    }

    // Hide previous results and errors
    hideAllSections();
    showLoading();

    try {
        const requestBody = {
            question: question,
            top_k: 3
        };

        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        displayResponse(data);
    } catch (error) {
        showError(`Failed to get response: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayResponse(data) {
    // Render answer as markdown (or plain text if marked not available)
    const raw = data.answer || '';
    if (typeof marked !== 'undefined') {
        answerContent.innerHTML = marked.parse(raw, { gfm: true, breaks: true });
    } else {
        answerContent.textContent = raw;
    }
    responseSection.classList.remove('hidden');

    // Display evidence cards
    if (data.evidence && data.evidence.length > 0) {
        evidenceList.innerHTML = '';
        data.evidence.forEach((ev, index) => {
            const card = createEvidenceCard(ev, index + 1);
            evidenceList.appendChild(card);
        });
        evidenceSection.classList.remove('hidden');
    } else {
        evidenceSection.classList.add('hidden');
    }
}

function createEvidenceCard(ev, index) {
    const card = document.createElement('div');
    card.className = 'evidence-card';
    
    const title = ev.paper_title || 'Unknown Title';
    const doi = ev.doi || '';
    const sentence = ev.sentence || '';
    const year = ev.year != null && ev.year !== '' ? String(ev.year) : '';
    
    const yearPill = year ? `<span class="evidence-pill evidence-pill-year">${escapeHtml(year)}</span>` : '';
    const doiUrl = doi && (doi.startsWith('http://') || doi.startsWith('https://')) ? doi : doi ? `https://doi.org/${doi}` : '';
    const doiPill = doiUrl
        ? `<a href="${escapeHtml(doiUrl)}" target="_blank" rel="noopener noreferrer" class="evidence-pill evidence-pill-doi"><span class="evidence-pill-icon" aria-hidden="true">↗</span> ${escapeHtml(doi)}</a>`
        : '';
    const pills = [yearPill, doiPill].filter(Boolean).join('');
    
    card.innerHTML = `
        <div class="evidence-header">
            <span class="evidence-index">#${index}</span>
            <div class="evidence-pills">${pills}</div>
        </div>
        <div class="evidence-body">
            <h4 class="evidence-title">${escapeHtml(title)}</h4>
            <p class="evidence-sentence">${escapeHtml(sentence)}</p>
        </div>
    `;
    
    return card;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showLoading() {
    loadingIndicator.classList.remove('hidden');
}

function hideLoading() {
    loadingIndicator.classList.add('hidden');
}

function showError(message) {
    errorContent.textContent = message;
    errorSection.classList.remove('hidden');
}

function hideAllSections() {
    responseSection.classList.add('hidden');
    errorSection.classList.add('hidden');
}

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('API is healthy');
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('Could not connect to API:', error.message);
    }
}

// Check health when page loads
checkHealth();
