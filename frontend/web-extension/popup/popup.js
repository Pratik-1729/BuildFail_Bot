// Import API endpoints
import { API_ENDPOINTS } from '../utils/config.js';

const logInput = document.getElementById("logInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const fetchLogsBtn = document.getElementById("fetchLogs");
const resultDiv = document.getElementById("result");

// Check for auto-analysis on popup open
chrome.storage.local.get(['autoAnalysis'], (result) => {
  if (result.autoAnalysis) {
    const analysis = result.autoAnalysis;
    // Check if analysis is recent (within 5 minutes)
    if (Date.now() - analysis.timestamp < 5 * 60 * 1000) {
      logInput.value = analysis.logs;
      showMessage("🔄 Auto-analyzing failed build...", "info");
      // Auto-analyze
      setTimeout(() => {
        analyzeBtn.click();
      }, 500);
      // Clear stored analysis
      chrome.storage.local.remove(['autoAnalysis']);
    }
  }
});

// Auto-fetch logs when popup opens (if on GitHub Actions page)
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  if (tabs[0] && tabs[0].url && tabs[0].url.includes('github.com') && tabs[0].url.includes('/actions/runs/')) {
    // Auto-fetch logs
    chrome.runtime.sendMessage({ action: "fetch_logs_from_page" }, (response) => {
      if (response?.logs) {
        logInput.value = response.logs;
        const repoInfo = response.repo ? ` (Repo: ${response.repo})` : '';
        showMessage(`✅ Logs auto-fetched!${repoInfo} Click Analyze.`, "success");
      } else {
        console.warn("No logs found in response:", response);
        showMessage("⚠️ Could not auto-fetch logs. Try clicking 'Fetch Logs' or paste manually.", "warning");
      }
    });
  }
});

// Auto-fetch logs from active GitHub tab (with smart extraction)
fetchLogsBtn.onclick = () => {
  chrome.runtime.sendMessage({ action: "fetch_logs_from_page" }, (response) => {
    if (response?.logs) {
      logInput.value = response.logs;
      const status = response.status ? ` (Status: ${response.status})` : '';
      const repoInfo = response.repo ? ` (Repo: ${response.repo})` : '';
      showMessage(`✅ Logs fetched successfully!${status}${repoInfo}`, "success");
    } else {
      const errorMsg = response?.error ? ` (${response.error})` : '';
      showMessage(`⚠️ No logs found${errorMsg} — please paste manually.`, "warning");
    }
  });
};

// Extract repo name from GitHub URL
function extractRepoFromUrl(url) {
  if (!url) return null;
  // Match: https://github.com/owner/repo/actions/runs/123
  const match = url.match(/github\.com\/([^\/]+\/[^\/]+)\//);
  if (match) {
    return match[1]; // Returns "owner/repo"
  }
  return null;
}

// Analyze logs using backend API
analyzeBtn.onclick = async () => {
  const logText = logInput.value.trim();
  if (!logText) {
    showMessage("Please paste or fetch logs first!", "warning");
    return;
  }

  resultDiv.innerHTML = `<div class="loading">⏳ Analyzing logs...</div>`;
  resultDiv.classList.remove("hidden");

  // Get current tab URL to extract repo name
  let repoName = null;
  let runId = null;
  let status = null;
  
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0] && tabs[0].url) {
      repoName = extractRepoFromUrl(tabs[0].url);
      // Extract run ID from URL: /actions/runs/123456
      const runIdMatch = tabs[0].url.match(/\/actions\/runs\/(\d+)/);
      if (runIdMatch) {
        runId = runIdMatch[1];
      }
    }
  } catch (err) {
    console.warn("Could not get tab URL:", err);
  }

  try {
    const res = await fetch(API_ENDPOINTS.MANUAL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        log_text: logText,
        repo: repoName || undefined, // Send repo name if available
        run_id: runId || undefined,    // Send run ID if available
        status: status || undefined   // Send status if available
      }),
    });

    if (!res.ok) throw new Error("Server error");
    const data = await res.json();

    const label = data.predicted_label || "unknown";
    const conf = (data.confidence * 100).toFixed(1);
    const suggestion = data.suggestion || "No suggestion available.";
    const fixCommand = data.fix_command || null; // New: actionable fix command

    renderResult(label, conf, suggestion, logText, fixCommand);
    
    // Attach event listeners to rating buttons after rendering
    setTimeout(() => {
      document.querySelectorAll('.rating-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const label = btn.dataset.label;
          const conf = parseFloat(btn.dataset.conf);
          const rating = parseInt(btn.dataset.rating);
          const logExcerpt = btn.dataset.log;
          submitFeedback(label, conf, rating, logExcerpt);
        });
      });
    }, 100);
  } catch (err) {
    console.error(err);
    showMessage("Failed to analyze logs. Check if backend is running.", "error");
  }
};

// UI helpers
function renderResult(label, conf, suggestion, logText, fixCommand = null) {
  const color =
    label === "success"
      ? "success"
      : label === "failed"
      ? "failed"
      : label === "skipped"
      ? "skipped"
      : "unknown";

  // Build fix command section if available
  let fixCommandSection = '';
  if (fixCommand) {
    fixCommandSection = `
      <div class="fix-command-section">
        <p><strong>Quick Fix:</strong></p>
        <div class="fix-command-box">
          <code id="fixCommand">${fixCommand}</code>
          <button id="copyFixBtn" class="copy-btn" title="Copy to clipboard"> Copy</button>
        </div>
      </div>
    `;
  }

  resultDiv.innerHTML = `
    <div class="result-card ${color}">
      <h3>Prediction: <span>${label.toUpperCase()}</span></h3>
      <p><strong>Confidence:</strong> ${conf}%</p>
      <p><strong>Suggestion:</strong> ${suggestion}</p>
      ${fixCommandSection}
      <div class="feedback-section">
        <p><strong>Was this prediction helpful?</strong></p>
        <div class="rating-buttons">
          ${[1, 2, 3, 4, 5].map(r => 
            `<button class="rating-btn" data-rating="${r}" data-label="${label}" data-conf="${conf}" data-log="${logText.substring(0, 100).replace(/"/g, '&quot;')}">⭐ ${r}</button>`
          ).join('')}
        </div>
      </div>
    </div>
  `;
  
  // Attach event listeners
  setTimeout(() => {
    // Rating buttons
    document.querySelectorAll('.rating-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const label = btn.dataset.label;
        const conf = parseFloat(btn.dataset.conf);
        const rating = parseInt(btn.dataset.rating);
        const logExcerpt = btn.dataset.log;
        submitFeedback(label, conf, rating, logExcerpt);
      });
    });

    // Copy fix command button
    const copyBtn = document.getElementById('copyFixBtn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const fixCmd = document.getElementById('fixCommand').textContent;
        navigator.clipboard.writeText(fixCmd).then(() => {
          copyBtn.textContent = '✓ Copied!';
          copyBtn.style.background = '#16a34a';
          setTimeout(() => {
            copyBtn.textContent = 'Copy';
            copyBtn.style.background = '';
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy:', err);
          showMessage('Failed to copy command', 'error');
        });
      });
    }
  }, 100);
}

// Submit feedback to backend
async function submitFeedback(label, confidence, rating, logExcerpt) {
  try {
    // Use endpoint with trailing slash to match backend route
    const feedbackUrl = API_ENDPOINTS.FEEDBACK.endsWith('/') 
      ? API_ENDPOINTS.FEEDBACK 
      : API_ENDPOINTS.FEEDBACK + '/';
    
    const response = await fetch(feedbackUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label,
        confidence: confidence / 100,
        log_excerpt: logExcerpt.substring(0, 250), // Limit length
        rating
      }),
    });
    
    if (response.ok) {
      showMessage("Feedback submitted! Thank you!", "success");
    } else {
      const errorText = await response.text();
      console.error("Feedback error response:", response.status, errorText);
      showMessage(`Failed to submit feedback (${response.status})`, "warning");
    }
  } catch (err) {
    console.error("Feedback error:", err);
    showMessage("Error submitting feedback", "error");
  }
}

function showMessage(text, type = "info") {
  resultDiv.classList.remove("hidden");
  resultDiv.innerHTML = `<div class="msg ${type}">${text}</div>`;
}
