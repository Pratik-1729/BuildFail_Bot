// Status badge management
let currentStatus = null;

function updateBadge(status) {
  if (status === 'failed') {
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#dc2626' }); // red
  } else if (status === 'success') {
    chrome.action.setBadgeText({ text: '✓' });
    chrome.action.setBadgeBackgroundColor({ color: '#16a34a' }); // green
  } else {
    chrome.action.setBadgeText({ text: '' });
  }
  currentStatus = status;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "fetch_logs_from_page") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || tabs.length === 0) {
        sendResponse({ logs: null, error: "No active tab found" });
        return;
      }

      const tab = tabs[0];
      
      // Check if tab is a GitHub Actions page
      if (!tab.url || !tab.url.includes('github.com')) {
        sendResponse({ logs: null, error: "Not a GitHub page" });
        return;
      }

      chrome.tabs.sendMessage(tab.id, { action: "get_logs" }, (response) => {
        // Check for errors
        if (chrome.runtime.lastError) {
          const errorMsg = chrome.runtime.lastError.message || String(chrome.runtime.lastError);
          console.error("Error sending message to content script:", errorMsg);
          sendResponse({ logs: null, error: errorMsg });
          return;
        }

        // Handle response
        if (response) {
          // Update badge based on status
          if (response.status) {
            updateBadge(response.status);
          }
          // Forward all response data including repo, run_id, etc.
          sendResponse(response);
        } else {
          sendResponse({ logs: null, error: "No response from content script" });
        }
      });
    });
    return true; // keep async channel open
  }

  if (message.action === "build_status_changed") {
    console.log(`🔍 Build status changed: ${message.status}`);
    updateBadge(message.status);
    
    // Show notification for failed builds
    if (message.status === 'failed') {
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: 'Build Failed',
        message: 'A build has failed. Click to analyze.',
        priority: 2
      }).catch(err => {
        console.warn("Could not create notification:", err);
      });
    }
    sendResponse({ success: true });
    return true;
  }

  if (message.action === "auto_analyze") {
    console.log("🚨 Auto-analyzing failed build...");
    // Store analysis request for popup to pick up
    chrome.storage.local.set({ 
      autoAnalysis: {
        logs: message.logs,
        url: message.url,
        status: message.status,
        timestamp: Date.now()
      }
    });
    
    // Update badge
    updateBadge('failed');
    
    sendResponse({ success: true });
    return true;
  }
});
