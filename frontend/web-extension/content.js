(() => {
  console.log("🧩 BuildFailBot content script loaded.");

  // Run only on GitHub Actions "runs" pages
  if (!window.location.href.includes("/actions/runs/")) {
    console.log("⚠️ Not a GitHub Actions run page — skipping.");
    return;
  }

  let buildStatus = null;
  let lastAnalyzedStatus = null;
  let autoAnalysisEnabled = true;

  // Detect build status from page
  const detectBuildStatus = () => {
    // Try multiple selectors for GitHub Actions status
    const statusSelectors = [
      'span[title*="failed"]',
      'span[title*="success"]',
      'span[title*="cancelled"]',
      '.status-badge[data-status*="failure"]',
      '.status-badge[data-status*="success"]',
      '[data-testid*="status"]',
      'svg[aria-label*="failed"]',
      'svg[aria-label*="success"]',
    ];

    for (const selector of statusSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        const text = element.textContent?.toLowerCase() || element.getAttribute('title')?.toLowerCase() || element.getAttribute('aria-label')?.toLowerCase() || '';
        if (text.includes('failed') || text.includes('failure')) {
          return 'failed';
        }
        if (text.includes('success') || text.includes('succeeded')) {
          return 'success';
        }
        if (text.includes('cancelled') || text.includes('canceled')) {
          return 'cancelled';
        }
      }
    }

    // Fallback: check for common failure indicators
    const pageText = document.body.textContent?.toLowerCase() || '';
    if (pageText.includes('failed') && !pageText.includes('success')) {
      return 'failed';
    }
    if (pageText.includes('succeeded') || pageText.includes('success')) {
      return 'success';
    }

    return null;
  };

  // Extract repository name from URL
  const extractRepoFromUrl = () => {
    const url = window.location.href;
    // Match: https://github.com/owner/repo/actions/runs/123
    const match = url.match(/github\.com\/([^\/]+\/[^\/]+)\//);
    return match ? match[1] : null;
  };

  // Smart log extraction - focus on errors only
  const extractLogs = (smartMode = true) => {
    let logs = null;

    // Try to find log containers - GitHub Actions specific selectors
    const logSelectors = [
      // GitHub Actions log containers
      'div[data-testid="log-line"]',
      'div[data-testid="log-line"] pre',
      'div[class*="log-line"]',
      'div[class*="log-line"] pre',
      'div[class*="log-content"]',
      'div[class*="log-content"] pre',
      'div[class*="ansi"]',
      'div[class*="ansi"] pre',
      // Generic selectors
      'pre',
      'div[data-test-id*="log"] pre',
      '.log-line',
      '.log-content pre',
      '[class*="log"] pre',
      'code[class*="log"]',
      // GitHub Actions specific
      'div[role="log"]',
      'div[role="log"] pre',
      'div[aria-label*="log"]',
      'div[aria-label*="log"] pre',
    ];

    for (const selector of logSelectors) {
      const elements = document.querySelectorAll(selector);
      if (elements.length > 0) {
        // Combine all matching elements
        logs = Array.from(elements)
          .map((el) => {
            const text = el.innerText || el.textContent || '';
            return text.trim();
          })
          .filter(text => text.length > 0)
          .join("\n");
        
        if (logs && logs.length >= 50) {
          console.log(`✅ Found logs using selector: ${selector} (${logs.length} chars)`);
          break;
        }
      }
    }

    // Fallback: try to get text from the entire page if no specific log container found
    if (!logs || logs.length < 50) {
      // Look for any large text blocks that might be logs
      const allPreElements = document.querySelectorAll('pre');
      if (allPreElements.length > 0) {
        logs = Array.from(allPreElements)
          .map((el) => el.innerText || el.textContent || '')
          .filter(text => text.length > 20)
          .join("\n");
      }
    }

    if (!logs || logs.length < 50) {
      console.warn("⚠️ No logs found. Available selectors:", logSelectors);
      return null;
    }

    // Smart extraction: focus on error sections
    if (smartMode) {
      const lines = logs.split('\n');
      const errorLines = [];
      let inErrorSection = false;
      let errorContext = 0;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].toLowerCase();
        
        // Detect error indicators
        const isErrorLine = 
          line.includes('error') ||
          line.includes('failed') ||
          line.includes('exception') ||
          line.includes('traceback') ||
          line.includes('fatal') ||
          line.includes('failed:') ||
          line.includes('✖') ||
          line.includes('❌') ||
          line.match(/error:\s/i) ||
          line.match(/failed:\s/i);

        if (isErrorLine) {
          inErrorSection = true;
          errorContext = 5; // Keep 5 lines after error
          errorLines.push(lines[i]);
        } else if (inErrorSection && errorContext > 0) {
          errorLines.push(lines[i]);
          errorContext--;
        } else if (inErrorSection && errorContext === 0) {
          inErrorSection = false;
        }
      }

      // If we found error lines, use them; otherwise use all logs
      if (errorLines.length > 10) {
        logs = errorLines.join('\n');
        console.log(`📜 Smart extraction: ${errorLines.length} error-focused lines`);
      }
    }

    return logs;
  };

  // Retry until logs appear
  const waitForLogs = async (maxRetries = 15, interval = 1000, smartMode = true) => {
    for (let i = 0; i < maxRetries; i++) {
      const logs = extractLogs(smartMode);
      if (logs && logs.length > 50) return logs;
      await new Promise((r) => setTimeout(r, interval));
    }
    return null;
  };

  // Monitor build status changes
  const monitorBuildStatus = () => {
    const currentStatus = detectBuildStatus();
    
    if (currentStatus !== buildStatus) {
      buildStatus = currentStatus;
      console.log(`🔍 Build status detected: ${buildStatus}`);

      // Notify background script of status change
      chrome.runtime.sendMessage({
        action: "build_status_changed",
        status: buildStatus,
        url: window.location.href
      });

      // Auto-analyze if build failed and auto-analysis is enabled
      if (currentStatus === 'failed' && autoAnalysisEnabled && lastAnalyzedStatus !== 'failed') {
        console.log("🚨 Build failed! Auto-analyzing...");
        lastAnalyzedStatus = 'failed';
        
        waitForLogs(20, 500, true).then((logs) => {
          if (logs) {
            chrome.runtime.sendMessage({
              action: "auto_analyze",
              logs: logs,
              url: window.location.href,
              status: 'failed'
            });
          }
        });
      }
    }
  };

  // Start monitoring
  const observer = new MutationObserver(() => {
    monitorBuildStatus();
  });

  // Initial check
  monitorBuildStatus();

  // Observe DOM changes for dynamic content
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'data-status', 'title', 'aria-label']
  });

  // Check periodically as fallback
  setInterval(monitorBuildStatus, 2000);

  // Listen for popup messages
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "get_logs") {
      console.log("📨 Message received from popup:", msg);
      const smartMode = msg.smart !== false;
      
      // Extract repo name from URL
      const repoName = extractRepoFromUrl();
      const runIdMatch = window.location.href.match(/\/actions\/runs\/(\d+)/);
      const runId = runIdMatch ? runIdMatch[1] : null;
      
      // Use promise to handle async properly
      waitForLogs(15, 1000, smartMode)
        .then((logs) => {
          if (logs) {
            console.log(`✅ Logs fetched (${logs.length} chars).`);
            sendResponse({ 
              logs, 
              status: buildStatus,
              repo: repoName,
              run_id: runId
            });
          } else {
            console.warn("❌ Could not find logs after waiting.");
            sendResponse({ 
              logs: null, 
              status: buildStatus,
              repo: repoName,
              run_id: runId
            });
          }
        })
        .catch((err) => {
          console.error("Error fetching logs:", err);
          const errorMsg = err?.message || String(err);
          sendResponse({ 
            logs: null, 
            error: errorMsg, 
            status: buildStatus,
            repo: repoName,
            run_id: runId
          });
        });
      
      return true; // Keep channel open for async response
    }

    if (msg.action === "get_status") {
      sendResponse({ status: buildStatus });
      return false; // Synchronous response
    }

    if (msg.action === "toggle_auto_analysis") {
      autoAnalysisEnabled = msg.enabled !== false;
      sendResponse({ enabled: autoAnalysisEnabled });
      return false; // Synchronous response
    }

    // Return false for unhandled messages
    return false;
  });
})();
