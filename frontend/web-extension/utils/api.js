import { API_ENDPOINTS } from './config.js';

export async function analyzeLog(logText) {
  const response = await fetch(API_ENDPOINTS.PREDICT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: logText }),
  });
  return response.json();
}

export async function sendFeedback(log_excerpt, label, confidence, rating) {
  await fetch(API_ENDPOINTS.FEEDBACK, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      log_excerpt,
      label,
      confidence,
      rating
    }),
  });
}