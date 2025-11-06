import React, { useEffect, useState } from "react";
import { API_ENDPOINTS } from "../config/api";

export default function FeedbackTable() {
  const [feedback, setFeedback] = useState([]);
  const [summary, setSummary] = useState({ average_rating: 0, total_feedback: 0 });
  const [loading, setLoading] = useState(false);

  const fetchFeedback = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_ENDPOINTS.FEEDBACK);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setFeedback(data.feedback || []);
      setSummary(data.summary || { average_rating: 0, total_feedback: 0 });
    } catch (err) {
      console.error("Error fetching feedback:", err);
      setFeedback([]); // Set empty array on error
      setSummary({ average_rating: 0, total_feedback: 0 });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeedback();
    const interval = setInterval(fetchFeedback, 30000); // auto-refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) return <p className="text-gray-500">Loading feedback...</p>;
  if (!feedback.length)
    return (
      <p className="text-gray-500 text-center mt-6">
        No feedback available yet.
      </p>
    );

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Summary Header */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-gray-800">User Feedback</h2>
        <div className="flex items-center space-x-6">
          <div className="text-center">
            <p className="text-gray-500 text-sm">Average Rating</p>
            <p className="text-xl font-bold text-yellow-500">
              ⭐ {summary.average_rating?.toFixed(2) || 0}
            </p>
          </div>
          <div className="text-center">
            <p className="text-gray-500 text-sm">Total Feedback</p>
            <p className="text-xl font-bold text-gray-700">
              {summary.total_feedback}
            </p>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-5">
        <div
          className="bg-yellow-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${(summary.average_rating / 5) * 100}%` }}
        ></div>
      </div>

      {/* Feedback Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="bg-gray-100 text-gray-600 uppercase text-xs">
              <th className="p-2 text-left">Timestamp</th>
              <th className="p-2 text-left">Label</th>
              <th className="p-2 text-left">Confidence</th>
              <th className="p-2 text-left">Rating</th>
              <th className="p-2 text-left">Log Excerpt</th>
            </tr>
          </thead>
          <tbody>
            {feedback.map((f, i) => (
              <tr
                key={i}
                className="border-b hover:bg-gray-50 transition text-gray-700"
              >
                <td className="p-2">{new Date(f.timestamp).toLocaleString()}</td>
                <td className="p-2 capitalize">{f.label}</td>
                <td className="p-2">{(f.confidence * 100).toFixed(2)}%</td>
                <td className="p-2 text-yellow-600 font-medium">
                  ⭐ {f.rating || "-"}
                </td>
                <td className="p-2 text-gray-600">
                  {f.log_excerpt?.slice(0, 80)}...
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
