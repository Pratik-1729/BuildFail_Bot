import React, { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import StatCard from "../components/StatCard";
import ChartsSection from "../components/ChartsSection";
import LogsTable from "../components/LogsTable";
import FeedbackTable from "../components/FeedbackTable";
import { API_ENDPOINTS } from "../config/api";

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState(localStorage.getItem("activeTab") || "dashboard");
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [retrainStatus, setRetrainStatus] = useState(null);
  const [retrainHistory, setRetrainHistory] = useState([]);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [pollInterval, setPollInterval] = useState(null);

  // ----------------------------------------------------------------------
  // Fetch Logs for Dashboard
  // ----------------------------------------------------------------------
  const fetchLogs = async () => {
    setLoading(true);
    try {
      const res = await fetch(API_ENDPOINTS.DASHBOARD_LOGS);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setLogs(data.logs || []);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Failed to fetch logs:", err);
      setLogs([]); // Set empty array on error
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------------------------------------
  // Fetch Retraining History
  // ----------------------------------------------------------------------
  const fetchRetrainHistory = async () => {
    try {
      const res = await fetch(API_ENDPOINTS.RETRAIN_HISTORY);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setRetrainHistory(data.history || []);
    } catch (err) {
      console.error("Failed to fetch retrain history:", err);
      setRetrainHistory([]); // Set empty array on error
    }
  };

  // ----------------------------------------------------------------------
  // Trigger Retraining
  // ----------------------------------------------------------------------
  const handleRetrain = async () => {
    setRetrainLoading(true);
    setRetrainStatus("Retraining model... please wait ⏳");

    try {
      const res = await fetch(API_ENDPOINTS.RETRAIN, {
        method: "POST",
      });
      const data = await res.json();
      setRetrainStatus(data.message || "Retraining started successfully!");

      // Poll retrain history until it updates
      const interval = setInterval(async () => {
        const res = await fetch(API_ENDPOINTS.RETRAIN_HISTORY);
        const data = await res.json();
        setRetrainHistory(data.history || []);
        const lastEntry = data.history[data.history.length - 1];
        if (lastEntry && lastEntry.status === "success") {
          clearInterval(interval);
          setRetrainStatus("Retraining completed successfully!");
          setRetrainLoading(false);
        } else if (lastEntry && lastEntry.status === "failed") {
          clearInterval(interval);
          setRetrainStatus("Retraining failed. Check logs for details.");
          setRetrainLoading(false);
        }
      }, 20000); // 20s polling interval

      setPollInterval(interval);
    } catch (err) {
      console.error("Retrain error:", err);
      setRetrainStatus("Failed to start retraining");
      setRetrainLoading(false);
    }
  };

  // ----------------------------------------------------------------------
  // Initial Data Load
  // ----------------------------------------------------------------------
  useEffect(() => {
    fetchLogs();
    fetchRetrainHistory();
    const interval = setInterval(fetchLogs, 30000); // refresh every 30s
    return () => {
      clearInterval(interval);
      if (pollInterval) clearInterval(pollInterval);
    };
  }, []);

  // ----------------------------------------------------------------------
  // Basic Stats
  // ----------------------------------------------------------------------
  const total = logs.length;
  const success = logs.filter((l) => l.label === "success").length;
  const failed = logs.filter((l) => l.label === "failed").length;
  const skipped = logs.filter((l) => l.label === "skipped").length;

  // ----------------------------------------------------------------------
  // Render Retraining History Table
  // ----------------------------------------------------------------------
  const renderRetrainHistory = () => (
    <div className="mt-6">
      <h2 className="text-lg font-semibold mb-4">Retraining History</h2>
      {retrainHistory.length === 0 ? (
        <p className="text-gray-500">No retraining history found.</p>
      ) : (
        <div className="overflow-x-auto bg-white shadow rounded-lg">
          <table className="min-w-full text-sm text-left border">
            <thead className="bg-gray-100 text-gray-700">
              <tr>
                <th className="px-4 py-2 border">Start Time (UTC)</th>
                <th className="px-4 py-2 border">Status</th>
                <th className="px-4 py-2 border">Accuracy</th>
                <th className="px-4 py-2 border">F1 Score</th>
                <th className="px-4 py-2 border">Active Model</th>
                <th className="px-4 py-2 border">Steps</th>
                <th className="px-4 py-2 border">End Time</th>
              </tr>
            </thead>
            <tbody>
              {retrainHistory
                .slice()
                .reverse()
                .map((entry, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-2 border">{entry.start_time}</td>
                    <td
                      className={`px-4 py-2 border font-semibold ${
                        entry.status === "success"
                          ? "text-green-600"
                          : entry.status === "failed"
                          ? "text-red-600"
                          : "text-yellow-600"
                      }`}
                    >
                      {entry.status}
                    </td>
                    <td className="px-4 py-2 border">
                      {entry.metrics && typeof entry.metrics.accuracy === 'number' 
                        ? entry.metrics.accuracy.toFixed(4) 
                        : "-"}
                    </td>
                    <td className="px-4 py-2 border">
                      {entry.metrics && typeof entry.metrics.f1_score === 'number' 
                        ? entry.metrics.f1_score.toFixed(4) 
                        : "-"}
                    </td>
                    <td className="px-4 py-2 border">
                      {entry.metrics ? entry.metrics.active_model : "-"}
                    </td>
                    <td className="px-4 py-2 border">
                      {entry.steps ? entry.steps.join(", ") : "-"}
                    </td>
                    <td className="px-4 py-2 border">{entry.end_time || "-"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  // ----------------------------------------------------------------------
  // Main Render
  // ----------------------------------------------------------------------
  return (
    <div className="flex h-screen">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 flex flex-col bg-gray-50">
        <Header />
        <div className="p-6 overflow-y-auto">
          {/* Last updated indicator */}
          <div className="text-right text-sm text-gray-500 mb-4">
            {lastUpdated && (
              <span>
                🔄 Last updated at <strong>{lastUpdated}</strong>
              </span>
            )}
          </div>

          {/* Dashboard Tab */}
          {activeTab === "dashboard" && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard title="Total Logs" value={total} color="blue" />
                <StatCard title="Successful Builds" value={success} color="green" />
                <StatCard title="Failed Builds" value={failed} color="red" />
              </div>
              <ChartsSection logs={logs} />
            </>
          )}

          {/* Logs Tab */}
          {activeTab === "logs" && <LogsTable logs={logs} loading={loading} />}

          {/* Feedback Tab */}
          {activeTab === "feedback" && <FeedbackTable />}

          {/* Retrain Tab */}
          {activeTab === "retrain" && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Model Retraining</h2>

              <button
                onClick={handleRetrain}
                disabled={retrainLoading}
                className={`px-6 py-2 rounded-lg text-white font-medium ${
                  retrainLoading ? "bg-gray-400" : "bg-indigo-600 hover:bg-indigo-700"
                }`}
              >
                {retrainLoading ? "Retraining..." : "Start Retraining"}
              </button>

              {retrainStatus && (
                <p className="mt-4 text-gray-700">
                  <strong>Status:</strong> {retrainStatus}
                </p>
              )}

              {renderRetrainHistory()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
