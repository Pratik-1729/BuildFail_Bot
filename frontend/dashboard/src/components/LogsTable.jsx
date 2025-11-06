import React from "react";

export default function LogsTable({ logs, loading }) {
  if (loading) return <p className="text-gray-500 p-4">Loading logs...</p>;
  if (!logs || logs.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center">
        <p className="text-gray-500">No logs available yet.</p>
        <p className="text-sm text-gray-400 mt-2">
          Logs will appear here once you start ingesting them via webhook or manual analysis.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-4 overflow-auto">
      <h2 className="text-xl font-semibold mb-4">Ingested Logs</h2>

      <table className="min-w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-100 text-gray-700 uppercase text-xs">
            <th className="p-2 text-left">Timestamp</th>
            <th className="p-2 text-left">Repository</th>
            <th className="p-2 text-left">Run ID</th>
            <th className="p-2 text-left">Status</th>
            <th className="p-2 text-left">Label</th>
            <th className="p-2 text-left">Confidence</th>
            <th className="p-2 text-left">Failed Step</th>
            <th className="p-2 text-left">Suggestion</th>
            <th className="p-2 text-left">Excerpt</th>
          </tr>
        </thead>

        <tbody>
          {logs.map((log, i) => (
            <tr key={i} className="border-b hover:bg-gray-50 transition">
              <td className="p-2 text-gray-700">
                {log.timestamp ? new Date(log.timestamp).toLocaleString() : "-"}
              </td>
              <td className="p-2 text-blue-600 font-medium">{log.repo || "-"}</td>
              <td className="p-2">{log.run_id || "-"}</td>
              <td
                className={`p-2 font-semibold ${
                  log.status === "failure"
                    ? "text-red-600"
                    : log.status === "success"
                    ? "text-green-600"
                    : "text-gray-600"
                }`}
              >
                {log.status || "-"}
              </td>
              <td className="p-2">{log.label || "-"}</td>
              <td className="p-2">{(log.confidence * 100).toFixed(2)}%</td>
              <td className="p-2">{log.failed_step || "-"}</td>
              <td className="p-2 text-gray-700">{log.suggestion || "-"}</td>
              <td className="p-2 text-gray-600 max-w-lg truncate">
                {log.clean_log_excerpt || "-"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
