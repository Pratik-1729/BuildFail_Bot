import React, { useMemo } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  LineChart,
  Line,
} from "recharts";

const COLORS = {
  failed: "#f87171",
  success: "#34d399",
  skipped: "#facc15",
  unknown: "#9ca3af",
};

export default function ChartsSection({ logs }) {
  // --- Aggregation logic ---
  const { chartData, trendData } = useMemo(() => {
    const counts = { failed: 0, success: 0, skipped: 0, unknown: 0 };
    const dailyCounts = {};

    logs.forEach((log) => {
      const label = log.label?.toLowerCase();
      const date = log.timestamp ? log.timestamp.split("T")[0] : "unknown";

      // Count by label
      if (counts[label] !== undefined) counts[label]++;
      else counts.unknown++;

      // Count by date for trend
      if (!dailyCounts[date]) dailyCounts[date] = { date, failed: 0, success: 0, skipped: 0 };
      if (label in dailyCounts[date]) dailyCounts[date][label]++;
    });

    const chartData = [
      { name: "Failed", value: counts.failed },
      { name: "Success", value: counts.success },
      { name: "Skipped", value: counts.skipped },
      { name: "Unknown", value: counts.unknown },
    ];

    const trendData = Object.values(dailyCounts).sort((a, b) => new Date(a.date) - new Date(b.date));
    return { chartData, trendData };
  }, [logs]);

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mt-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800"> Build Insights</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Pie Chart */}
        <div className="h-80">
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={Object.values(COLORS)[index]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div className="h-80">
          <ResponsiveContainer>
            <BarChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#2563eb">
                {chartData.map((entry, index) => (
                  <Cell key={`bar-${index}`} fill={Object.values(COLORS)[index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trend Chart */}
      <div className="h-80 mt-10">
        <h3 className="text-lg font-medium text-gray-700 mb-3">Build Trends Over Time</h3>
        <ResponsiveContainer>
          <LineChart data={trendData}>
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="failed" stroke={COLORS.failed} strokeWidth={2} />
            <Line type="monotone" dataKey="success" stroke={COLORS.success} strokeWidth={2} />
            <Line type="monotone" dataKey="skipped" stroke={COLORS.skipped} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
