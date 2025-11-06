import React from "react";

export default function StatCard({ title, value, color }) {
  const colorMap = {
    blue: "bg-blue-600",
    green: "bg-green-600",
    red: "bg-red-600",
    yellow: "bg-yellow-500",
    gray: "bg-gray-500",
  };

  return (
    <div className={`p-4 rounded-xl text-white shadow-md ${colorMap[color] || "bg-gray-500"}`}>
      <h2 className="text-lg font-medium">{title}</h2>
      <p className="text-3xl font-bold mt-2">{value}</p>
    </div>
  );
}
