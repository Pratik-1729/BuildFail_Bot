import React, { useState, useEffect } from "react";
import { BarChart3, Activity, FileText, MessageSquare, RefreshCcw } from "lucide-react";

const Sidebar = ({ activeTab, setActiveTab }) => {
  const [selected, setSelected] = useState(activeTab);

  useEffect(() => {
    const saved = localStorage.getItem("activeTab");
    if (saved) setSelected(saved);
  }, []);

  const handleSelect = (tab) => {
    setSelected(tab);
    setActiveTab(tab);
    localStorage.setItem("activeTab", tab);
  };

  const linkClasses = (tab) =>
    `flex items-center px-6 py-3 cursor-pointer transition-all ${
      selected === tab
        ? "bg-blue-600 text-white"
        : "text-gray-300 hover:bg-gray-700 hover:text-white"
    }`;

  return (
    <div className="w-64 bg-gray-900 text-white flex flex-col h-screen">
      {/* Sidebar Header */}
      <div className="text-2xl font-bold px-6 py-4 border-b border-gray-700">
        BuildFailBot
      </div>

      {/* Navigation Tabs */}
      <nav className="flex-1 mt-4 space-y-1">
        <div
          className={linkClasses("dashboard")}
          onClick={() => handleSelect("dashboard")}
        >
          <BarChart3 className="mr-3" /> Dashboard
        </div>

        <div
          className={linkClasses("logs")}
          onClick={() => handleSelect("logs")}
        >
          <FileText className="mr-3" /> Logs
        </div>

        <div
          className={linkClasses("feedback")}
          onClick={() => handleSelect("feedback")}
        >
          <MessageSquare className="mr-3" /> Feedback
        </div>

        {/* 🧠 New Retrain Model Tab */}
        <div
          className={linkClasses("retrain")}
          onClick={() => handleSelect("retrain")}
        >
          <RefreshCcw className="mr-3" /> Retrain Model
        </div>
      </nav>

      {/* Footer */}
      <footer className="text-xs text-gray-400 text-center pb-4 border-t border-gray-700">
        © 2025 BuildFailBot
      </footer>
    </div>
  );
};

export default Sidebar;
