import React from "react";
import { User } from "lucide-react";

export default function Header() {
  return (
    <header className="flex justify-between items-center bg-white shadow px-6 py-4">
      <h1 className="text-2xl font-semibold text-gray-800">BuildFailBot Dashboard</h1>
      <div className="flex items-center space-x-4">
        <span className="text-gray-600">Welcome, Admin</span>
        <User className="w-6 h-6 text-gray-600" />
      </div>
    </header>
  );
}
