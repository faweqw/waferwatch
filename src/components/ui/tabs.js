// tabs.js
import { useState } from 'react';

export function Tabs({ value, onValueChange, children }) {
  return <div>{children}</div>;
}

export function TabsList({ children, className = '' }) {
  return <div className={className}>{children}</div>;
}

export function TabsTrigger({ value, children, ...props }) {
  return (
    <button value={value} {...props} className="px-4 py-2 border rounded">
      {children}
    </button>
  );
}

export function TabsContent({ value, children }) {
  return <div className="mt-2">{children}</div>;
}
