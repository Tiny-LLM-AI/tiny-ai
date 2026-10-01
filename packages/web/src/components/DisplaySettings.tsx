import { createContext, useContext } from "react";

export interface DisplaySettings {
  /** Print the number inside every heatmap cell (otherwise only on hover). */
  showCellValues: boolean;
}

export const DisplaySettingsContext = createContext<DisplaySettings>({ showCellValues: true });

export function useDisplaySettings(): DisplaySettings {
  return useContext(DisplaySettingsContext);
}
