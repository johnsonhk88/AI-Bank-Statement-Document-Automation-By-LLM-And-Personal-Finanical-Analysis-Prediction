type DateRangeFilterProps = {
  dateFrom: string;
  dateTo: string;
  currency: string;
  onDateFromChange: (value: string) => void;
  onDateToChange: (value: string) => void;
  onCurrencyChange: (value: string) => void;
};

const PRESET_RANGES = [
  { label: "1M", months: 1 },
  { label: "3M", months: 3 },
  { label: "6M", months: 6 },
  { label: "1Y", months: 12 },
  { label: "3Y", months: 36 },
  { label: "5Y", months: 60 },
  { label: "All", months: 120 },
];

const CURRENCIES = ["HKD", "USD", "EUR", "GBP", "CNY", "JPY"];

function subtractMonths(dateStr: string, months: number): string {
  const d = new Date(dateStr);
  d.setMonth(d.getMonth() - months);
  return d.toISOString().slice(0, 10);
}

export function DateRangeFilter({
  dateFrom,
  dateTo,
  currency,
  onDateFromChange,
  onDateToChange,
  onCurrencyChange,
}: DateRangeFilterProps) {
  const handlePreset = (months: number) => {
    const today = new Date().toISOString().slice(0, 10);
    onDateToChange(today);
    onDateFromChange(subtractMonths(today, months));
  };

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-1">
        {PRESET_RANGES.map((r) => (
          <button
            key={r.label}
            type="button"
            onClick={() => handlePreset(r.months)}
            className="px-3 py-1 text-xs font-medium rounded-full border border-gray-300 hover:bg-blue-50 hover:border-blue-400 hover:text-blue-700 transition-colors"
          >
            {r.label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <input
          type="date"
          value={dateFrom}
          onChange={(e) => onDateFromChange(e.target.value)}
          className="text-sm border border-gray-300 rounded px-2 py-1"
        />
        <span className="text-gray-400">to</span>
        <input
          type="date"
          value={dateTo}
          onChange={(e) => onDateToChange(e.target.value)}
          className="text-sm border border-gray-300 rounded px-2 py-1"
        />
      </div>
      <select
        value={currency}
        onChange={(e) => onCurrencyChange(e.target.value)}
        className="text-sm border border-gray-300 rounded px-2 py-1"
      >
        {CURRENCIES.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </div>
  );
}
