"""CSS and reusable HTML fragments for extraction reports."""

REPORT_CSS = """
<style>
.extraction-report {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    color: #0f172a;
    background: #f8fafc;
    padding: 18px;
    line-height: 1.5;
}
.er-container { max-width: 1200px; margin: 0 auto; }

/* Hero banner */
.er-hero {
    background: linear-gradient(135deg, #e0f2fe, #d1fae5);
    color: #0f172a;
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 12px 28px rgba(15,23,42,0.08);
    margin-bottom: 18px;
}
.er-hero h2 { margin: 0 0 4px 0; font-size: 22px; color: #0b1727; }
.er-hero .er-subtitle { font-size: 13px; color: #475569; margin-bottom: 12px; }

/* Corporate mark — one per surface, opposite the title (docs/internal/19, §26).
   Wraps under the title on narrow viewports rather than crushing the subtitle. */
.er-hero-head {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 16px; flex-wrap: wrap;
}
.er-hero-titles { min-width: 0; }
.er-brand { display: inline-flex; align-items: center; text-decoration: none;
    color: #0b1727; font-weight: 700; font-size: 15px; flex: none; }
/* Keep in step with _LOGO_RENDER_HEIGHT; the packaged asset is 2x this. */
.er-brand img { display: block; height: 56px; width: auto; }

/* Stats grid */
.er-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px;
    margin-bottom: 6px;
}
.er-stat-card {
    background: #fff;
    border-radius: 12px;
    padding: 10px 14px;
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 4px 12px rgba(15,23,42,0.05);
}
.er-stat-card .label {
    color: #475569; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.er-stat-card .value {
    font-weight: 700; font-size: 18px; color: #0f172a;
}
.er-stat-card .value.success { color: #059669; }
.er-stat-card .value.danger { color: #dc2626; }

/* Success rate bar */
.er-rate-bar {
    height: 8px; border-radius: 4px; background: #e2e8f0;
    margin-top: 6px; overflow: hidden;
}
.er-rate-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #059669, #10b981);
    transition: width 0.3s;
}

/* Section cards */
.er-section {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 16px;
    background: #fff;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
    margin-bottom: 14px;
}
.er-section h3 {
    margin: 0 0 10px 0; font-size: 15px; color: #1e293b;
}

/* Data table */
.er-table {
    width: 100%; border-collapse: collapse; font-size: 12px;
    overflow-x: auto; display: block;
}
.er-table th {
    background: #f1f5f9; color: #475569; text-align: left;
    padding: 6px 10px; border-bottom: 2px solid #e2e8f0;
    position: sticky; top: 0; font-weight: 600;
}
.er-table td {
    padding: 5px 10px; border-bottom: 1px solid #f1f5f9;
    max-width: 250px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
}
.er-table tr:hover td { background: #f8fafc; }

/* Error rows */
.er-error-row td { color: #991b1b; }

/* Parameters */
.er-params pre {
    background: #f8fafc; padding: 10px; border-radius: 8px;
    overflow: auto; font-size: 12px; margin: 4px 0;
}
.er-params summary {
    cursor: pointer; color: #2563eb; font-weight: 600; font-size: 13px;
}

/* Map container */
.er-map-card {
    max-width: 900px; margin: 0 auto 16px auto;
    border: 1px solid #e2e8f0; border-radius: 14px;
    overflow: hidden; background: #fff;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}
.er-map-card h3 {
    margin: 0; padding: 10px 16px; font-size: 14px; color: #1e293b;
    border-bottom: 1px solid #f1f5f9;
}

/* Chips */
.er-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.er-chip {
    background: #e2e8f0; border-radius: 10px; padding: 3px 8px;
    font-size: 11px; color: #334155;
}

/* Footer */
.er-footer {
    text-align: center; font-size: 11px; color: #94a3b8;
    margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0;
}
.er-footer a { color: #475569; text-decoration: underline; }
.er-footer a:hover { color: #0b1727; }
</style>
"""
