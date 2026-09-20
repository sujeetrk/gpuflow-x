import { useCallback, useEffect, useState } from 'react'
import './App.css'

const REFRESH_MS = 3000

async function fetchJson(path) {
  const response = await fetch(`/api${path}`)
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return response.json()
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return '—'
  }

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function MetricCard({ label, value, unit, detail, accent = '' }) {
  return (
    <article className={`metric-card ${accent}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className="metric-detail">{detail}</div>
    </article>
  )
}

function SectionTitle({ eyebrow, title, right }) {
  return (
    <div className="section-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
      </div>
      {right && <div className="section-right">{right}</div>}
    </div>
  )
}

function InsightList({ title, items, emptyText, type = 'warning' }) {
  return (
    <div className="copilot-column">
      <div className="column-heading">
        <span className={`column-dot ${type}`} />
        {title}
      </div>

      {items.length ? (
        items.map((item, i) => (
          <div className="insight-item" key={`${i}-${item}`}>
            <span className="item-marker">›</span>
            <span>{item}</span>
          </div>
        ))
      ) : (
        <div className="muted-copy">{emptyText}</div>
      )}
    </div>
  )
}

function App() {
  const [activePage, setActivePage] = useState('overview')
  const [metrics, setMetrics] = useState(null)
  const [copilot, setCopilot] = useState(null)
  const [connection, setConnection] = useState('connecting')
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])
  const [refreshing, setRefreshing] = useState(false)

  const refresh = useCallback(async () => {
    setRefreshing(true)

    try {
      const [metricData, copilotData] = await Promise.all([
        fetchJson('/metrics'),
        fetchJson('/ops/copilot'),
      ])

      setMetrics(metricData)
      setCopilot(copilotData)
      setConnection('connected')
      setError('')
      setLastUpdated(new Date())

      setHistory((previous) => [
        ...previous.slice(-19),
        {
          time: new Date().toLocaleTimeString([], {
            minute: '2-digit',
            second: '2-digit',
          }),
          completed: Number(metricData.requests_completed ?? 0),
          queue: Number(metricData.queue_length ?? 0),
        },
      ])
    } catch (err) {
      setConnection('disconnected')
      setError(err.message || 'Unable to reach the API')
    } finally {
      setRefreshing(false)
    }
  }, [])

  // Poll the API every 3 seconds.
  // The first request is triggered by the user or the first interval tick.
  useEffect(() => {
    const timer = setInterval(() => {
      void refresh()
    }, REFRESH_MS)

    return () => clearInterval(timer)
  }, [refresh])

  const observed = copilot?.observed_metrics ?? {}

  const submitted = Number(
    metrics?.requests_submitted ?? observed.requests_submitted ?? 0,
  )
  const completed = Number(
    metrics?.requests_completed ?? observed.requests_completed ?? 0,
  )
  const failed = Number(
    metrics?.requests_failed ?? observed.requests_failed ?? 0,
  )
  const queue = Number(
    metrics?.queue_length ?? observed.queue_length ?? 0,
  )
  const batches = Number(metrics?.batches_executed ?? 0)

  const successRate = completed + failed > 0
    ? (completed / (completed + failed)) * 100
    : null

  const latency = observed.average_latency_ms
  const queueTime = observed.average_queue_time_ms
  const inferenceTime = observed.average_inference_time_ms

  const bottlenecks = copilot?.bottlenecks ?? []
  const recommendations = copilot?.recommendations ?? []

  const pages = {
    overview: {
      label: 'Overview',
      eyebrow: 'SCHEDULER OBSERVABILITY',
      title: 'System overview',
      subtitle: 'Monitor inference workloads, scheduler activity, and AI-generated operational insights.',
    },
    scheduler: {
      label: 'Scheduler metrics',
      eyebrow: 'SCHEDULER TELEMETRY',
      title: 'Scheduler metrics',
      subtitle: 'Inspect request throughput, queue activity, batch execution, and observed latency.',
    },
    copilot: {
      label: 'AI Copilot',
      eyebrow: 'GPUFLOW INTELLIGENCE',
      title: 'AI Operations Copilot',
      subtitle: 'Review operational insights and recommendations grounded in backend-reported metrics.',
    },
  }

  const page = pages[activePage]

  const navItems = [
    { id: 'overview', label: 'Overview', icon: '▦' },
    { id: 'scheduler', label: 'Scheduler metrics', icon: '⌁' },
    { id: 'copilot', label: 'AI Copilot', icon: '✳' },
  ]

  const navigationButtonStyle = {
    width: '100%',
    border: 0,
    font: 'inherit',
    textAlign: 'left',
    cursor: 'pointer',
    background: 'transparent',
    color: 'inherit',
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">G<span>×</span></div>
          <div>
            <div className="brand-name">GPUFlow-X</div>
            <div className="brand-caption">INFERENCE CONTROL PLANE</div>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        {navItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            style={navigationButtonStyle}
            onClick={() => setActivePage(item.id)}
            aria-current={activePage === item.id ? 'page' : undefined}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}

        <div className="sidebar-bottom">
          <div className="system-label">SYSTEM STATUS</div>
          <div className="sidebar-status">
            <span className={`status-dot ${connection}`} />
            {connection === 'connected'
              ? 'API connected'
              : connection === 'disconnected'
                ? 'API unavailable'
                : 'Connecting'}
          </div>
          <div className="sidebar-version">Real-time monitor · v1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            GPUFlow-X <span>/</span> {page.label}
          </div>

          <div className="topbar-right">
            <div className={`connection-pill ${connection}`}>
              <span className="status-dot" />
              {connection === 'connected'
                ? 'LIVE'
                : connection === 'disconnected'
                  ? 'OFFLINE'
                  : 'CONNECTING'}
            </div>

            <button
              className="refresh-button"
              onClick={refresh}
              disabled={refreshing}
            >
              <span className={refreshing ? 'spin' : ''}>↻</span> Refresh
            </button>
          </div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow">{page.eyebrow}</div>
            <h1>
              {page.title}
              <span className="heading-period">.</span>
            </h1>
            <p className="page-subtitle">{page.subtitle}</p>
          </div>

          <div className="updated">
            <span className="updated-label">LAST UPDATED</span>
            <strong>
              {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Waiting for API'}
            </strong>
            <span className="updated-frequency">Auto-refresh · 3 seconds</span>
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <span>Connection error</span>
            {error}. Start the FastAPI backend on port 8000.
          </div>
        )}

        {/* OVERVIEW PAGE */}
        {activePage === 'overview' && (
          <>
            <SectionTitle
              eyebrow="LIVE TELEMETRY"
              title="Workload metrics"
              right={
                <span className="live-label">
                  <span className={`status-dot ${connection}`} />
                  {connection === 'connected' ? 'Updating live' : 'Waiting for API'}
                </span>
              }
            />

            <section className="metrics-grid">
              <MetricCard
                label="Requests submitted"
                value={formatNumber(submitted)}
                detail="Total accepted requests"
                accent="accent-purple"
              />
              <MetricCard
                label="Requests completed"
                value={formatNumber(completed)}
                detail="Successfully processed"
                accent="accent-green"
              />
              <MetricCard
                label="Requests failed"
                value={formatNumber(failed)}
                detail="Failed inference requests"
                accent="accent-red"
              />
              <MetricCard
                label="Queue length"
                value={formatNumber(queue)}
                detail={queue > 0 ? 'Requests waiting' : 'No queued requests'}
                accent="accent-blue"
              />
              <MetricCard
                label="Batches executed"
                value={formatNumber(batches)}
                detail="Scheduler batch count"
              />
              <MetricCard
                label="Success rate"
                value={successRate === null ? '—' : formatNumber(successRate, 1)}
                unit={successRate === null ? '' : '%'}
                detail="Completed / completed + failed"
              />
            </section>

            <section className="middle-grid">
              <article className="panel activity-panel">
                <SectionTitle
                  eyebrow="SCHEDULER ACTIVITY"
                  title="Workload trend"
                  right={
                    <span className="chart-legend">
                      <i /> Completed requests
                    </span>
                  }
                />

                {history.length > 1 ? (
                  <div className="chart-wrap">
                    <svg
                      viewBox="0 0 600 180"
                      preserveAspectRatio="none"
                      className="trend-chart"
                    >
                      {[25, 65, 105, 145].map((y) => (
                        <line
                          key={y}
                          x1="0"
                          y1={y}
                          x2="600"
                          y2={y}
                          className="chart-gridline"
                        />
                      ))}

                      <polyline
                        points={history.map((point, i) => {
                          const max = Math.max(
                            ...history.map((p) => p.completed),
                            1,
                          )
                          const x = (i / Math.max(history.length - 1, 1)) * 590 + 5
                          const y = 150 - (point.completed / max) * 125
                          return `${x},${y}`
                        }).join(' ')}
                        className="chart-line"
                      />
                    </svg>

                    <div className="chart-labels">
                      <span>{history[0]?.time}</span>
                      <span>{history[Math.floor(history.length / 2)]?.time}</span>
                      <span>{history[history.length - 1]?.time}</span>
                    </div>
                  </div>
                ) : (
                  <div className="chart-empty">
                    <div className="chart-empty-icon">⌁</div>
                    <strong>Collecting scheduler data</strong>
                    <span>Trend appears as live metric samples arrive.</span>
                  </div>
                )}

                <div className="chart-footer">
                  <div>
                    <span className="footer-dot purple" />
                    Completed total <strong>{formatNumber(completed)}</strong>
                  </div>
                  <div>
                    <span className="footer-dot blue" />
                    Current queue <strong>{formatNumber(queue)}</strong>
                  </div>
                </div>
              </article>

              <article className="panel latency-panel">
                <SectionTitle
                  eyebrow="LATENCY TELEMETRY"
                  title="Performance snapshot"
                />

                <div className="latency-main">
                  <div>
                    <div className="latency-label">Average total latency</div>
                    <div className="latency-value">
                      {formatNumber(latency, 2)}
                      <span> ms</span>
                    </div>
                  </div>
                  <div className="latency-glyph">⌁</div>
                </div>

                <div className="latency-row">
                  <span>Average queue time</span>
                  <strong>{formatNumber(queueTime, 2)} <small>ms</small></strong>
                </div>
                <div className="latency-row">
                  <span>Average inference time</span>
                  <strong>{formatNumber(inferenceTime, 2)} <small>ms</small></strong>
                </div>

                <div className="latency-note">
                  <span className="note-icon">i</span>
                  Values are derived from completed-request telemetry reported by the Copilot API.
                </div>
              </article>
            </section>

            <section className="panel copilot-panel">
              <div className="copilot-header">
                <div className="copilot-symbol">✳</div>
                <div className="copilot-heading">
                  <div className="eyebrow">GPUFLOW INTELLIGENCE</div>
                  <h2>AI Operations Copilot</h2>
                  <p>Operational insights grounded in observed scheduler metrics.</p>
                </div>
                <span className="ai-badge">AI INSIGHTS</span>
              </div>

              <div className="copilot-summary">
                <div className="insight-icon">✧</div>
                <div>
                  <div className="insight-title">System summary</div>
                  <p>{copilot?.summary ?? 'Waiting for Copilot data from the backend…'}</p>
                </div>
              </div>

              <div className="copilot-columns">
                <InsightList
                  title="Observations"
                  items={bottlenecks}
                  emptyText="No Copilot observations available yet."
                  type="warning"
                />
                <InsightList
                  title="Recommendations"
                  items={recommendations}
                  emptyText="Recommendations will appear when available."
                  type="recommendation"
                />
              </div>

              <div className="decision-footer">
                <span className="decision-icon">◇</span>
                <div>
                  <strong>Latest scheduler decision</strong>
                  <p>{copilot?.decision_explanation ?? 'No decision explanation available.'}</p>
                </div>
              </div>
            </section>
          </>
        )}

        {/* SCHEDULER METRICS PAGE */}
        {activePage === 'scheduler' && (
          <>
            <SectionTitle
              eyebrow="REQUEST PROCESSING"
              title="Request & queue metrics"
              right={
                <span className="live-label">
                  <span className={`status-dot ${connection}`} />
                  Backend telemetry
                </span>
              }
            />

            <section className="metrics-grid">
              <MetricCard
                label="Requests submitted"
                value={formatNumber(submitted)}
                detail="Total accepted requests"
                accent="accent-purple"
              />
              <MetricCard
                label="Requests completed"
                value={formatNumber(completed)}
                detail="Successfully processed"
                accent="accent-green"
              />
              <MetricCard
                label="Requests failed"
                value={formatNumber(failed)}
                detail="Failed inference requests"
                accent="accent-red"
              />
              <MetricCard
                label="Queue length"
                value={formatNumber(queue)}
                detail="Currently queued requests"
                accent="accent-blue"
              />
              <MetricCard
                label="Batches executed"
                value={formatNumber(batches)}
                detail="Total scheduler batches"
              />
              <MetricCard
                label="Success rate"
                value={successRate === null ? '—' : formatNumber(successRate, 1)}
                unit={successRate === null ? '' : '%'}
                detail="Completed / completed + failed"
              />
            </section>

            <section className="middle-grid">
              <article className="panel latency-panel">
                <SectionTitle
                  eyebrow="LATENCY BREAKDOWN"
                  title="Request performance"
                />

                <div className="latency-main">
                  <div>
                    <div className="latency-label">Average total latency</div>
                    <div className="latency-value">
                      {formatNumber(latency, 2)}<span> ms</span>
                    </div>
                  </div>
                  <div className="latency-glyph">⌁</div>
                </div>

                <div className="latency-row">
                  <span>Average queue time</span>
                  <strong>{formatNumber(queueTime, 2)} <small>ms</small></strong>
                </div>
                <div className="latency-row">
                  <span>Average inference time</span>
                  <strong>{formatNumber(inferenceTime, 2)} <small>ms</small></strong>
                </div>

                <div className="latency-note">
                  <span className="note-icon">i</span>
                  Latency values are based on completed-request telemetry.
                </div>
              </article>

              <article className="panel activity-panel">
                <SectionTitle
                  eyebrow="SCHEDULER ACTIVITY"
                  title="Completed workload trend"
                />

                {history.length > 1 ? (
                  <div className="chart-wrap">
                    <svg
                      viewBox="0 0 600 180"
                      preserveAspectRatio="none"
                      className="trend-chart"
                    >
                      {[25, 65, 105, 145].map((y) => (
                        <line
                          key={y}
                          x1="0"
                          y1={y}
                          x2="600"
                          y2={y}
                          className="chart-gridline"
                        />
                      ))}

                      <polyline
                        points={history.map((point, i) => {
                          const max = Math.max(
                            ...history.map((p) => p.completed),
                            1,
                          )
                          const x = (i / Math.max(history.length - 1, 1)) * 590 + 5
                          const y = 150 - (point.completed / max) * 125
                          return `${x},${y}`
                        }).join(' ')}
                        className="chart-line"
                      />
                    </svg>

                    <div className="chart-labels">
                      <span>{history[0]?.time}</span>
                      <span>{history[Math.floor(history.length / 2)]?.time}</span>
                      <span>{history[history.length - 1]?.time}</span>
                    </div>
                  </div>
                ) : (
                  <div className="chart-empty">
                    <div className="chart-empty-icon">⌁</div>
                    <strong>Collecting scheduler data</strong>
                    <span>Trend appears as live metric samples arrive.</span>
                  </div>
                )}

                <div className="chart-footer">
                  <div>
                    <span className="footer-dot purple" />
                    Completed total <strong>{formatNumber(completed)}</strong>
                  </div>
                  <div>
                    <span className="footer-dot blue" />
                    Current queue <strong>{formatNumber(queue)}</strong>
                  </div>
                </div>
              </article>
            </section>

            <section className="panel copilot-panel">
              <div className="copilot-header">
                <div className="copilot-symbol">⌁</div>
                <div className="copilot-heading">
                  <div className="eyebrow">SCHEDULER STATUS</div>
                  <h2>Current workload</h2>
                  <p>Live values reported by the GPUFlow-X backend.</p>
                </div>
              </div>

              <div className="copilot-columns">
                <div className="copilot-column">
                  <div className="column-heading">
                    <span className="column-dot recommendation" />
                    Processing status
                  </div>
                  <div className="insight-item">
                    <span className="item-marker">›</span>
                    <span>{formatNumber(completed)} requests completed successfully.</span>
                  </div>
                  <div className="insight-item">
                    <span className="item-marker">›</span>
                    <span>{formatNumber(failed)} requests failed.</span>
                  </div>
                  <div className="insight-item">
                    <span className="item-marker">›</span>
                    <span>{formatNumber(queue)} requests currently queued.</span>
                  </div>
                </div>

                <div className="copilot-column">
                  <div className="column-heading">
                    <span className="column-dot warning" />
                    Telemetry note
                  </div>
                  <div className="muted-copy">
                    Metrics reflect backend-reported values. No physical GPU telemetry is assumed.
                  </div>
                </div>
              </div>
            </section>
          </>
        )}

        {/* AI COPILOT PAGE */}
        {activePage === 'copilot' && (
          <section className="panel copilot-panel">
            <div className="copilot-header">
              <div className="copilot-symbol">✳</div>
              <div className="copilot-heading">
                <div className="eyebrow">GPUFLOW INTELLIGENCE</div>
                <h2>AI Operations Copilot</h2>
                <p>Operational insights grounded in observed scheduler metrics.</p>
              </div>
              <span className="ai-badge">AI INSIGHTS</span>
            </div>

            <div className="copilot-summary">
              <div className="insight-icon">✧</div>
              <div>
                <div className="insight-title">System summary</div>
                <p>{copilot?.summary ?? 'Waiting for Copilot data from the backend…'}</p>
              </div>
            </div>

            <div className="copilot-columns">
              <InsightList
                title="Observations"
                items={bottlenecks}
                emptyText="No Copilot observations available yet."
                type="warning"
              />
              <InsightList
                title="Recommendations"
                items={recommendations}
                emptyText="Recommendations will appear when available."
                type="recommendation"
              />
            </div>

            <div className="decision-footer">
              <span className="decision-icon">◇</span>
              <div>
                <strong>Latest scheduler decision</strong>
                <p>
                  {copilot?.decision_explanation ??
                    'No decision explanation available.'}
                </p>
              </div>
            </div>

            <div className="copilot-summary">
              <div className="insight-icon">⌁</div>
              <div>
                <div className="insight-title">Observed metrics</div>
                <p>
                  Submitted: {formatNumber(submitted)} · Completed: {formatNumber(completed)} ·
                  Failed: {formatNumber(failed)} · Queue: {formatNumber(queue)}
                </p>
                <p>
                  Average latency: {formatNumber(latency, 2)} ms ·
                  Average queue time: {formatNumber(queueTime, 2)} ms ·
                  Average inference time: {formatNumber(inferenceTime, 2)} ms
                </p>
              </div>
            </div>
          </section>
        )}

        <footer className="page-footer">
          <span>GPUFlow-X <b>·</b> Adaptive GPU Inference Scheduler</span>
          <span>Metrics reflect backend-reported values. No physical GPU telemetry is assumed.</span>
        </footer>
      </main>
    </div>
  )
}

export default App
