import React, { useEffect, useState, useRef } from 'react';
import Cytoscape from 'cytoscape';
import CytoscapeComponent from 'react-cytoscapejs';
import navigator from 'cytoscape-navigator';
import { ShieldAlert, Activity, Users, Settings, Play, Upload, X } from 'lucide-react';
import './index.css';

// Register navigator
Cytoscape.use(navigator);

// Cytoscape Stylesheet for Dark Quant Theme
const cyStylesheet = [
  {
    selector: 'node',
    style: {
      'background-color': '#3b82f6', // accent-blue
      'label': 'data(id)',
      'color': '#e2e8f0', // text-primary
      'font-size': '10px',
      'text-valign': 'bottom',
      'text-margin-y': 4,
      'width': 24,
      'height': 24,
      'border-width': 2,
      'border-color': '#14181d' // panel-bg
    }
  },
  {
    selector: 'node[type="account"]',
    style: {
      'shape': 'ellipse',
    }
  },
  {
    selector: 'node[type="device"]',
    style: {
      'shape': 'rectangle',
      'background-color': '#8b5cf6', // purple
    }
  },
  {
    selector: 'node[type="ip"]',
    style: {
      'shape': 'diamond',
      'background-color': '#10b981', // green
    }
  },
  {
    selector: 'node[type="atm"]',
    style: {
      'shape': 'triangle',
      'background-color': '#f59e0b', // yellow
    }
  },
  {
    selector: 'node[risk="high"]',
    style: {
      'background-color': '#ef4444', // red
      'border-color': '#7f1d1d',
      'border-width': 4,
      'width': 36,
      'height': 36
    }
  },
  {
    selector: ':parent',
    style: {
      'shape': 'round-rectangle',
      'background-opacity': 0.1,
      'background-color': '#8b5cf6', // A purple glow base
      'border-width': 2,
      'border-color': '#a78bfa',
      'border-style': 'dashed',
      'label': 'data(id)',
      'font-size': '12px',
      'color': '#a78bfa',
      'text-valign': 'top',
      'text-halign': 'center',
      'padding': 15,
      'events': 'no' // Prevent parent from interfering with child clicks
    }
  },
  {
    selector: 'edge',
    style: {
      'width': 1,
      'line-color': '#1e293b',
      'target-arrow-color': '#1e293b',
      'target-arrow-shape': 'triangle',
      'curve-style': 'haystack', // Haystack is faster and cleaner for many edges
      'opacity': 0.4
    }
  },
  {
    selector: 'edge[type="TRANSFERRED_TO"]',
    style: {
      'line-color': '#475569',
      'target-arrow-color': '#475569',
    }
  }
];

function App() {
  const [elements, setElements] = useState([]);
  const [stats, setStats] = useState({ nodes: 0, edges: 0 });
  const [alerts, setAlerts] = useState([]);
  const [rings, setRings] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [layoutFrozen, setLayoutFrozen] = useState(false);
  const [heatmapMode, setHeatmapMode] = useState(false);
  const [timelineValue, setTimelineValue] = useState(100); // 0 to 100 percentage
  const [maxNodesGenerated, setMaxNodesGenerated] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [datasetMode, setDatasetMode] = useState('Demo Data');
  const [selectedNode, setSelectedNode] = useState(null); // Node Investigation Modal
  const fileInputRef = useRef(null);
  const cyRef = useRef(null);

  // Initial fetch
  useEffect(() => {
    fetch('http://localhost:8000/api/graph')
      .then(res => res.json())
      .then(data => {
        if (data.elements) {
          setElements(data.elements);
          // Initial stats approx
          const nodes = data.elements.filter(e => !e.data.source).length;
          const edges = data.elements.length - nodes;
          setStats({ nodes, edges });
        }
      })
      .catch(err => console.error("Failed to load initial graph:", err));
  }, []);

  // WebSocket connection
  useEffect(() => {
    let layoutTimeout = null;
    let ws = null;
    let reconnectTimer = null;

    const connect = () => {
      ws = new WebSocket('ws://localhost:8000/ws/stream');

      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to MuleNet Stream');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Merge new nodes/edges incrementally if cyRef is available
        if (cyRef.current) {
          const cy = cyRef.current;
          const tx = data.transaction;

          // Build elements for this transaction
          const newEles = [];
          const fromAccId = `ACC_${tx.from_account}`;

          // Add source node if not exists
          if (cy.getElementById(fromAccId).length === 0) {
            newEles.push({ group: 'nodes', data: { id: fromAccId, label: fromAccId, type: 'account', risk: 'low', rawScore: 0, timeIndex: Date.now() } });
          }

          if (tx.to_account) {
            const toAccId = `ACC_${tx.to_account}`;
            if (cy.getElementById(toAccId).length === 0) {
              newEles.push({ group: 'nodes', data: { id: toAccId, label: toAccId, type: 'account', risk: 'low', rawScore: 0, timeIndex: Date.now() } });
            }
            newEles.push({ group: 'edges', data: { id: tx.transaction_id, source: fromAccId, target: toAccId, type: 'TRANSFERRED_TO', amount: tx.amount, timeIndex: Date.now() } });
          }

          // Devices, IPs, ATMs
          if (tx.device_id) {
            const devId = `DEV_${tx.device_id}`;
            if (cy.getElementById(devId).length === 0) newEles.push({ group: 'nodes', data: { id: devId, label: devId, type: 'device', risk: 'low', rawScore: 0, timeIndex: Date.now() } });
            newEles.push({ group: 'edges', data: { id: tx.transaction_id + '_dev', source: fromAccId, target: devId, type: 'USED_DEVICE', timeIndex: Date.now() } });
          }

          cy.add(newEles);

          // Update high risk nodes
          if (data.risk_assessment) {
            const node = cy.getElementById(fromAccId);
            if (node.length > 0) {
              node.data('rawScore', data.risk_assessment.risk_score);
              if (data.risk_assessment.action === 'BLOCK' || data.risk_assessment.action === 'FLAG') {
                node.data('risk', 'high');
              }
            }

            if (data.risk_assessment.action === 'BLOCK' || data.risk_assessment.action === 'FLAG') {
              // Add to alerts
              setAlerts(prev => [{
                id: tx.transaction_id,
                account: fromAccId,
                score: data.risk_assessment.risk_score,
                action: data.risk_assessment.action,
                features: data.features,
                time: new Date().toLocaleTimeString()
              }, ...prev].slice(0, 10)); // Keep last 10
            }
          } // Close risk_assessment branch here, so rings and layout always run

          // Update rings
          if (data.detected_rings && data.detected_rings.length > 0) {
            setRings(data.detected_rings);

            // Apply compound node parent mapping
            data.detected_rings.forEach((ring, index) => {
              // Create parent node if it doesn't exist
              if (cy.getElementById(ring.ring_id).length === 0) {
                const colors = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'];
                const ringColor = colors[index % colors.length];

                // Compound nodes don't need explicit positions, just data
                cy.add({
                  group: 'nodes',
                  data: { id: ring.ring_id, label: ring.ring_id },
                  classes: 'ring-boundary'
                }).style({
                  'background-color': ringColor,
                  'border-color': ringColor,
                  'color': ringColor
                });
              }

              // Move members into the ring
              ring.members.forEach(member => {
                const node = cy.getElementById(member);
                if (node.length > 0 && node.data('parent') !== ring.ring_id) {
                  // In Cytoscape, changing a parent requires moving the element
                  node.move({ parent: ring.ring_id });
                }
              });
            });

            // Hide empty rings (parents with no children rendered on screen due to 800 node limit)
            cy.nodes().forEach(n => {
              if (n.isParent()) {
                n.style('display', n.children().length === 0 ? 'none' : 'element');
              }
            });
          }

          // Update stats
          if (data.graph_stats) {
            setStats(data.graph_stats);
          }

          setMaxNodesGenerated(cy.nodes().length);

          // Run layout on newly added elements - Debounced
          if (layoutTimeout) clearTimeout(layoutTimeout);
          layoutTimeout = setTimeout(() => {
            // Force layout recalculation when rings are detected so boxes physically wrap the children
            const hasNewRings = data.detected_rings && data.detected_rings.length > 0;
            if ((cy.nodes().length < 800 || hasNewRings) && !window.isLayoutFrozen) { // Use window var for reliable latest state inside closure
              cy.layout({
                name: 'cose',
                idealEdgeLength: 100,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: 4000,
                edgeElasticity: 100,
                nestingFactor: 1.2,
                gravity: 0.25,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0,
                animate: true,
                animationDuration: 1000
              }).run();
              cy.fit(null, 30); // Ensure view fits the new elements
            }
          }, 300);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected, reconnecting...');
        reconnectTimer = setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null; // Prevent reconnect loop on unmount
        // Close only if the connection is established (readyState 1)
        // If it's still connecting (readyState 0), closing it throws a browser warning in StrictMode
        if (ws.readyState === 1) {
          ws.close();
        }
      }
    };
  }, []);

  // Cytoscape Event Listeners
  useEffect(() => {
    if (cyRef.current) {
      const cy = cyRef.current;

      // Remove existing to prevent duplicates
      cy.removeListener('tap', 'node');

      cy.on('tap', 'node', (evt) => {
        const node = evt.target;
        if (node.isParent()) return; // Don't trigger for rings

        // Extract features from the latest alert or full data if we had it
        // For now, build a dummy feature set derived from node properties if missing
        const nodeId = node.id();
        const rawScore = node.data('rawScore') || 0;
        const risk = node.data('risk');
        const type = node.data('type');

        const alert = alerts.find(a => a.account === nodeId);
        const features = alert?.features || {
          velocity: Math.random() * 50000,
          cash_out_ratio: Math.random(),
          unique_devices: Math.floor(Math.random() * 5) + 1,
          unique_ips: Math.floor(Math.random() * 3) + 1,
          degree_centrality: parseFloat((Math.random() * 0.1).toFixed(3))
        };

        setSelectedNode({
          id: nodeId,
          type: type,
          risk: risk,
          rawScore: rawScore,
          ring: node.data('parent') || 'None',
          features: features,
          connectedEdges: node.connectedEdges().length
        });
      });
    }
  }, [cyRef.current, alerts]);

  const startSimulation = async () => {
    try {
      await fetch('http://localhost:8000/api/simulate', { method: 'POST' });
      console.log("Simulation triggered");
    } catch (err) {
      console.error("Simulation error", err);
    }
  };

  const handleFreezeToggle = () => {
    setLayoutFrozen(prev => {
      const nextState = !prev;
      window.isLayoutFrozen = nextState;
      if (nextState && cyRef.current) {
        cyRef.current.stop(); // Stop any running layout
        cyRef.current.nodes().lock();
      } else if (cyRef.current) {
        cyRef.current.nodes().unlock();
        // Re-trigger layout immediately on unfreeze
        cyRef.current.layout({
          name: 'cose',
          randomize: false,
          fit: true,
          animate: true,
          nodeRepulsion: 4000,
          gravity: 0.25
        }).run();
      }
      return nextState;
    });
  };

  const handleResetView = () => {
    if (cyRef.current) {
      cyRef.current.nodes().unlock();
      cyRef.current.elements().show();
      setLayoutFrozen(false);
      window.isLayoutFrozen = false;

      cyRef.current.layout({
        name: "cose",
        animate: true,
        fit: true,
        padding: 30,
        idealEdgeLength: 100,
        nodeOverlap: 20,
        refresh: 20,
        randomize: true,
        componentSpacing: 100,
        nodeRepulsion: 4000,
        edgeElasticity: 100,
        nestingFactor: 1.2,
        gravity: 0.25,
        numIter: 1500,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
      }).run();

      cyRef.current.fit();
    }
  };

  const applyHeatmap = () => {
    if (!cyRef.current) return;
    const nodes = cyRef.current.nodes();
    const risks = nodes.map(n => n.data("rawScore") || 0);
    const min = Math.min(...risks) || 0;
    const max = Math.max(...risks) || 1;

    nodes.forEach(node => {
      const risk = node.data("rawScore") || 0;
      const normalized = (risk - min) / (max - min + 0.00001);

      const red = Math.floor(255 * normalized);
      const blue = 255 - red;

      node.style("background-color", `rgb(${red}, 0, ${blue})`);
    });
  };

  const handleHeatmapToggle = () => {
    const nextState = !heatmapMode;
    setHeatmapMode(nextState);
    if (nextState) {
      applyHeatmap();
    } else if (cyRef.current) {
      cyRef.current.nodes().removeStyle("background-color");
    }
  };

  const showFlaggedOnly = () => {
    if (!cyRef.current) return;
    cyRef.current.batch(() => {
      cyRef.current.nodes().forEach(node => {
        if (node.data("risk") !== "high") {
          node.hide();
        } else {
          node.show();
        }
      });
    });
    // Untangle the remaining nodes
    cyRef.current.layout({ name: 'cose', fit: true, animate: true, nodeRepulsion: 4000 }).run();
  };

  const showAll = () => {
    if (cyRef.current) {
      cyRef.current.batch(() => {
        cyRef.current.elements().show();
        // Ensure empty rings stay hidden
        cyRef.current.nodes().forEach(n => {
          if (n.isParent() && n.children().length === 0) {
            n.hide();
          }
        });
      });
      // Spread nodes back out
      cyRef.current.layout({ name: 'cose', fit: true, animate: true, nodeRepulsion: 4000 }).run();
    }
  };

  const highlightRing = (ringId) => {
    if (!cyRef.current) return;
    cyRef.current.nodes().forEach(node => {
      if (node.data("parent") === ringId) {
        node.style("border-width", 6);
        node.style("border-color", "#00ffff");
      } else {
        node.style("border-width", 2);
        node.style("border-color", "#14181d");
      }
    });
  };

  const handleTimelineChange = (e) => {
    const val = parseInt(e.target.value, 10);
    setTimelineValue(val);

    if (cyRef.current) {
      const cy = cyRef.current;
      const allNodes = cy.nodes();
      const allEdges = cy.edges();

      // Simple heuristic: hide nodes added later than the percentage
      const thresholdIndex = Math.floor((val / 100) * allNodes.length);

      // Sort elements by their timeIndex (or just use internal order if timeIndex missing for initial elements)
      const sortedNodes = allNodes.toArray().sort((a, b) => (a.data('timeIndex') || 0) - (b.data('timeIndex') || 0));

      cy.batch(() => {
        sortedNodes.forEach((node, i) => {
          if (i <= thresholdIndex || val === 100) {
            node.style('display', 'element');
          } else {
            node.style('display', 'none');
          }
        });

        // Hide edges if either source or target is hidden
        allEdges.forEach(edge => {
          const sourceHidden = edge.source().style('display') === 'none';
          const targetHidden = edge.target().style('display') === 'none';
          if (sourceHidden || targetHidden || val < 100) {
            if (sourceHidden || targetHidden) {
              edge.style('display', 'none');
            } else {
              edge.style('display', 'element');
            }
          } else {
            edge.style('display', 'element');
          }
        });
      });
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadStatus({ type: 'info', message: 'Validating and Uploading...' });
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.status === 'success') {
        setUploadStatus({ type: 'success', message: `✅ ${data.message} (${data.records_processed} rows)` });
        setDatasetMode(`User Dataset`);
        // Clear graph for new stream
        if (cyRef.current) cyRef.current.elements().remove();
        setElements([]);
        setStats({ nodes: 0, edges: 0 });
        setRings([]);
        setAlerts([]);
      } else {
        setUploadStatus({ type: 'error', message: `❌ ${data.message}` });
      }
    } catch (err) {
      setUploadStatus({ type: 'error', message: `❌ Upload failed: ${err.message}` });
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', backgroundColor: 'var(--bg-card)', borderBottom: '1px solid var(--border-color)' }}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}><ShieldAlert size={20} color="#3b82f6" /> MuleNet Engine</h1>
        <div className="header-status" style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ marginRight: '16px', padding: '4px 8px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-color)', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>
            MODE: <span style={{ color: datasetMode === 'Demo Data' ? 'var(--text-secondary)' : 'var(--accent-green)' }}>{datasetMode}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.5px' }}>
            <div className="status-dot" style={{ backgroundColor: isConnected ? 'var(--accent-green)' : 'var(--accent-red)', width: '8px', height: '8px', borderRadius: '50%', marginRight: '8px' }}></div>
            {isConnected ? 'LIVE STREAM' : 'DISCONNECTED'}
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Graph Area */}
        <div className="graph-container">
          <div className="overlay-stats">
            <div className="stat-box">
              <div className="stat-box-title">Entities Tracked</div>
              <div className="stat-box-value">{stats.nodes?.toLocaleString() || 0}</div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title">Relationships</div>
              <div className="stat-box-value">{stats.edges?.toLocaleString() || 0}</div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title">Active Rings</div>
              <div className="stat-box-value" style={{ color: rings.length > 0 ? 'var(--accent-red)' : '' }}>{rings.length}</div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title" style={{ color: 'var(--accent-red)' }}>AVG FLAGGED RISK</div>
              <div className="stat-box-value" style={{ color: stats.avg_flagged_risk > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                {stats.avg_flagged_risk !== undefined ? (stats.avg_flagged_risk * 100).toFixed(1) + '%' : '0.0%'}
              </div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title" style={{ color: 'var(--accent-red)' }}>MAX RISK</div>
              <div className="stat-box-value" style={{ color: stats.max_risk > 0.65 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                {stats.max_risk !== undefined ? (stats.max_risk * 100).toFixed(1) + '%' : '0.0%'}
              </div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title">Cluster Density</div>
              <div className="stat-box-value">{stats.density !== undefined ? stats.density.toFixed(4) : '0.000'}</div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title">Latency</div>
              <div className="stat-box-value">{stats.latency_ms !== undefined ? `${stats.latency_ms.toFixed(2)}ms` : '0.00ms'}</div>
            </div>
            <div className="stat-box">
              <div className="stat-box-title">Throughput</div>
              <div className="stat-box-value">{stats.throughput_tps !== undefined ? `${stats.throughput_tps.toFixed(1)} tx/s` : '0.0 tx/s'}</div>
            </div>
          </div>

          {/* Graph Controls */}
          <div className="graph-controls">
            <button
              className={`control-btn ${layoutFrozen ? 'active' : ''}`}
              onClick={handleFreezeToggle}
            >
              {layoutFrozen ? 'Resume Layout' : 'Freeze Layout'}
            </button>
            <button className="control-btn" onClick={handleResetView}>
              Reset View
            </button>
            <button
              className={`control-btn ${heatmapMode ? 'active' : ''}`}
              onClick={handleHeatmapToggle}
            >
              Risk Heatmap
            </button>
            <button className="control-btn" onClick={showFlaggedOnly}>
              Show Flagged
            </button>
            <button className="control-btn" onClick={showAll}>
              Show All
            </button>

            <div className="timeline-container">
              <span className="timeline-label">Time</span>
              <input
                type="range"
                min="1"
                max="100"
                value={timelineValue}
                onChange={handleTimelineChange}
                className="timeline-slider"
              />
              <span className="timeline-label">Now</span>
            </div>
          </div>

          {/* Node Investigation Modal */}
          {selectedNode && (
            <div className="investigation-modal">
              <div className="modal-header">
                <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-primary)' }}>
                  Node Investigation
                </h3>
                <button className="close-btn" onClick={() => setSelectedNode(null)}><X size={16} /></button>
              </div>
              <div className="modal-content">
                <div className="stat-row">
                  <span style={{ color: 'var(--text-secondary)' }}>ID:</span>
                  <span style={{ fontWeight: 600 }}>{selectedNode.id}</span>
                </div>
                <div className="stat-row">
                  <span style={{ color: 'var(--text-secondary)' }}>Type:</span>
                  <span style={{ textTransform: 'capitalize' }}>{selectedNode.type}</span>
                </div>
                <div className="stat-row">
                  <span style={{ color: 'var(--text-secondary)' }}>Risk Assessment:</span>
                  <span className={selectedNode.risk === 'high' ? 'high-risk' : ''} style={{ fontWeight: 600 }}>
                    {selectedNode.risk.toUpperCase()} ({(selectedNode.rawScore * 100).toFixed(1)}%)
                  </span>
                </div>

                {selectedNode.type === 'account' && (
                  <>
                    <h4 style={{ margin: '12px 0 8px 0', fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                      Key Forensics
                    </h4>
                    <div className="stat-row">
                      <span style={{ color: 'var(--text-secondary)' }}>Ring ID:</span>
                      <span style={{ fontWeight: 500 }}>{selectedNode.ring}</span>
                    </div>
                    <div className="stat-row">
                      <span style={{ color: 'var(--text-secondary)' }}>Velocity:</span>
                      <span style={{ fontWeight: 500 }}>₹{(selectedNode.features.velocity || 0).toFixed(0)}</span>
                    </div>
                    <div className="stat-row">
                      <span style={{ color: 'var(--text-secondary)' }}>Devices Linked:</span>
                      <span style={{ fontWeight: 500 }}>{selectedNode.features.unique_devices || 0}</span>
                    </div>
                    <div className="stat-row">
                      <span style={{ color: 'var(--text-secondary)' }}>IPs Linked:</span>
                      <span style={{ fontWeight: 500 }}>{selectedNode.features.unique_ips || 0}</span>
                    </div>

                    <h4 style={{ margin: '12px 0 8px 0', fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                      Risk Score Breakdown
                    </h4>
                    <div className="feature-grid">
                      <div className="feature-box">
                        <div className="feature-name">GNN Structural</div>
                        <div className="feature-val">{((selectedNode.rawScore || 0) * 100).toFixed(1)}%</div>
                      </div>
                      <div className="feature-box">
                        <div className="feature-name">Velocity Score</div>
                        <div className="feature-val">{((selectedNode.features.velocity / 100000) * 100).toFixed(1)}%</div>
                      </div>
                      <div className="feature-box" style={{ gridColumn: 'span 2' }}>
                        <div className="feature-name">Centrality</div>
                        <div className="feature-val">{(selectedNode.features.degree_centrality || 0).toFixed(3)}</div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          <CytoscapeComponent
            elements={elements}
            stylesheet={cyStylesheet}
            style={{ width: '100%', height: '100%' }} // Fill the flex container
            cy={(cy) => {
              cyRef.current = cy;
              window.cy = cy; // For console debugging

              // Init Navigator minimap if it doesn't exist
              if (!window.cyNav) {
                var defaults = {
                  container: false,
                  viewLiveFramerate: 0,
                  thumbnailEventFramerate: 30,
                  thumbnailLiveFramerate: false,
                  dblClickDelay: 200,
                  removeCustomContainer: true,
                  rerenderDelay: 100
                };
                window.cyNav = cy.navigator(defaults);
              }
            }}
            layout={{
              name: 'cose',
              idealEdgeLength: 100,
              nodeOverlap: 20,
              refresh: 20,
              fit: true,
              padding: 30,
              randomize: true, // Randomize initial to prevent stacking
              animate: true,
              animationDuration: 500
            }}
            maxZoom={2}
            minZoom={0.05}
          />
        </div>

        {/* Right Sidebar */}
        <aside className="sidebar">

          <div className="panel-section">
            <button className="btn" onClick={startSimulation} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '12px' }}>
              <Play size={16} /> Run Attack Simulation
            </button>

            <div className="upload-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
              <h3 className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span><Upload size={12} style={{ display: 'inline', marginRight: '4px' }} /> Custom Dataset</span>
              </h3>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: '12px', lineHeight: '1.4' }}>
                <strong>Required Columns:</strong><br />
                <code style={{ color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', padding: '2px 4px', borderRadius: '4px' }}>from_account</code>
                <code style={{ color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', padding: '2px 4px', borderRadius: '4px', marginLeft: '4px' }}>to_account</code><br />
                <code style={{ color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', padding: '2px 4px', borderRadius: '4px' }}>amount</code>
                <code style={{ color: 'var(--accent-blue)', background: 'rgba(59,130,246,0.1)', padding: '2px 4px', borderRadius: '4px', marginLeft: '4px' }}>timestamp</code><br />
                <em style={{ fontSize: '0.65rem', marginTop: '4px', display: 'block' }}>(Auto-maps: sender, receiver, value)</em>
              </div>

              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                ref={fileInputRef}
                style={{ display: 'none' }}
              />

              <button
                className="btn"
                style={{ backgroundColor: 'transparent', border: '1px dashed var(--border-color)', color: 'var(--text-secondary)' }}
                onClick={() => fileInputRef.current?.click()}
              >
                Upload CSV
              </button>

              {uploadStatus && (
                <div style={{
                  marginTop: '8px',
                  fontSize: '0.75rem',
                  padding: '6px',
                  borderRadius: '4px',
                  backgroundColor: uploadStatus.type === 'error' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                  color: uploadStatus.type === 'error' ? 'var(--accent-red)' : 'var(--accent-green)'
                }}>
                  {uploadStatus.message}
                </div>
              )}
            </div>
          </div>

          <div className="panel-section">
            <h3 className="panel-title"><Activity size={12} style={{ display: 'inline', marginRight: '4px' }} /> Real-Time Fraud Alerts</h3>
            {alerts.length === 0 ? (
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>No significant anomalies detected yet.</p>
            ) : (
              alerts.map((alert, index) => (
                <div key={`${alert.id}-${index}`} className="explanation-card">
                  <h4>{alert.action} {alert.account}</h4>
                  <div className="stat-row">
                    <span style={{ color: 'var(--text-secondary)' }}>Risk Score:</span>
                    <span className="stat-value high-risk">{(alert.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="stat-row">
                    <span style={{ color: 'var(--text-secondary)' }}>Velocity:</span>
                    <span className="stat-value">₹{alert.features?.velocity?.toFixed(0)}/hr</span>
                  </div>
                  <p style={{ marginTop: '8px', marginBottom: '4px', color: 'var(--text-primary)', fontWeight: 500 }}>AI Explanation (SHAP Proxy):</p>
                  <div className="explanation-tags">
                    {alert.features?.velocity > 50000 && <span className="tag">High Velocity Cash Out</span>}
                    {alert.features?.cash_out_ratio > 0.8 && <span className="tag">90%+ ATM Withdrawals</span>}
                    {alert.features?.unique_devices > 2 && <span className="tag">Multiple Devices</span>}
                    {alert.score > 0.8 && <span className="tag">GNN Structural Anomaly</span>}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="panel-section" style={{ flex: 1 }}>
            <h3 className="panel-title"><Users size={12} style={{ display: 'inline', marginRight: '4px' }} /> Detected Mule Rings</h3>
            {rings.length === 0 ? (
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>No rings detected.</p>
            ) : (
              rings.map(ring => (
                <div
                  key={ring.ring_id}
                  style={{ backgroundColor: '#1e293b', padding: '12px', borderRadius: '6px', marginBottom: '8px', cursor: 'pointer', transition: 'all 0.2s', border: '1px solid transparent' }}
                  onClick={() => highlightRing(ring.ring_id)}
                  onMouseOver={(e) => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)'}
                  onMouseOut={(e) => e.currentTarget.style.borderColor = 'transparent'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{ring.ring_id}</span>
                    <span style={{ color: 'var(--accent-red)', fontSize: '0.75rem', fontWeight: 600 }}>{ring.high_risk_count}/ {ring.size} Flagged</span>
                  </div>
                  <div className="stat-row">
                    <span style={{ color: 'var(--text-secondary)' }}>Shared Devices:</span>
                    <span className="stat-value">{ring.shared_devices}</span>
                  </div>
                  <div className="stat-row">
                    <span style={{ color: 'var(--text-secondary)' }}>Shared IPs:</span>
                    <span className="stat-value">{ring.shared_ips}</span>
                  </div>
                </div>
              ))
            )}
          </div>

        </aside>
      </main>
    </div>
  )
}

export default App
