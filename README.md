# MuleNet: Real-Time Cross-Channel Graph Neural Network for Mule Ring Detection

MuleNet is a bank-grade intelligence engine designed to detect multi-channel mule accounts and fraud rings in real-time. It constructs a heterogeneous entity graph using logs from Apps, Web, ATMs, and UPI, calculates node features securely, runs Graph Neural Network (GNN) inference, and displays anomalous activity on a React-based Quant Terminal dashboard powered by Cytoscape.js.

## Features
- **Synthetic Data Engine:** Simulates 4 primary mule ring patterns.
- **Heterogeneous Graph:** Maps Accounts, Devices, IPs, and ATMs.
- **Graph Neural Network:** Uses PyTorch Geometric (GraphSAGE/GCN) to predict fraud probability by analyzing node network structures.
- **Risk Engine:** Combines AI (GNN) with deterministic heuristics (velocity rules) for an Explainable AI (XAI) output.
- **Community Detection:** Applies the Louvain algorithm to discover coordinated multi-actor fraud rings.
- **Real-Time Streaming:** FastAPI WebSockets stream live transactions to the UI.
- **Premium Quant UI:** Cytoscape.js network visualization with institutional dark aesthetic.

## Tech Stack
- **Backend:** Python, FastAPI, PyTorch Geometric, NetworkX, python-louvain, Pandas.
- **Frontend:** React, Vite, Cytoscape.js, Lucide Icons, Vanilla CSS (Dark Mode).
- **Deployment:** Docker, Docker Compose.

## Running Locally (Docker)

1. Ensure Docker Desktop is installed.
2. Build and start the containers from the root directory:
```bash
docker-compose up --build
```
3. The frontend is accessible at: `http://localhost:4173`
4. The backend is accessible at: `http://localhost:8000`

## Running Locally (Manual)

### Backend
1. `cd backend`
2. `python -m venv venv`
3. `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. `pip install -r requirements.txt`
5. `python main.py`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

Launch the app and click **Run Attack Simulation** to watch the real-time GNN at work!
