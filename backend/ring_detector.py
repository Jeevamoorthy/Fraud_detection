import community.community_louvain as community_louvain
import networkx as nx

class RingDetector:
    def __init__(self):
        pass

    def detect_rings(self, nx_graph, high_risk_accounts):
        """
        Uses Louvain community detection on a homogenous projection of the graph.
        Identifies communities with multiple high-risk accounts.
        """
        # Create a simplified undirected account-to-account (and shared IP/Device) graph
        # For Louvain, we need an undirected graph
        undirected_g = nx_graph.to_undirected()
        
        try:
            partition = community_louvain.best_partition(undirected_g, random_state=42)
        except Exception as e:
            # If graph is empty or not suitable
            return []

        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
            
        detected_rings = []
        for comm_id, nodes in communities.items():
            # Count high risk accounts in this community
            risk_count = sum(1 for n in nodes if n in high_risk_accounts)
            
            # Sub-graph analysis: Check for shared device/IP in the community
            devices = [n for n in nodes if n.startswith("DEV_")]
            ips = [n for n in nodes if n.startswith("IP_")]
            
            # Rule: 3+ high risk accounts OR (2+ high risk + shared device/IP)
            if risk_count >= 3 or (risk_count >= 2 and (len(devices) > 0 or len(ips) > 0)):
                detected_rings.append({
                    "ring_id": f"RING_{comm_id}",
                    "size": len(nodes),
                    "high_risk_count": risk_count,
                    "shared_devices": len(devices),
                    "shared_ips": len(ips),
                    "members": nodes
                })
                
        return detected_rings
