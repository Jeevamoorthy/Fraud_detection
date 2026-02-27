import networkx as nx
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def extract_features(self, nx_graph, account_id=None):
        """
        Extracts features for all account nodes or a specific account node.
        """
        features = {}
        target_nodes = [account_id] if account_id else [n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'account']
        
        in_degrees = dict(nx_graph.in_degree())
        out_degrees = dict(nx_graph.out_degree())
        
        # Calculate centrality on the homogenous projection for simplicity, or just use directed degrees
        try:
            # Degree centrality is quick
            centrality = nx.degree_centrality(nx_graph)
        except:
            centrality = {n: 0 for n in target_nodes}

        for node in target_nodes:
            if node not in nx_graph:
                continue
                
            in_deg = in_degrees.get(node, 0)
            out_deg = out_degrees.get(node, 0)
            
            # temporal/velocity features
            amounts_in = []
            amounts_out = []
            timestamps = []
            withdrawals = 0
            devices = set()
            ips = set()
            
            for u, v, data in nx_graph.in_edges(node, data=True):
                if data.get('type') == 'TRANSFERRED_TO':
                    amounts_in.append(data.get('amount', 0))
                    if isinstance(data.get('timestamp'), str):
                        timestamps.append(pd.to_datetime(data.get('timestamp')))
                    else:
                        timestamps.append(data.get('timestamp'))
                        
            for u, v, data in nx_graph.out_edges(node, data=True):
                if data.get('type') == 'TRANSFERRED_TO':
                    amounts_out.append(data.get('amount', 0))
                    if isinstance(data.get('timestamp'), str):
                        timestamps.append(pd.to_datetime(data.get('timestamp')))
                    else:
                        timestamps.append(data.get('timestamp'))
                elif data.get('type') == 'WITHDREW_AT':
                    withdrawals += data.get('amount', 0)
                elif data.get('type') == 'USED_DEVICE':
                    devices.add(v)
                elif data.get('type') == 'LOGGED_IN_FROM':
                    ips.add(v)

            total_amount_in = sum(amounts_in)
            total_amount_out = sum(amounts_out)
            
            # Velocity: amount / time_diff (simplified)
            velocity = 0
            timestamps = sorted([ts for ts in timestamps if pd.notna(ts)])
            if len(timestamps) > 1:
                time_diff = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0 # in hours
                if time_diff > 0.01: # Prevent micro-second bursts creating infinite velocity
                    velocity = (total_amount_in + total_amount_out) / time_diff
                else:
                    velocity = total_amount_in + total_amount_out # instantaneous burst
                    
            cash_out_ratio = (withdrawals / total_amount_in) if total_amount_in > 0 else 0
            
            features[node] = {
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_in": total_amount_in,
                "total_out": total_amount_out,
                "velocity": velocity,
                "cash_out_ratio": cash_out_ratio,
                "unique_devices": len(devices),
                "unique_ips": len(ips),
                "centrality": centrality.get(node, 0)
            }
            
        if account_id:
            return features.get(account_id, {})
        return pd.DataFrame.from_dict(features, orient='index')

if __name__ == "__main__":
    from graph_builder import GraphBuilder
    import os
    if os.path.exists("../dataset/synthetic_transactions.csv"):
        df = pd.read_csv("../dataset/synthetic_transactions.csv")
        gb = GraphBuilder()
        g = gb.build_from_dataframe(df.head(1000))
        fe = FeatureEngineer()
        features_df = fe.extract_features(g)
        print("Extracted Features:")
        print(features_df.head())
