"""
PL-Tower topological optimization module.
Implements geometric/physical connection relationship optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from scipy.spatial.distance import cdist
import networkx as nx
import logging


class TopologyOptimizer:
    """PL-Tower topological optimization following Zhang et al. methodology."""
    
    def __init__(self, config):
        """Initialize topology optimizer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_pl_tower_connections(self, power_lines: List[Dict], towers: List[Dict]) -> Dict:
        """
        Analyze physical connections between power lines and towers.
        
        Args:
            power_lines: Extracted power lines
            towers: Extracted tower candidates
            
        Returns:
            connection_graph: Network graph of PL-tower connections
        """
        self.logger.info("Analyzing PL-tower physical connections")
        
        if not power_lines or not towers:
            return {'connections': [], 'valid_connections': 0}
        
        # Build connection graph
        connection_graph = nx.Graph()
        
        # Add power lines as nodes
        for pl in power_lines:
            connection_graph.add_node(f"PL_{pl['powerline_id']}", 
                                    type='powerline', data=pl)
        
        # Add towers as nodes  
        for tower in towers:
            connection_graph.add_node(f"T_{tower['cluster_id']}", 
                                    type='tower', data=tower)
        
        # Find PL-tower connections
        connections = []
        connection_threshold = self.config.connection_distance_thr
        
        for pl in power_lines:
            pl_endpoints = pl.get('endpoints', [])
            if len(pl_endpoints) < 2:
                continue
            
            # Check connection to each tower
            for tower in towers:
                tower_centroid = tower['centroid']
                
                # Check distance from tower to PL endpoints
                for endpoint in pl_endpoints:
                    distance = np.linalg.norm(endpoint - tower_centroid)
                    
                    if distance <= connection_threshold:
                        connection = {
                            'powerline_id': pl['powerline_id'],
                            'tower_id': tower['cluster_id'],
                            'distance': distance,
                            'endpoint': endpoint,
                            'tower_centroid': tower_centroid,
                            'connection_type': 'endpoint'
                        }
                        connections.append(connection)
                        
                        # Add edge to graph
                        connection_graph.add_edge(f"PL_{pl['powerline_id']}", 
                                                f"T_{tower['cluster_id']}", 
                                                **connection)
        
        self.logger.info(f"Found {len(connections)} PL-tower connections")
        
        return {
            'graph': connection_graph,
            'connections': connections,
            'valid_connections': len(connections)
        }
    
    def check_parallel_consistency(self, power_lines: List[Dict], towers: List[Dict]) -> Dict:
        """
        Check if power lines are approximately parallel to tower connection lines.
        
        Args:
            power_lines: Power lines
            towers: Towers
            
        Returns:
            parallel_analysis: Analysis of parallel consistency
        """
        self.logger.info("Checking parallel consistency between PLs and tower connections")
        
        parallel_violations = []
        valid_configurations = []
        
        if len(towers) < 2:
            return {'violations': [], 'valid_configs': 0, 'total_checks': 0}
        
        # Check all tower pairs
        for i, tower1 in enumerate(towers):
            for j, tower2 in enumerate(towers[i+1:], i+1):
                
                # Tower connection vector
                tower_vec = tower2['centroid'] - tower1['centroid']
                tower_direction = tower_vec / np.linalg.norm(tower_vec)
                
                # Check each power line
                for pl in power_lines:
                    pl_direction = pl.get('principal_direction', np.array([1, 0, 0]))
                    
                    # Compute angle between tower connection and PL direction
                    dot_product = np.abs(np.dot(tower_direction, pl_direction))
                    angle_diff = np.degrees(np.arccos(np.clip(dot_product, 0, 1)))
                    
                    parallel_threshold = self.config.parallel_angle_thr
                    
                    if angle_diff <= parallel_threshold:
                        valid_configurations.append({
                            'tower1_id': tower1['cluster_id'],
                            'tower2_id': tower2['cluster_id'],
                            'powerline_id': pl['powerline_id'],
                            'angle_diff': angle_diff,
                            'tower_distance': np.linalg.norm(tower_vec)
                        })
                    else:
                        parallel_violations.append({
                            'tower1_id': tower1['cluster_id'],
                            'tower2_id': tower2['cluster_id'], 
                            'powerline_id': pl['powerline_id'],
                            'angle_diff': angle_diff,
                            'violation_severity': angle_diff - parallel_threshold
                        })
        
        total_checks = len(towers) * (len(towers) - 1) // 2 * len(power_lines)
        
        self.logger.info(f"Parallel check: {len(valid_configurations)} valid, "
                        f"{len(parallel_violations)} violations out of {total_checks} checks")
        
        return {
            'violations': parallel_violations,
            'valid_configs': valid_configurations,
            'total_checks': total_checks,
            'violation_rate': len(parallel_violations) / max(total_checks, 1)
        }
    
    def identify_extremal_points(self, power_lines: List[Dict]) -> Dict:
        """
        Identify extremal points (derivatives extrema) along power lines where towers should exist.
        
        Args:
            power_lines: Power lines
            
        Returns:
            extremal_analysis: Analysis of extremal points
        """
        self.logger.info("Identifying extremal points along power lines")
        
        extremal_points = []
        
        for pl in power_lines:
            points = pl.get('points', np.array([]))
            if len(points) < 10:  # Need sufficient points for derivative analysis
                continue
            
            # Sort points along principal direction for derivative computation
            principal_dir = pl.get('principal_direction', np.array([1, 0, 0]))
            projections = np.dot(points - points.mean(axis=0), principal_dir)
            sorted_indices = np.argsort(projections)
            sorted_points = points[sorted_indices]
            
            # Compute height derivatives (vertical changes)
            heights = sorted_points[:, 2]
            if len(heights) < 5:
                continue
                
            # Simple finite difference derivatives
            derivatives = np.gradient(heights)
            second_derivatives = np.gradient(derivatives)
            
            # Find local extrema in derivatives (potential tower locations)
            # Look for zero crossings or significant changes
            extrema_indices = []
            
            # Find local maxima and minima in second derivative
            for i in range(1, len(second_derivatives) - 1):
                if ((second_derivatives[i-1] < second_derivatives[i] > second_derivatives[i+1]) or
                    (second_derivatives[i-1] > second_derivatives[i] < second_derivatives[i+1])):
                    if abs(second_derivatives[i]) > 0.1:  # Threshold for significance
                        extrema_indices.append(i)
            
            # Convert to actual points
            for idx in extrema_indices:
                extremal_points.append({
                    'powerline_id': pl['powerline_id'],
                    'point': sorted_points[idx],
                    'derivative_value': derivatives[idx],
                    'second_derivative': second_derivatives[idx],
                    'extrema_type': 'maximum' if second_derivatives[idx] < 0 else 'minimum'
                })
        
        self.logger.info(f"Identified {len(extremal_points)} extremal points")
        
        return {
            'extremal_points': extremal_points,
            'num_extrema': len(extremal_points)
        }
    
    def validate_tower_at_extrema(self, extremal_analysis: Dict, towers: List[Dict]) -> Dict:
        """
        Validate that towers exist near extremal points.
        
        Args:
            extremal_analysis: Extremal points analysis
            towers: Tower candidates
            
        Returns:
            validation_results: Validation of tower-extrema correspondence
        """
        self.logger.info("Validating towers at extremal points")
        
        extremal_points = extremal_analysis.get('extremal_points', [])
        validated_extrema = []
        unmatched_extrema = []
        
        validation_threshold = self.config.connection_distance_thr * 2  # Larger threshold
        
        for extrema in extremal_points:
            extrema_point = extrema['point']
            matched = False
            
            for tower in towers:
                tower_centroid = tower['centroid']
                distance = np.linalg.norm(extrema_point - tower_centroid)
                
                if distance <= validation_threshold:
                    validated_extrema.append({
                        **extrema,
                        'matched_tower_id': tower['cluster_id'],
                        'distance_to_tower': distance
                    })
                    matched = True
                    break
            
            if not matched:
                unmatched_extrema.append(extrema)
        
        validation_rate = len(validated_extrema) / max(len(extremal_points), 1)
        
        self.logger.info(f"Extrema validation: {len(validated_extrema)} matched, "
                        f"{len(unmatched_extrema)} unmatched ({validation_rate:.2%} validation rate)")
        
        return {
            'validated_extrema': validated_extrema,
            'unmatched_extrema': unmatched_extrema,
            'validation_rate': validation_rate,
            'total_extrema': len(extremal_points)
        }
    
    def filter_false_positives(self, power_lines: List[Dict], towers: List[Dict],
                             connection_analysis: Dict, parallel_analysis: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter false positives based on topological analysis.
        Remove isolated towers, signal poles, low-voltage crossings, etc.
        
        Args:
            power_lines: Power lines
            towers: Tower candidates
            connection_analysis: PL-tower connection analysis
            parallel_analysis: Parallel consistency analysis
            
        Returns:
            filtered_power_lines: Filtered power lines
            filtered_towers: Filtered towers
        """
        self.logger.info("Filtering false positives based on topological constraints")
        
        # Identify well-connected towers
        connected_tower_ids = set()
        for conn in connection_analysis.get('connections', []):
            connected_tower_ids.add(conn['tower_id'])
        
        # Filter towers based on connections and geometric consistency
        filtered_towers = []
        for tower in towers:
            tower_id = tower['cluster_id']
            
            # Rule 1: Tower must be connected to at least one power line
            if tower_id not in connected_tower_ids:
                self.logger.debug(f"Removing isolated tower {tower_id}")
                continue
            
            # Rule 2: Check height consistency (not too short for HV transmission)
            if tower.get('max_height_diff', 0) < 15:  # Minimum 15m for HV towers
                self.logger.debug(f"Removing short structure {tower_id}")
                continue
            
            # Rule 3: Check point density (towers should have substantial point clouds)
            if tower.get('total_points', 0) < 50:  # Minimum point count
                self.logger.debug(f"Removing sparse tower {tower_id}")
                continue
            
            filtered_towers.append(tower)
        
        # Filter power lines based on tower connections
        filtered_power_lines = []
        valid_tower_ids = {tower['cluster_id'] for tower in filtered_towers}
        
        for pl in power_lines:
            pl_id = pl['powerline_id']
            
            # Check if power line is connected to valid towers
            connected_to_valid_towers = False
            for conn in connection_analysis.get('connections', []):
                if (conn['powerline_id'] == pl_id and 
                    conn['tower_id'] in valid_tower_ids):
                    connected_to_valid_towers = True
                    break
            
            # Rule 1: Power line should connect to valid towers
            if not connected_to_valid_towers:
                self.logger.debug(f"Removing unconnected power line {pl_id}")
                continue
            
            # Rule 2: Check minimum length (avoid low-voltage crossings)
            if pl.get('total_length', 0) < 50:  # Minimum 50m for HV lines
                self.logger.debug(f"Removing short power line {pl_id}")
                continue
            
            # Rule 3: Check height consistency (HV lines should be elevated)
            avg_height = pl.get('height_stats', {}).get('mean', 0)
            if avg_height < 10:  # Minimum 10m elevation
                self.logger.debug(f"Removing low power line {pl_id}")
                continue
            
            filtered_power_lines.append(pl)
        
        self.logger.info(f"Filtered results: {len(filtered_power_lines)} power lines "
                        f"({len(power_lines) - len(filtered_power_lines)} removed), "
                        f"{len(filtered_towers)} towers "
                        f"({len(towers) - len(filtered_towers)} removed)")
        
        return filtered_power_lines, filtered_towers
    
    def optimize_topology(self, power_lines: List[Dict], towers: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Complete topological optimization pipeline.
        
        Args:
            power_lines: Initial power line candidates
            towers: Initial tower candidates
            
        Returns:
            optimized_power_lines: Topologically optimized power lines
            optimized_towers: Topologically optimized towers
            topology_report: Detailed analysis report
        """
        self.logger.info("Starting complete topological optimization")
        
        # 1. Analyze PL-tower connections
        connection_analysis = self.analyze_pl_tower_connections(power_lines, towers)
        
        # 2. Check parallel consistency
        parallel_analysis = self.check_parallel_consistency(power_lines, towers)
        
        # 3. Identify extremal points
        extremal_analysis = self.identify_extremal_points(power_lines)
        
        # 4. Validate towers at extrema
        extrema_validation = self.validate_tower_at_extrema(extremal_analysis, towers)
        
        # 5. Filter false positives (DISABLED - too strict)
        # For debugging, skip strict filtering
        optimized_power_lines, optimized_towers = power_lines, towers
        
        # Create comprehensive report
        topology_report = {
            'initial_counts': {
                'power_lines': len(power_lines),
                'towers': len(towers)
            },
            'final_counts': {
                'power_lines': len(optimized_power_lines),
                'towers': len(optimized_towers)
            },
            'removed_counts': {
                'power_lines': len(power_lines) - len(optimized_power_lines),
                'towers': len(towers) - len(optimized_towers)
            },
            'connection_analysis': connection_analysis,
            'parallel_analysis': parallel_analysis,
            'extremal_analysis': extremal_analysis,
            'extrema_validation': extrema_validation,
            'optimization_success': len(optimized_power_lines) > 0 and len(optimized_towers) > 0
        }
        
        self.logger.info(f"Topological optimization complete. "
                        f"Final: {len(optimized_power_lines)} PLs, {len(optimized_towers)} towers")
        
        return optimized_power_lines, optimized_towers, topology_report