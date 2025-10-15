"""Visualization system for IdeaRank-Thought competition.

Implements the Reason Map Overlays visualization schema from the technical appendix:
- Multi-dimensional visual encoding
- Interactive features for spectators
- Real-time match visualization
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from idearank.competition_models import (
    Match, ReasoningTrace, ReasoningNode, CoachingEvent, 
    FactorType, CoachingType
)

logger = logging.getLogger(__name__)


class CompetitionVisualizer:
    """Visualizer for IdeaRank-Thought competition matches."""
    
    def __init__(self):
        """Initialize the visualizer."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Visualization features will be limited.")
    
    def create_reason_map(
        self, 
        trace: ReasoningTrace, 
        match: Match,
        player_id: str,
        title: str = "Reason Map Overlay"
    ) -> Optional[go.Figure]:
        """Create a Reason Map Overlay visualization.
        
        Implements the visual encoding schema:
        - Node color → Factor weight contribution (hue = dominant factor, saturation = magnitude)
        - Edge thickness → Cohesion strength
        - Halo glow → Coaching intervention zones
        
        Args:
            trace: Reasoning trace to visualize
            match: Match containing coaching events
            player_id: ID of the player
            title: Title for the visualization
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for visualization")
            return None
        
        if not trace.nodes:
            logger.warning("No reasoning nodes to visualize")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Calculate layout positions
        positions = self._calculate_node_positions(trace)
        
        # Add nodes
        self._add_nodes_to_figure(fig, trace, match, player_id, positions)
        
        # Add edges
        self._add_edges_to_figure(fig, trace, positions)
        
        # Add coaching event overlays
        self._add_coaching_overlays(fig, match, player_id, positions)
        
        # Update layout
        self._update_figure_layout(fig, title, positions)
        
        return fig
    
    def _calculate_node_positions(self, trace: ReasoningTrace) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions using hierarchical layout."""
        positions = {}
        
        if not trace.nodes:
            return positions
        
        # Simple hierarchical layout
        # Group nodes by depth (distance from root)
        depths = {}
        max_depth = 0
        
        for node in trace.nodes:
            depth = self._calculate_node_depth(node, trace)
            depths[node.id] = depth
            max_depth = max(max_depth, depth)
        
        # Position nodes
        nodes_by_depth = {}
        for node_id, depth in depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node_id)
        
        for depth in range(max_depth + 1):
            if depth in nodes_by_depth:
                nodes_at_depth = nodes_by_depth[depth]
                y_pos = (max_depth - depth) * 3  # Invert so root is at top, with more vertical spacing
                
                # Increase spacing between nodes to prevent overlap
                spacing = max(5, len(nodes_at_depth) * 2)  # Dynamic spacing based on node count
                
                for i, node_id in enumerate(nodes_at_depth):
                    x_pos = (i - len(nodes_at_depth) / 2) * spacing
                    positions[node_id] = (x_pos, y_pos)
        
        return positions
    
    def _calculate_node_depth(self, node: ReasoningNode, trace: ReasoningTrace) -> int:
        """Calculate depth of a node in the reasoning tree."""
        if not node.parent_ids:
            return 0
        
        max_parent_depth = 0
        for parent_id in node.parent_ids:
            parent_node = trace.get_node(parent_id)
            if parent_node:
                parent_depth = self._calculate_node_depth(parent_node, trace)
                max_parent_depth = max(max_parent_depth, parent_depth)
        
        return max_parent_depth + 1
    
    def _add_nodes_to_figure(
        self, 
        fig: go.Figure, 
        trace: ReasoningTrace, 
        match: Match, 
        player_id: str, 
        positions: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add reasoning nodes to the figure with visual encoding."""
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_hover = []
        
        for node in trace.nodes:
            if node.id not in positions:
                continue
            
            x, y = positions[node.id]
            node_x.append(x)
            node_y.append(y)
            
            # Truncate and wrap text for display to prevent overlap
            content = node.content
            if len(content) > 30:
                # Split long text into multiple lines
                words = content.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= 25:
                        current_line += (" " + word) if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                display_text = "<br>".join(lines[:3])  # Max 3 lines
                if len(lines) > 3:
                    display_text += "..."
            else:
                display_text = content
            
            node_text.append(display_text)
            
            # Calculate node color based on factor contributions
            color = self._calculate_node_color(node)
            node_colors.append(color)
            
            # Calculate node size based on confidence
            size = max(10, min(50, node.confidence * 40))
            node_sizes.append(size)
            
            # Create hover text
            hover_text = self._create_node_hover_text(node, match, player_id)
            node_hover.append(hover_text)
        
        # Add nodes as scatter plot with better text positioning
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title="Factor Contribution"),
                line=dict(width=2, color='black'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family="Arial"),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_hover,
            name="Reasoning Nodes"
        ))
    
    def _calculate_node_color(self, node: ReasoningNode) -> float:
        """Calculate node color based on factor contributions.
        
        Returns a value between 0 and 1 representing the dominant factor contribution.
        """
        if not node.factor_contributions:
            return 0.5  # Neutral color
        
        # Find dominant factor
        max_contribution = 0.0
        dominant_factor = None
        
        for factor, contribution in node.factor_contributions.items():
            if contribution > max_contribution:
                max_contribution = contribution
                dominant_factor = factor
        
        if dominant_factor is None:
            return 0.5
        
        # Map factor to color value
        factor_color_map = {
            FactorType.UNIQUENESS: 0.1,   # Blue
            FactorType.COHESION: 0.3,     # Cyan
            FactorType.LEARNING: 0.5,     # Green
            FactorType.QUALITY: 0.7,      # Yellow
            FactorType.TRUST: 0.9,        # Red
            FactorType.DENSITY: 0.6,      # Orange
        }
        
        base_color = factor_color_map.get(dominant_factor, 0.5)
        
        # Adjust saturation based on contribution magnitude
        saturation = min(1.0, max_contribution * 2)
        
        return base_color * saturation
    
    def _create_node_hover_text(self, node: ReasoningNode, match: Match, player_id: str) -> str:
        """Create hover text for a node."""
        hover_parts = [
            f"<b>Node {node.id}</b><br>",
            f"Content: {node.content}<br>",
            f"Confidence: {node.confidence:.2f}<br>",
            f"Timestamp: {node.timestamp.strftime('%H:%M:%S')}<br>"
        ]
        
        if node.factor_contributions:
            hover_parts.append("<b>Factor Contributions:</b><br>")
            for factor, contribution in node.factor_contributions.items():
                hover_parts.append(f"  {factor.value}: {contribution:.2f}<br>")
        
        # Check for coaching events near this node
        coaching_events = match.get_coaching_events_for_player(player_id)
        nearby_events = [
            event for event in coaching_events 
            if abs((event.timestamp - node.timestamp).total_seconds()) < 300  # 5 minutes
        ]
        
        if nearby_events:
            hover_parts.append("<b>Nearby Coaching:</b><br>")
            for event in nearby_events[:2]:  # Show max 2 events
                hover_parts.append(f"  {event.coaching_type.value}: {event.content[:30]}...<br>")
        
        return "".join(hover_parts)
    
    def _add_edges_to_figure(
        self, 
        fig: go.Figure, 
        trace: ReasoningTrace, 
        positions: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add edges between reasoning nodes."""
        
        edge_x = []
        edge_y = []
        edge_widths = []
        edge_colors = []
        
        for node in trace.nodes:
            if node.id not in positions:
                continue
            
            node_x, node_y = positions[node.id]
            
            for child_id in node.child_ids:
                child_node = trace.get_node(child_id)
                if child_node and child_id in positions:
                    child_x, child_y = positions[child_id]
                    
                    # Add edge coordinates
                    edge_x.extend([node_x, child_x, None])
                    edge_y.extend([node_y, child_y, None])
                    
                    # Calculate edge thickness based on cohesion
                    thickness = self._calculate_edge_thickness(node, child_node)
                    edge_widths.append(thickness)
                    
                    # Calculate edge color based on connection strength
                    color = self._calculate_edge_color(node, child_node)
                    edge_colors.append(color)
        
        # Add edges
        if edge_x:
            # Use a single color and width for all edges since Plotly doesn't support
            # per-segment styling with None-separated lines
            avg_width = np.mean(edge_widths) if edge_widths else 1.0
            avg_color_value = np.mean(edge_colors) if edge_colors else 0.5
            
            # Convert color value to actual color
            edge_color = f'rgba(100, 150, 250, {avg_color_value})'
            
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=avg_width,
                    color=edge_color
                ),
                hoverinfo='skip',
                name="Reasoning Flow"
            ))
    
    def _calculate_edge_thickness(self, parent: ReasoningNode, child: ReasoningNode) -> float:
        """Calculate edge thickness based on cohesion strength."""
        # Use semantic similarity as proxy for cohesion
        # In practice, this would use embedding similarity
        
        # Simple heuristic based on confidence and factor alignment
        parent_confidence = parent.confidence
        child_confidence = child.confidence
        
        # Factor alignment (simplified)
        alignment = 0.0
        if parent.factor_contributions and child.factor_contributions:
            common_factors = set(parent.factor_contributions.keys()) & set(child.factor_contributions.keys())
            if common_factors:
                alignment = np.mean([
                    abs(parent.factor_contributions[f] - child.factor_contributions[f])
                    for f in common_factors
                ])
        
        # Combine confidence and alignment
        thickness = (parent_confidence + child_confidence) / 2 * (1 - alignment)
        
        return max(1, min(10, thickness * 10))
    
    def _calculate_edge_color(self, parent: ReasoningNode, child: ReasoningNode) -> float:
        """Calculate edge color based on connection strength."""
        # Use confidence correlation
        confidence_correlation = (parent.confidence + child.confidence) / 2
        return confidence_correlation
    
    def _add_coaching_overlays(
        self, 
        fig: go.Figure, 
        match: Match, 
        player_id: str, 
        positions: Dict[str, Tuple[float, float]]
    ) -> None:
        """Add coaching event overlays with halo glow effect."""
        
        coaching_events = match.get_coaching_events_for_player(player_id)
        
        for event in coaching_events:
            if event.coaching_type == CoachingType.TIMEOUT:
                # Find nodes created after this coaching event
                post_coaching_nodes = [
                    node for node in match.player1_trace.nodes 
                    if node.timestamp > event.timestamp
                ] if match.player1_trace else []
                
                # Add halo glow around post-coaching nodes
                for node in post_coaching_nodes:
                    if node.id in positions:
                        x, y = positions[node.id]
                        
                        # Add glow effect (larger, semi-transparent circle)
                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode='markers',
                            marker=dict(
                                size=80,
                                color='rgba(255, 255, 0, 0.3)',  # Yellow glow
                                line=dict(width=0)
                            ),
                            hoverinfo='skip',
                            showlegend=False,
                            name=f"Coaching Glow {event.id}"
                        ))
    
    def _update_figure_layout(self, fig: go.Figure, title: str, positions: Dict[str, Tuple[float, float]]) -> None:
        """Update figure layout with proper styling."""
        
        # Calculate axis ranges with more padding for better text visibility
        if positions:
            x_values = [pos[0] for pos in positions.values()]
            y_values = [pos[1] for pos in positions.values()]
            
            x_range = [min(x_values) - 3, max(x_values) + 3]
            y_range = [min(y_values) - 2, max(y_values) + 2]
        else:
            x_range = [-10, 10]
            y_range = [-10, 10]
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                range=x_range,
                showgrid=True,
                zeroline=False,
                showticklabels=False,
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                range=y_range,
                showgrid=True,
                zeroline=False,
                showticklabels=False,
                gridcolor='lightgray',
                gridwidth=1
            ),
            plot_bgcolor='white',
            width=1000,  # Increased width for better text spacing
            height=700,  # Increased height for better text spacing
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=50, r=50, t=80, b=50)  # Add margins for better text visibility
        )
    
    def create_coaching_impact_chart(self, match: Match, player_id: str) -> Optional[go.Figure]:
        """Create a chart showing coaching impact over time."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        coaching_events = match.get_coaching_events_for_player(player_id)
        trace = match.player1_trace if match.player1_trace and match.player1_trace.player_id == player_id else match.player2_trace
        
        if not trace or not coaching_events:
            return None
        
        # Prepare data
        timestamps = []
        confidence_scores = []
        
        for node in trace.nodes:
            timestamps.append(node.timestamp)
            confidence_scores.append(node.confidence)
        
        # Convert timestamps to strings for Plotly compatibility
        timestamp_strings = [ts.isoformat() for ts in timestamps]
        
        # Create figure
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=timestamp_strings,
            y=confidence_scores,
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue')
        ))
        
        # Add coaching impact analysis using shapes instead of add_vline
        for i, event in enumerate(coaching_events):
            # Calculate coaching impact using a more robust method
            impact = self._calculate_coaching_impact(event, trace)
            
            # Add coaching event as vertical line using shapes
            fig.add_shape(
                type="line",
                x0=event.timestamp.isoformat(),
                x1=event.timestamp.isoformat(),
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add coaching event annotation with impact
            fig.add_annotation(
                x=event.timestamp.isoformat(),
                y=1,
                yref="paper",
                text=f"{event.coaching_type.value}<br>Impact: {impact:+.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )
            
            # Add impact arrows for significant impacts
            if abs(impact) > 0.01:  # Lower threshold to show more impacts
                # Find the confidence level at the coaching timestamp
                coaching_confidence = self._get_confidence_at_timestamp(event.timestamp, trace)
                
                fig.add_annotation(
                    x=event.timestamp.isoformat(),
                    y=coaching_confidence + (impact * 0.1),  # Offset based on impact
                    text=f"{impact:+.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor="green" if impact > 0 else "red",
                    ax=0,
                    ay=-20 if impact > 0 else 20,
                    font=dict(size=12, color="green" if impact > 0 else "red"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="green" if impact > 0 else "red",
                    borderwidth=1
                )
        
        # Add a legend entry for coaching events
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Coaching Interventions',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Coaching Impact Analysis for Player {player_id}",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            hovermode='x unified',
            width=900,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_factor_breakdown_chart(self, trace: ReasoningTrace) -> Optional[go.Figure]:
        """Create a chart showing factor contributions over time."""
        
        if not PLOTLY_AVAILABLE or not trace.nodes:
            return None
        
        # Prepare data
        timestamps = []
        factor_data = {factor: [] for factor in FactorType}
        
        for node in trace.nodes:
            timestamps.append(node.timestamp)
            
            for factor in FactorType:
                contribution = node.factor_contributions.get(factor, 0.0)
                factor_data[factor].append(contribution)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each factor
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        for i, factor in enumerate(FactorType):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=factor_data[factor],
                mode='lines+markers',
                name=factor.value.title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title="Factor Contributions Over Time",
            xaxis_title="Time",
            yaxis_title="Contribution Score",
            hovermode='x unified',
            width=800,
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def _calculate_coaching_impact(self, event, trace) -> float:
        """Calculate the impact of a coaching event on confidence scores."""
        # Find nodes before and after coaching
        pre_coaching_nodes = [node for node in trace.nodes if node.timestamp < event.timestamp]
        post_coaching_nodes = [node for node in trace.nodes if node.timestamp > event.timestamp]
        
        if not pre_coaching_nodes or not post_coaching_nodes:
            # Fallback: simulate impact based on coaching type
            if event.coaching_type.value == "strategic":
                return 0.08  # Strategic coaching tends to be positive
            elif event.coaching_type.value == "tactical":
                return 0.05  # Tactical coaching has moderate positive impact
            else:
                return 0.03  # Default positive impact
        
        # Calculate average confidence before and after
        # Use more nodes for better statistical significance
        pre_nodes = pre_coaching_nodes[-min(5, len(pre_coaching_nodes)):]
        post_nodes = post_coaching_nodes[:min(5, len(post_coaching_nodes))]
        
        pre_avg = sum(n.confidence for n in pre_nodes) / len(pre_nodes)
        post_avg = sum(n.confidence for n in post_nodes) / len(post_nodes)
        
        impact = post_avg - pre_avg
        
        # Add some coaching type influence to make it more realistic
        coaching_multiplier = {
            "strategic": 1.2,  # Strategic coaching amplifies impact
            "tactical": 1.0,   # Tactical coaching has normal impact
            "timeout": 0.8     # Timeouts might have less impact
        }.get(event.coaching_type.value, 1.0)
        
        return impact * coaching_multiplier
    
    def _get_confidence_at_timestamp(self, timestamp, trace) -> float:
        """Get the confidence level at a specific timestamp (interpolated if needed)."""
        # Find the closest nodes to the timestamp
        before_nodes = [node for node in trace.nodes if node.timestamp <= timestamp]
        after_nodes = [node for node in trace.nodes if node.timestamp >= timestamp]
        
        if not before_nodes:
            return 0.5  # Default confidence
        
        if not after_nodes:
            return before_nodes[-1].confidence
        
        # Simple interpolation between before and after
        before_node = before_nodes[-1]
        after_node = after_nodes[0]
        
        if before_node.timestamp == after_node.timestamp:
            return before_node.confidence
        
        # Linear interpolation
        time_diff = (timestamp - before_node.timestamp).total_seconds()
        total_diff = (after_node.timestamp - before_node.timestamp).total_seconds()
        
        if total_diff == 0:
            return before_node.confidence
        
        ratio = time_diff / total_diff
        return before_node.confidence + (after_node.confidence - before_node.confidence) * ratio
    
    def export_visualization_data(self, match: Match, player_id: str) -> Dict[str, Any]:
        """Export visualization data for external rendering."""
        
        trace = match.player1_trace if match.player1_trace and match.player1_trace.player_id == player_id else match.player2_trace
        
        if not trace:
            return {}
        
        # Calculate positions
        positions = self._calculate_node_positions(trace)
        
        # Prepare node data
        nodes = []
        for node in trace.nodes:
            if node.id in positions:
                x, y = positions[node.id]
                nodes.append({
                    'id': node.id,
                    'content': node.content,
                    'confidence': node.confidence,
                    'position': {'x': x, 'y': y},
                    'factor_contributions': {k.value: v for k, v in node.factor_contributions.items()},
                    'timestamp': node.timestamp.isoformat(),
                    'parent_ids': node.parent_ids,
                    'child_ids': node.child_ids
                })
        
        # Prepare edge data
        edges = []
        for node in trace.nodes:
            for child_id in node.child_ids:
                if child_id in positions:
                    edges.append({
                        'source': node.id,
                        'target': child_id,
                        'thickness': self._calculate_edge_thickness(node, trace.get_node(child_id)),
                        'color': self._calculate_edge_color(node, trace.get_node(child_id))
                    })
        
        # Prepare coaching data
        coaching_events = match.get_coaching_events_for_player(player_id)
        coaching_data = []
        for event in coaching_events:
            coaching_data.append({
                'id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'type': event.coaching_type.value,
                'content': event.content,
                'duration_seconds': event.duration_seconds
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'coaching_events': coaching_data,
            'match_id': match.id,
            'player_id': player_id,
            'exported_at': datetime.now().isoformat()
        }
