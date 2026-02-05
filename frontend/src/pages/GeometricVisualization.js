import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

/**
 * GeometricVisualization Component
 * 
 * Renders D3.js-powered geometric pattern visualization for document pages.
 * Displays Golden Ratio distances and Right Angle patterns detected in text.
 */
const GeometricVisualization = ({ 
  documentId, 
  pageNumber, 
  apiBaseUrl = '/api/patterns' 
}) => {
  const svgRef = useRef(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch geometric analysis data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${apiBaseUrl}/geometric/${documentId}/${pageNumber}?filter_significant=true&max_patterns=20`
        );
        
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        const visualizationData = JSON.parse(result.visualization_json);
        setData(visualizationData);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    if (documentId && pageNumber) {
      fetchData();
    }
  }, [documentId, pageNumber, apiBaseUrl]);
  
  // Render D3 visualization
  useEffect(() => {
    if (!data || !svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous
    
    const { config, data: vizData } = data;
    const width = config.width || 800;
    const height = config.height || 1000;
    const margin = config.margin || { top: 40, right: 40, bottom: 40, left: 40 };
    
    // Set SVG dimensions
    svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom);
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Background image (if provided)
    if (config.background_image && config.background_image.url) {
      g.append('image')
        .attr('href', config.background_image.url)
        .attr('width', width)
        .attr('height', height)
        .attr('opacity', config.background_image.opacity || 0.3);
    }
    
    // Create scales based on coordinate system
    const coordSys = vizData.coordinate_system || { width: 612, height: 792 };
    const xScale = d3.scaleLinear()
      .domain([0, coordSys.width])
      .range([0, width]);
    const yScale = d3.scaleLinear()
      .domain([0, coordSys.height])
      .range([0, height]);
    
    // Draw elements
    vizData.elements.forEach((element, idx) => {
      const points = element.points || [];
      
      if (element.type === 'line' && points.length >= 2) {
        // Golden Ratio lines
        g.append('line')
          .attr('x1', xScale(points[0].x))
          .attr('y1', yScale(points[0].y))
          .attr('x2', xScale(points[1].x))
          .attr('y2', yScale(points[1].y))
          .attr('stroke', element.style?.stroke || '#FFD700')
          .attr('stroke-width', element.style?.stroke_width || 2)
          .attr('opacity', element.style?.opacity || 0.8)
          .attr('class', 'geometric-line');
      } else if (element.type === 'triangle' && points.length >= 3) {
        // Right Angle triangles
        const pathData = `M ${xScale(points[0].x)} ${yScale(points[0].y)} 
                          L ${xScale(points[1].x)} ${yScale(points[1].y)} 
                          L ${xScale(points[2].x)} ${yScale(points[2].y)} Z`;
        g.append('path')
          .attr('d', pathData)
          .attr('stroke', element.style?.stroke || '#DC143C')
          .attr('stroke-width', element.style?.stroke_width || 2)
          .attr('fill', element.style?.fill || 'none')
          .attr('opacity', element.style?.opacity || 0.7)
          .attr('class', 'geometric-triangle');
      }
      
      // Draw point markers
      points.forEach(point => {
        g.append('circle')
          .attr('cx', xScale(point.x))
          .attr('cy', yScale(point.y))
          .attr('r', 4)
          .attr('fill', element.style?.stroke || '#333')
          .attr('class', 'geometric-point');
        
        // Character labels
        if (point.label) {
          g.append('text')
            .attr('x', xScale(point.x) + 6)
            .attr('y', yScale(point.y) - 6)
            .attr('font-size', '12px')
            .attr('fill', '#333')
            .text(point.label);
        }
      });
    });
    
    // Add legend
    const legendData = [
      { color: '#FFD700', label: 'Golden Ratio' },
      { color: '#DC143C', label: 'Right Angle' }
    ];
    
    const legend = svg.append('g')
      .attr('transform', `translate(${margin.left + 10}, ${margin.top + 10})`);
    
    legendData.forEach((item, i) => {
      const row = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      row.append('rect')
        .attr('width', 16)
        .attr('height', 16)
        .attr('fill', item.color)
        .attr('opacity', 0.8);
      
      row.append('text')
        .attr('x', 22)
        .attr('y', 12)
        .attr('font-size', '12px')
        .text(item.label);
    });
    
  }, [data]);
  
  if (loading) {
    return <div className="geometric-loading">Loading geometric analysis...</div>;
  }
  
  if (error) {
    return <div className="geometric-error">Error: {error}</div>;
  }
  
  return (
    <div className="geometric-visualization-container">
      <h3>Geometric Pattern Analysis</h3>
      <p>Page {pageNumber} • {data?.data?.elements?.length || 0} patterns detected</p>
      <svg ref={svgRef} className="geometric-svg" />
    </div>
  );
};

export default GeometricVisualization;
