import React from 'react';
import { render, screen } from '@testing-library/react';

import TheGeometry from './TheGeometry';
import researchService from '../../../services/researchService';

jest.mock('../../../services/researchService');

describe('TheGeometry', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    researchService.listDocuments.mockResolvedValue({
      documents: [{ id: 1, filename: 'Doc One', processing_status: 'completed', total_pages: 1 }],
    });
  });

  it('shows the no-data state with explicit warnings', async () => {
    researchService.getGeometricAnalysis.mockResolvedValue({
      document_id: 1,
      status: 'no_data',
      warnings: ['Not enough stored pattern coordinates are available for geometry analysis.'],
      total_measurements: 0,
      angle_measurements: 0,
      distance_measurements: 0,
      ratio_measurements: 0,
      sacred_geometry_patterns: 0,
      mathematical_constants_found: [],
      significance_scores: {},
      potential_coordinates: [],
      coordinate_pairs: [],
      historical_sites: [],
    });

    render(<TheGeometry currentDocument={{ id: 1, filename: 'Doc One' }} onOpenDocument={jest.fn()} />);

    expect(await screen.findByText(/no geometry data available/i)).toBeInTheDocument();
    expect(screen.getByText(/no plausible coordinate pairs were derived/i)).toBeInTheDocument();
  });

  it('renders populated coordinate and historical match results', async () => {
    researchService.getGeometricAnalysis.mockResolvedValue({
      document_id: 1,
      status: 'ok',
      warnings: [],
      total_measurements: 8,
      angle_measurements: 3,
      distance_measurements: 3,
      ratio_measurements: 2,
      sacred_geometry_patterns: 1,
      mathematical_constants_found: ['pi'],
      significance_scores: { mean_angle_confidence: 0.9123 },
      potential_coordinates: [],
      coordinate_pairs: [
        {
          latitude: 29.9792,
          longitude: 31.1342,
          combined_confidence: 0.88,
          methods: ['direct_angle_interpretation', 'direct_angle_interpretation'],
        },
      ],
      historical_sites: [
        {
          site: {
            name: 'Great Pyramid of Giza',
            lat: 29.9792,
            lon: 31.1342,
            significance: 'ancient_wonder',
          },
          detected_coordinates: {
            latitude: 29.9792,
            longitude: 31.1342,
            combined_confidence: 0.88,
            methods: ['direct_angle_interpretation', 'direct_angle_interpretation'],
          },
          accuracy: {
            latitude_error: 0.0,
            longitude_error: 0.0,
            total_error: 0.0,
          },
          match_confidence: 0.88,
        },
      ],
    });

    render(<TheGeometry currentDocument={{ id: 1, filename: 'Doc One' }} onOpenDocument={jest.fn()} />);

    expect((await screen.findAllByText(/29\.9792°, 31\.1342°/i)).length).toBeGreaterThan(0);
    expect(screen.getByText(/great pyramid of giza/i)).toBeInTheDocument();
    expect(screen.getByText('pi')).toBeInTheDocument();
  });
});
