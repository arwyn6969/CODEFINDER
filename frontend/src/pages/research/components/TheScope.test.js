import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import TheScope from './TheScope';
import researchService from '../../../services/researchService';

jest.mock('../../../services/researchService');

describe('TheScope', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    researchService.transliterate.mockResolvedValue([
      { hebrew: 'פפי', description: 'Standard (PPI)' },
    ]);
    researchService.findELS.mockResolvedValue({
      total_length: 35,
      found_count: 1,
      matches: [
        {
          term: 'FOX',
          skip: 1,
          start_index: 13,
          end_index: 15,
          location: [13, 15],
          direction: 'forward',
        },
      ],
      persisted_patterns: 0,
    });
    researchService.getELSVisualization.mockResolvedValue({
      grid: [
        [
          { char: 'F', index: 13, row: 0 },
          { char: 'O', index: 14, row: 0 },
          { char: 'X', index: 15, row: 0 },
        ],
      ],
      dimensions: { rows: 1, cols: 3, row_width: 1 },
      center_index: 13,
      skip: 1,
      viewport: { start_row: 13, start_col: 0 },
      highlights: [
        { index: 13, grid_row: 0, grid_col: 0, visible: true },
        { index: 14, grid_row: 0, grid_col: 1, visible: true },
        { index: 15, grid_row: 0, grid_col: 2, visible: true },
      ],
    });
  });

  it('renders normalized ELS start indices and opens the matrix view', async () => {
    render(<TheScope />);

    await userEvent.type(screen.getByPlaceholderText(/enter english term/i), 'FOX');
    await userEvent.click(screen.getByRole('button', { name: /initiate scan/i }));

    expect(await screen.findByText('13')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /view/i }));

    await waitFor(() => {
      expect(researchService.getELSVisualization).toHaveBeenCalledWith(
        expect.objectContaining({ centerIndex: 13, skip: 1, termLength: 3 })
      );
    });

    expect(await screen.findByText(/els matrix view/i)).toBeInTheDocument();
    await waitFor(() => {
      expect(document.body.querySelector('[title="Index: 13"]')).not.toBeNull();
    });
  });
});
