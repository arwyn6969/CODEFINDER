import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import TheDesk from './TheDesk';
import researchService from '../../../services/researchService';

jest.mock('../../../services/researchService');

describe('TheDesk', () => {
  let getSelectionSpy;

  beforeEach(() => {
    jest.clearAllMocks();

    researchService.getDocumentContent.mockResolvedValue({
      content: 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG',
    });
    researchService.calculateGematria.mockResolvedValue({
      results: {
        english_standard: { score: 213, breakdown: [200, 5, 8] },
        hebrew_standard: { score: 0, breakdown: [] },
        greek_isopsephy: { score: 0, breakdown: [] },
      },
      persisted_patterns: 0,
    });

    getSelectionSpy = jest.spyOn(window, 'getSelection').mockReturnValue({
      toString: () => 'THE',
    });
  });

  afterEach(() => {
    getSelectionSpy.mockRestore();
  });

  it('renders zero-score gematria results without stringifying objects', async () => {
    render(
      <TheDesk
        currentDocument={{
          id: 1,
          filename: 'verify-alpha.txt',
          page_count: 9,
          total_pages: 1,
          processing_status: 'completed',
        }}
        onSetDocument={jest.fn()}
      />
    );

    const documentText = await screen.findByText(/the quick brown fox jumps over the lazy dog/i);
    fireEvent.mouseUp(documentText);

    await waitFor(() => {
      expect(researchService.calculateGematria).toHaveBeenCalledWith('THE');
    });

    expect(await screen.findByText(/hebrew standard/i)).toBeInTheDocument();
    expect(screen.getByText(/greek isopsephy/i)).toBeInTheDocument();
    expect(screen.queryByText('[object Object]')).not.toBeInTheDocument();
    expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/pages:/i)).toHaveTextContent('Pages: 1');
    expect(screen.getByText('completed')).toBeInTheDocument();
  });
});
