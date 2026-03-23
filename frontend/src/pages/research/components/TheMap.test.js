import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import TheMap from './TheMap';
import researchService from '../../../services/researchService';

jest.mock('../../../services/researchService');
jest.mock('react-force-graph-2d', () => {
  return function MockForceGraph2D(props) {
    return (
      <button type="button" onClick={() => props.onNodeClick({ id: 1, name: 'Doc One' })}>
        Trigger Node Click
      </button>
    );
  };
});

describe('TheMap', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
      configurable: true,
      value: 800,
    });
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
      configurable: true,
      value: 600,
    });

    researchService.listDocuments.mockResolvedValue({
      documents: [
        { id: 1, filename: 'Doc One', processing_status: 'completed', total_pages: 1 },
        { id: 2, filename: 'Doc Two', processing_status: 'completed', total_pages: 1 },
      ],
    });
    researchService.getRelationshipNetwork.mockResolvedValue({
      network: {
        nodes: [
          { id: 1, title: 'Doc One', centrality: 1, community: 1 },
          { id: 2, title: 'Doc Two', centrality: 1, community: 1 },
        ],
        edges: [{ source: 1, target: 2, weight: 1 }],
      },
      metrics: { density: 0.5, modularity: 0.1, community_count: 1 },
    });
  });

  it('passes the full document metadata to the desk handoff', async () => {
    const onOpenDocument = jest.fn();

    render(<TheMap onOpenDocument={onOpenDocument} />);

    await userEvent.click(await screen.findByRole('button', { name: /trigger node click/i }));

    expect(await screen.findByText(/detected communities: 1/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(onOpenDocument).toHaveBeenCalledWith(
        expect.objectContaining({ id: 1, filename: 'Doc One', processing_status: 'completed' })
      );
    });
  });
});
