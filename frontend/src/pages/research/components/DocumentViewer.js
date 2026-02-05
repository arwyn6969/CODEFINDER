import React from 'react';
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { Button, Tooltip } from 'antd';
import { ZoomInOutlined, ZoomOutOutlined, ReloadOutlined } from '@ant-design/icons';

const DocumentViewer = ({ src, alt = "Document" }) => {
    return (
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#000', overflow: 'hidden' }}>
            <TransformWrapper
                initialScale={1}
                minScale={0.5}
                maxScale={4}
                centerOnInit={true}
            >
                {({ zoomIn, zoomOut, resetTransform }) => (
                    <React.Fragment>
                        <div style={{ 
                            position: 'absolute', 
                            top: 16, 
                            right: 16, 
                            zIndex: 10,
                            display: 'flex',
                            gap: 8
                        }}>
                            <Tooltip title="Zoom In">
                                <Button shape="circle" icon={<ZoomInOutlined />} onClick={() => zoomIn()} />
                            </Tooltip>
                            <Tooltip title="Zoom Out">
                                <Button shape="circle" icon={<ZoomOutOutlined />} onClick={() => zoomOut()} />
                            </Tooltip>
                            <Tooltip title="Reset">
                                <Button shape="circle" icon={<ReloadOutlined />} onClick={() => resetTransform()} />
                            </Tooltip>
                        </div>

                        <TransformComponent wrapperStyle={{ width: '100%', height: '100%' }} contentStyle={{ width: '100%', height: '100%' }}>
                            <img 
                                src={src} 
                                alt={alt} 
                                style={{ width: '100%', height: 'auto', objectFit: 'contain' }} 
                            />
                        </TransformComponent>
                    </React.Fragment>
                )}
            </TransformWrapper>
        </div>
    );
};

export default DocumentViewer;
