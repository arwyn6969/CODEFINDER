-- CODEFINDER Database Schema
-- Stores OCR results with bounding boxes for forensic analysis

-- Sources: Track different editions/scans
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_pages INTEGER,
    total_characters INTEGER,
    avg_confidence REAL,
    notes TEXT
);

-- Pages: Track per-page statistics  
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    image_width INTEGER,
    image_height INTEGER,
    char_count INTEGER,
    avg_confidence REAL,
    long_s_count INTEGER DEFAULT 0,
    ligature_count INTEGER DEFAULT 0,
    FOREIGN KEY (source_id) REFERENCES sources(id),
    UNIQUE(source_id, page_number)
);

-- Character Instances: Every detected character with bounding box
CREATE TABLE IF NOT EXISTS character_instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL,
    character TEXT NOT NULL,
    unicode_name TEXT,
    category TEXT,  -- lowercase, uppercase, punctuation, digit, long_s, ligature, other
    x INTEGER NOT NULL,       -- bounding box x
    y INTEGER NOT NULL,       -- bounding box y  
    width INTEGER NOT NULL,   -- bounding box width
    height INTEGER NOT NULL,  -- bounding box height
    confidence REAL,
    FOREIGN KEY (page_id) REFERENCES pages(id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_char_instances_char ON character_instances(character);
CREATE INDEX IF NOT EXISTS idx_char_instances_page ON character_instances(page_id);
CREATE INDEX IF NOT EXISTS idx_char_instances_category ON character_instances(category);
CREATE INDEX IF NOT EXISTS idx_pages_source ON pages(source_id);

-- Views for common queries
CREATE VIEW IF NOT EXISTS character_frequency AS
SELECT 
    s.name as source_name,
    ci.character,
    ci.category,
    COUNT(*) as count,
    AVG(ci.width) as avg_width,
    AVG(ci.height) as avg_height,
    AVG(ci.confidence) as avg_confidence
FROM character_instances ci
JOIN pages p ON ci.page_id = p.id
JOIN sources s ON p.source_id = s.id
GROUP BY s.name, ci.character, ci.category;

CREATE VIEW IF NOT EXISTS page_stats AS
SELECT 
    s.name as source_name,
    p.page_number,
    p.char_count,
    p.avg_confidence,
    p.long_s_count,
    p.ligature_count
FROM pages p
JOIN sources s ON p.source_id = s.id;


-- ============================================================================
-- SORT-LEVEL FORENSIC TABLES (Phase 2)
-- ============================================================================

-- Sort Images: Cropped character images linked to character instances
CREATE TABLE IF NOT EXISTS sort_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_instance_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,          -- Path to cropped sort image
    normalized_path TEXT,              -- Path to normalized (standard-size) version
    width INTEGER,                     -- Actual crop width in pixels
    height INTEGER,                    -- Actual crop height in pixels
    scale_factor REAL DEFAULT 1.0,     -- Scale at which image was extracted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (character_instance_id) REFERENCES character_instances(id)
);

-- Sort Fingerprints: Computed feature vectors for each sort image
CREATE TABLE IF NOT EXISTS sort_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sort_image_id INTEGER NOT NULL,
    hu_moments TEXT,                   -- JSON array of 7 Hu invariant moments
    contour_descriptor TEXT,           -- JSON array of contour Fourier descriptors
    edge_density REAL,                 -- Canny edge pixel ratio
    ink_density REAL,                  -- Black pixel ratio in binarized image
    texture_lbp TEXT,                  -- JSON array of Local Binary Pattern histogram
    perceptual_hash TEXT,              -- Perceptual hash string for fast matching
    feature_vector TEXT,               -- JSON array of full concatenated feature vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sort_image_id) REFERENCES sort_images(id)
);

-- Sort Matches: Pairs of sorts matched or unmatched across copies
CREATE TABLE IF NOT EXISTS sort_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sort_image_id_1 INTEGER NOT NULL,
    sort_image_id_2 INTEGER NOT NULL,
    similarity_score REAL NOT NULL,    -- 0.0 to 1.0, higher = more similar
    match_type TEXT NOT NULL,          -- 'same_sort', 'same_typeface', 'different'
    confidence REAL DEFAULT 0.0,       -- Confidence in the match classification
    verified_by_human BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sort_image_id_1) REFERENCES sort_images(id),
    FOREIGN KEY (sort_image_id_2) REFERENCES sort_images(id)
);

-- Indexes for sort-level tables
CREATE INDEX IF NOT EXISTS idx_sort_images_instance ON sort_images(character_instance_id);
CREATE INDEX IF NOT EXISTS idx_sort_fingerprints_image ON sort_fingerprints(sort_image_id);
CREATE INDEX IF NOT EXISTS idx_sort_matches_pair ON sort_matches(sort_image_id_1, sort_image_id_2);
CREATE INDEX IF NOT EXISTS idx_sort_matches_type ON sort_matches(match_type);
CREATE INDEX IF NOT EXISTS idx_sort_matches_score ON sort_matches(similarity_score);
