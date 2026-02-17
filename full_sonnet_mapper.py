#!/usr/bin/env python3
"""
Full 154 Sonnet Mapper
======================
Creates a complete mapping of all 154 Sonnets between Wright and Aspley editions.
Uses canonical first lines to identify each Sonnet regardless of page number.

Usage:
    python3 full_sonnet_mapper.py
"""

import logging
import json
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from character_database import CharacterDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# All 154 Sonnet first lines (modern spelling, will fuzzy match 1609 spelling)
SONNET_FIRST_LINES = {
    1: "From fairest creatures we desire increase",
    2: "When forty winters shall besiege thy brow",
    3: "Look in thy glass and tell the face thou viewest",
    4: "Unthrifty loveliness why dost thou spend",
    5: "Those hours that with gentle work did frame",
    6: "Then let not winters ragged hand deface",
    7: "Lo in the orient when the gracious light",
    8: "Music to hear why hearst thou music sadly",
    9: "Is it for fear to wet a widows eye",
    10: "For shame deny that thou bearst love to any",
    11: "As fast as thou shalt wane so fast thou growst",
    12: "When I do count the clock that tells the time",
    13: "O that you were your self but love you are",
    14: "Not from the stars do I my judgement pluck",
    15: "When I consider every thing that grows",
    16: "But wherefore do not you a mightier way",
    17: "Who will believe my verse in time to come",
    18: "Shall I compare thee to a summers day",
    19: "Devouring Time blunt thou the lions paws",
    20: "A womans face with natures own hand painted",
    21: "So is it not with me as with that Muse",
    22: "My glass shall not persuade me I am old",
    23: "As an unperfect actor on the stage",
    24: "Mine eye hath played the painter and hath steeled",
    25: "Let those who are in favour with their stars",
    26: "Lord of my love to whom in vassalage",
    27: "Weary with toil I haste me to my bed",
    28: "How can I then return in happy plight",
    29: "When in disgrace with fortune and mens eyes",
    30: "When to the sessions of sweet silent thought",
    31: "Thy bosom is endeared with all hearts",
    32: "If thou survive my well contented day",
    33: "Full many a glorious morning have I seen",
    34: "Why didst thou promise such a beauteous day",
    35: "No more be grieved at that which thou hast done",
    36: "Let me confess that we two must be twain",
    37: "As a decrepit father takes delight",
    38: "How can my Muse want subject to invent",
    39: "O how thy worth with manners may I sing",
    40: "Take all my loves my love yea take them all",
    41: "Those petty wrongs that liberty commits",
    42: "That thou hast her it is not all my grief",
    43: "When most I wink then do mine eyes best see",
    44: "If the dull substance of my flesh were thought",
    45: "The other two slight air and purging fire",
    46: "Mine eye and heart are at a mortal war",
    47: "Betwixt mine eye and heart a league is took",
    48: "How careful was I when I took my way",
    49: "Against that time if ever that time come",
    50: "How heavy do I journey on the way",
    51: "Thus can my love excuse the slow offence",
    52: "So am I as the rich whose blessed key",
    53: "What is your substance whereof are you made",
    54: "O how much more doth beauty beauteous seem",
    55: "Not marble nor the gilded monuments",
    56: "Sweet love renew thy force be it not said",
    57: "Being your slave what should I do but tend",
    58: "That god forbid that made me first your slave",
    59: "If there be nothing new but that which is",
    60: "Like as the waves make towards the pebbled shore",
    61: "Is it thy will thy image should keep open",
    62: "Sin of self love possesseth all mine eye",
    63: "Against my love shall be as I am now",
    64: "When I have seen by times fell hand defaced",
    65: "Since brass nor stone nor earth nor boundless sea",
    66: "Tired with all these for restful death I cry",
    67: "Ah wherefore with infection should he live",
    68: "Thus is his cheek the map of days outworn",
    69: "Those parts of thee that the worlds eye doth view",
    70: "That thou art blamed shall not be thy defect",
    71: "No longer mourn for me when I am dead",
    72: "O lest the world should task you to recite",
    73: "That time of year thou mayst in me behold",
    74: "But be contented when that fell arrest",
    75: "So are you to my thoughts as food to life",
    76: "Why is my verse so barren of new pride",
    77: "Thy glass will show thee how thy beauties wear",
    78: "So oft have I invoked thee for my Muse",
    79: "Whilst I alone did call upon thy aid",
    80: "O how I faint when I of you do write",
    81: "Or I shall live your epitaph to make",
    82: "I grant thou wert not married to my Muse",
    83: "I never saw that you did painting need",
    84: "Who is it that says most which can say more",
    85: "My tongue tied Muse in manners holds her still",
    86: "Was it the proud full sail of his great verse",
    87: "Farewell thou art too dear for my possessing",
    88: "When thou shalt be disposed to set me light",
    89: "Say that thou didst forsake me for some fault",
    90: "Then hate me when thou wilt if ever now",
    91: "Some glory in their birth some in their skill",
    92: "But do thy worst to steal thy self away",
    93: "So shall I live supposing thou art true",
    94: "They that have power to hurt and will do none",
    95: "How sweet and lovely dost thou make the shame",
    96: "Some say thy fault is youth some wantonness",
    97: "How like a winter hath my absence been",
    98: "From you have I been absent in the spring",
    99: "The forward violet thus did I chide",
    100: "Where art thou Muse that thou forgetst so long",
    101: "O truant Muse what shall be thy amends",
    102: "My love is strengthened though more weak in seeming",
    103: "Alack what poverty my Muse brings forth",
    104: "To me fair friend you never can be old",
    105: "Let not my love be called idolatry",
    106: "When in the chronicle of wasted time",
    107: "Not mine own fears nor the prophetic soul",
    108: "Whats in the brain that ink may character",
    109: "O never say that I was false of heart",
    110: "Alas tis true I have gone here and there",
    111: "O for my sake do you with Fortune chide",
    112: "Your love and pity doth the impression fill",
    113: "Since I left you mine eye is in my mind",
    114: "Or whether doth my mind being crowned with you",
    115: "Those lines that I before have writ do lie",
    116: "Let me not to the marriage of true minds",
    117: "Accuse me thus that I have scanted all",
    118: "Like as to make our appetites more keen",
    119: "What potions have I drunk of Siren tears",
    120: "That you were once unkind befriends me now",
    121: "Tis better to be vile than vile esteemed",
    122: "Thy gift thy tables are within my brain",
    123: "No Time thou shalt not boast that I do change",
    124: "If my dear love were but the child of state",
    125: "Wert thou obeyed by the commanding boy",
    126: "O thou my lovely boy who in thy power",
    127: "In the old age black was not counted fair",
    128: "How oft when thou my music music playst",
    129: "The expense of spirit in a waste of shame",
    130: "My mistress eyes are nothing like the sun",
    131: "Thou art as tyrannous so as thou art",
    132: "Thine eyes I love and they as pitying me",
    133: "Beshrew that heart that makes my heart to groan",
    134: "So now I have confessed that he is thine",
    135: "Whoever hath her wish thou hast thy Will",
    136: "If thy soul check thee that I come so near",
    137: "Thou blind fool Love what dost thou to mine eyes",
    138: "When my love swears that she is made of truth",
    139: "O call not me to justify the wrong",
    140: "Be wise as thou art cruel do not press",
    141: "In faith I do not love thee with mine eyes",
    142: "Love is my sin and thy dear virtue hate",
    143: "Lo as a careful housewife runs to catch",
    144: "Two loves I have of comfort and despair",
    145: "Those lips that Loves own hand did make",
    146: "Poor soul the centre of my sinful earth",
    147: "My love is as a fever longing still",
    148: "O me what eyes hath Love put in my head",
    149: "Canst thou O cruel say I love thee not",
    150: "O from what power hast thou this powerful might",
    151: "Love is too young to know what conscience is",
    152: "In loving thee thou knowst I am forsworn",
    153: "Cupid laid by his brand and fell asleep",
    154: "The little Love god lying once asleep",
}


def normalize_text(text: str) -> str:
    """Normalize for matching."""
    text = text.replace('ſ', 's').replace('ß', 's')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.lower().split())


def similarity(a: str, b: str) -> float:
    """Calculate similarity."""
    return SequenceMatcher(None, a, b).ratio()


class FullSonnetMapper:
    """Map all 154 Sonnets between editions."""
    
    def __init__(self, db: CharacterDatabase):
        self.db = db
    
    def extract_all_lines(self, edition: str) -> list:
        """Extract all text lines from an edition."""
        pages = self.db.get_page_numbers(edition)
        all_lines = []
        
        for page in pages:
            chars = self.db.get_characters_for_page(edition, page)
            if not chars:
                continue
            
            lines = defaultdict(list)
            for c in chars:
                line_y = c.y // 25
                lines[line_y].append(c)
            
            for y_key in sorted(lines.keys()):
                line_chars = sorted(lines[y_key], key=lambda x: x.x)
                line_text = ''.join(c.character for c in line_chars)
                avg_y = sum(c.y for c in line_chars) / len(line_chars)
                all_lines.append({
                    'page': page,
                    'y': avg_y,
                    'text': line_text,
                    'char_count': len(line_chars)
                })
        
        return all_lines
    
    def find_all_sonnets(self, edition: str) -> dict:
        """Find all 154 Sonnets in an edition."""
        lines = self.extract_all_lines(edition)
        found = {}
        
        for sonnet_num, first_line in SONNET_FIRST_LINES.items():
            normalized_target = normalize_text(first_line)
            first_words = normalized_target.split()[:5]  # First 5 words
            search_phrase = ' '.join(first_words)
            
            best_match = None
            best_score = 0
            
            for line_data in lines:
                text = line_data['text']
                if len(text) < 15:
                    continue
                
                normalized = normalize_text(text)
                
                # Check similarity of first 40 chars
                score = similarity(normalized[:40], search_phrase)
                
                if score > best_score and score > 0.45:  # Lower threshold
                    best_score = score
                    best_match = {
                        'page': line_data['page'],
                        'y': line_data['y'],
                        'text': text[:80],
                        'confidence': score
                    }
            
            if best_match:
                found[sonnet_num] = best_match
        
        return found


def main():
    db = CharacterDatabase("reports/characters.db")
    mapper = FullSonnetMapper(db)
    
    print("FULL 154 SONNET MAPPING")
    print("="*70)
    
    # Map both editions
    print("\nMapping Wright edition (154 Sonnets)...")
    wright_map = mapper.find_all_sonnets("wright")
    print(f"  Found: {len(wright_map)}/154 Sonnets")
    
    print("\nMapping Aspley edition (154 Sonnets)...")
    aspley_map = mapper.find_all_sonnets("aspley")
    print(f"  Found: {len(aspley_map)}/154 Sonnets")
    
    # Build comparison
    print("\n" + "="*70)
    print("SONNET PAGE MAPPING")
    print("="*70)
    print(f"{'Sonnet':<8} {'Wright':<8} {'Aspley':<8} {'Offset':<8} {'Status'}")
    print("-"*50)
    
    offsets = []
    matched = 0
    mismatched = 0
    
    for num in range(1, 155):
        w_data = wright_map.get(num, {})
        a_data = aspley_map.get(num, {})
        
        w_page = w_data.get('page', '-')
        a_page = a_data.get('page', '-')
        
        if isinstance(w_page, int) and isinstance(a_page, int):
            offset = a_page - w_page
            offsets.append(offset)
            
            if offset == 0:
                status = "✅ Same page"
                matched += 1
            else:
                status = f"⚠️ +{offset}" if offset > 0 else f"⚠️ {offset}"
                mismatched += 1
            
            offset_str = f"{offset:+d}"
        else:
            offset_str = "-"
            status = "❓ Not found"
        
        # Only print if there's a difference or not found
        if offset_str != "+0":
            print(f"{num:<8} {str(w_page):<8} {str(a_page):<8} {offset_str:<8} {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Sonnets found in Wright: {len(wright_map)}/154")
    print(f"Sonnets found in Aspley: {len(aspley_map)}/154")
    print(f"Sonnets on SAME page: {matched}")
    print(f"Sonnets on DIFFERENT pages: {mismatched}")
    
    if offsets:
        from collections import Counter
        offset_counts = Counter(offsets)
        print(f"\nPage offset distribution:")
        for offset, count in sorted(offset_counts.items()):
            print(f"  Offset {offset:+2d}: {count} Sonnets")
    
    # Save full mapping
    output = {
        'wright': {str(k): {'page': v['page'], 'confidence': v['confidence'], 'y': v.get('y', 0), 'text_sample': v.get('text', '')} 
                   for k, v in wright_map.items()},
        'aspley': {str(k): {'page': v['page'], 'confidence': v['confidence'], 'y': v.get('y', 0), 'text_sample': v.get('text', '')} 
                   for k, v in aspley_map.items()},
        'summary': {
            'wright_found': len(wright_map),
            'aspley_found': len(aspley_map),
            'same_page': matched,
            'different_page': mismatched
        }
    }
    
    output_path = Path("reports/full_sonnet_mapping.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull mapping saved to: {output_path}")
    
    db.close()


if __name__ == "__main__":
    main()
