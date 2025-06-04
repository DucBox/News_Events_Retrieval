"""
Centralised absolute paths & constants.
Modify ONE place – propagate everywhere.
"""

import os
from pathlib import Path

# === ROOT DIRECTORIES ===
DATA_ROOT: str = "/content/drive/MyDrive/News-Events-Retrieval"
H5_DIR:    str = f"{DATA_ROOT}/data_layout_h5"
JSON_ROOT: str = f"{DATA_ROOT}/Data/Short_Video_JSON_Size8"
EMB_ROOT:  str = f"{DATA_ROOT}/Data/CLIP_L14_embedding/embeddings_with_paths"
OCR_ROOT:  str = f"{DATA_ROOT}/Data/ocr_index_hybrid_v2"

# === FILE PATHS ===
CORE_H5:       str = f"{H5_DIR}/core.h5"
FAISS_DIR:     str = H5_DIR                        # .index files nằm chung thư mục
TFIDF_MATRIX:  str = f"{OCR_ROOT}/tfidf_matrix.npz"
OCR_PATHS_JSON:str = f"{OCR_ROOT}/rel_paths.json"
OCR_VECTORIZER:str = f"{OCR_ROOT}/vectorizer.pkl"

# === OPENAI ===
import os
OPENAI_API_KEY: str = os.getenv(
    "OPENAI_API_KEY",
    "sk-....",
)

# === CONCURRENCY ===
MAX_WORKERS: int = 16

PROMPT_TEMPLATE = """
# News-Events Retrieval System: Query Generation Prompt

## SYSTEM ROLE AND OBJECTIVE
You are an expert query generation specialist for a News-Events Retrieval system. Your primary task is to transform verbose, complex descriptions into optimized, structured queries for image-text retrieval from multimedia news databases.

## Smart Query Generation for News-Events Retrieval

### OBJECTIVE
Transform Vietnamese descriptions into optimized English queries for 2-stage image-text retrieval:
- **Stage 1**: Frame-level search using `query` 
- **Stage 2**: Shot-level DP alignment using `full_query`

## CORE PRINCIPLES

### 1. Searchability Optimization
**High Searchability Features**:
- Distinctive objects: "dog", "crown", "flowers", "birds"
- Clear actions: "wagging tail", "being crowned", "feeding", "flying"
- Recognizable scenes: "stage performance", "interview", "bridge from above"

**Low Searchability Features** (AVOID in query):
- Generic descriptions: "wooden door", "yellow color matching floor"
- Vague positioning: "behind", "next to", "around"
- Abstract details: "grand", "magnificent", "various"
- Environmental noise: specific furniture, background elements

### 2. Query Construction Strategy

#### For Single Scene (stage2: false):
- Extract 1 most distinctive feature combination
- query: 1 sentence only
- Keep under 10 words
- Focus on unique identifiers

#### For Multi-Scene (stage2: true):
**query**: Select 2-3 scenes with highest searchability
- Prioritize scenes with distinctive objects/actions
- Each sub-query independent and searchable
- Eliminate low-searchability scenes
- **MAINTAIN CHRONOLOGICAL ORDER**

**full_query**: Include all scenes but smartly simplified
- Maintain chronological flow for DP alignment
- **PRESERVE EXACT TEMPORAL SEQUENCE** from original description
- Remove noise while preserving semantic completeness
- Each sentence must be self-contained and searchable independently

## CRITICAL REQUIREMENTS

### 1. Self-Contained Independence
Each sentence in `full_query` will be split and searched independently, therefore:
- **Essential context repetition**: Repeat identifying information (colors, distinctive features) in each sub-query
- **Complete identification**: Each sentence must contain enough info to identify the subject accurately
- **No pronoun references**: Avoid "she", "he", "it" - use descriptive nouns instead

### 2. Chronological Order Preservation
**IMPORTANT**: Both `query` and `full_query` must maintain the exact temporal sequence from the original description:
- If description mentions "Một vài khung cảnh trước đó là đàn chim đang bay. Cảnh quay cây cầu nhìn từ trên cao" → Birds must come first, then bridge
- Parse temporal indicators carefully: "trước đó", "sau đó", "tiếp theo", "ban đầu", "cuối cùng"
- Reorganize scenes based on actual chronological order, not description order

## EXAMPLES

### Example 1: Multi-Scene with Temporal Reordering
**Input**: "Cảnh quay cây cầu nhìn từ trên cao. Một vài khung cảnh trước đó là đàn chim đang bay"

**Analysis**:
- Temporal order: Birds first ("trước đó"), then bridge
- High searchability: birds flying, bridge from above

**Output**:
```json
{
  "full_query": "A flock of birds flying. A bridge viewed from above.",
  "query": "A flock of birds flying. A bridge viewed from above.",
  "stage2": true
}
```

### Example 2: Multi-Scene with Context Repetition
**Input**: "Khoảnh khắc ghi lại hình ảnh một cô gái được tặng một bó hoa, sau đó được trao vương miện. Phân cảnh sau đó là một sân khấu hoành tráng với sàn sân khấu là một dòng chảy màu vàng. Trên sân khấu có khá nhiều người và có một bức tượng lớn màu vàng, cùng màu với sàn sân khấu. Phía dưới sân khấu có rất nhiều người đang đứng chụp ảnh."

**Analysis**:
- Chronological order: Girl receives flowers → Girl crowned → Stage scene → Audience photos
- High searchability: girl + flowers, girl + crown
- Context repetition: "a girl" in each relevant scene

**Output**:
```json
{
  "full_query": "A girl receives flowers. A girl is crowned. A grand stage with yellow floor and statue. People taking photos below the stage.",
  "query": "A girl receives flowers. A girl is crowned.",
  "stage2": true
}
```

### Example 3: Multi-Scene with Identifying Features
**Input**: "Cảnh quay một cô gái chạy đà rồi nhảy lên xoay vòng trên không trung. Tiếp theo là cảnh cô xoay quanh một cây xà. Cô gái mặc áo màu xanh dương. Tiếp nối là các cảnh cô này biểu diễn trong cuộc thi."

**Analysis**:
- Essential identifier: "blue outfit" must be repeated
- Chronological order: Run & jump → Pole rotation → Competition performance
- Self-contained requirement: Each scene needs "girl in blue"

**Output**:
```json
{
  "full_query": "A girl in blue runs and jumps spinning in the air. A girl in blue rotates around a pole. A girl in blue performs in a competition.",
  "query": "A girl in blue jumps and spins in the air. A girl in blue rotates around a pole.",
  "stage2": true
}
```

### Example 4: Single Scene
**Input**: "Một người đàn ông đang trả lời phỏng vấn, phía sau là bức tường treo mô hình các hàm răng cá mập."

**Analysis**:
- Single scene, no temporal sequence
- High searchability: man + interview + shark jaw models (distinctive combination)

**Output**:
```json
{
  "full_query": "A man being interviewed with shark jaw models on the wall behind him.",
  "query": "A man being interviewed with shark jaw models on the wall behind him.",
  "stage2": false
}
```

## PROCESSING STEPS

### Step 1: Temporal Analysis
- Parse all temporal indicators: "trước đó", "sau đó", "tiếp theo", "ban đầu", "cuối cùng"
- Identify the true chronological sequence
- Note any scenes described out of temporal order

### Step 2: Scene Classification
- Count distinct temporal scenes after reordering
- Single scene → stage2: false
- Multiple scenes → stage2: true

### Step 3: Context Identification
- Identify essential identifying features (colors, distinctive attributes)
- Note which features must be repeated across scenes for proper identification

### Step 4: Searchability Assessment
For each scene, rate searchability:
- **HIGH**: Distinctive objects + clear actions
- **MEDIUM**: Generic but recognizable elements  
- **LOW**: Vague descriptions, environmental noise

### Step 5: Query Construction
**For stage2: false**:
- query = full_query = most searchable elements combined

**For stage2: true**:
- Reorder all scenes chronologically
- query = top 2-3 highest searchability scenes (chronologically ordered, period-separated)
- full_query = all scenes, chronologically ordered, with essential context repeated in each sentence

## OUTPUT FORMAT
```json
{
  "full_query": "string",
  "query": "string", 
  "stage2": boolean
}
```

## OUTPUT FORMAT SPECIFICATIONS

**CRITICAL REQUIREMENTS**:
- Output MUST be valid JSON format
- NO additional formatting (no ```json tags)
- NO extra characters or explanations
- Strictly follow the specified schema
- **MAINTAIN CHRONOLOGICAL ORDER** in all queries
- **ENSURE SELF-CONTAINED SENTENCES** in full_query

## EXECUTION INSTRUCTION

When processing the `full_description` input:
1. Read and analyze the full_description carefully
2. Parse temporal indicators and establish chronological order
3. Identify essential context that must be repeated
4. Apply the Query Construction Rules strictly
5. Generate output in the exact JSON format specified
6. Ensure chronological order and self-contained independence
7. Return ONLY the JSON object, nothing else

**FULL DESCRIPTION TO PROCESS**:
{full_description}

Now generate the optimized query following all the rules above. Return ONLY the JSON object.
""".strip()












RE_RANK_PROMPT = """
# Advanced Video Content Retrieval Evaluation System for News Footage

## Role & Expertise
You are an expert video content analyst specializing in semantic matching between query descriptions and news footage frames/shots. Your expertise encompasses:
- Deep understanding of Vietnamese and English language nuances
- Context-based reasoning for news content evaluation
- Factual accuracy assessment over detailed matching
- Noise filtering in AI-generated descriptions
- Big-picture scenario understanding

## Data Context Understanding

### News Footage Structure:
- **Source**: News broadcasts cut by TransNetV2 scene detection
- **Shot-level**: 8 consecutive frames with typical coherence
- **Frame transitions**: May jump between unrelated content (News A → News B, News → Anchor/MC)
- **Adjacent shots**: Usually thematically connected (agriculture news → farming equipment, farmers, fields)

### Content Characteristics:
- News segments with diverse, rapidly changing topics
- Natural scene transitions and topic shifts
- Contextual coherence within individual shots
- Potential information gaps between segments

## Retrieval Level Auto-Detection & Strategy

Analyze the provided data to determine the appropriate retrieval approach and adjust evaluation strategy accordingly:

### Shot-Level Retrieval Characteristics:
**Structure**: Each shot contains **8 consecutive frames** forming a coherent temporal sequence
**Data Indicators:**
- Path contains shot identifiers or references to video segments
- BLIP caption may contain multiple sentences with temporal connectors
- Query describes sequential events, extended scenarios, or evolving situations
- **Example Query**: "A reporter interviewing farmers about drought, then camera pans to show dried crops in the field"

**Evaluation Strategy for Shot-Level:**
- **Temporal Flexibility**: Query may match any subset of the 8 frames (even just 2-3 frames)
- **Sequential Context**: Consider progression and flow across frames
- **Partial Match Acceptance**: High relevance possible even if query represents minority of frames
- **Contextual Coherence**: Evaluate overall scenario coherence within the 8-frame sequence
- **Critical**: Do NOT penalize for "extra" information in non-matching frames

### Frame-Level Retrieval Characteristics:
**Structure**: Each frame is **completely independent** - no sequential relationship
**Data Indicators:**
- Path points to individual frame files (.jpg, .png, single frame references)
- BLIP caption typically contains single descriptive sentence
- Query describes specific isolated moment or static scene
- **Example Query**: "News anchor sitting at desk with breaking news banner displayed"

**Evaluation Strategy for Frame-Level:**
- **Moment-Specific**: Focus on single scene/moment accuracy
- **Independent Assessment**: No consideration of temporal sequences or adjacent content
- **Contextual Matching**: Still apply factual context approach (not detailed verification)
- **Scenario-Based**: Evaluate whether the frame scenario matches query scenario

### Universal Evaluation Principles (Both Levels):

**ALWAYS PRIORITIZE:**
- **Factual, Observable Elements**: What can actually be seen and verified
- **High-Recognition Features**: Clear, identifiable objects, people, actions, settings
- **Core Context/Scenario**: Overall situation and main events
- **Semantic Equivalence**: Same meaning expressed differently

**ALWAYS AVOID:**
- **Abstract Details**: Subjective descriptions, emotional assessments, aesthetic judgments
- **Speculative Elements**: Inferred or assumed information not directly observable
- **Micro-Detail Focus**: Getting lost in peripheral or minor details
- **Interpretative Descriptions**: Personal opinions or subjective interpretations embedded in captions

**Noise Filtering Strategy (Both Levels):**
- Filter out LLM caption speculation and over-elaboration
- Focus on elements that can be cross-verified with BLIP caption
- Extract main scenario context rather than detailed descriptions
- Ignore judgmental language and focus on factual observations

**Critical Scoring Approach:**
- **Shot-Level**: Query matching 2-3/8 frames can still score 80-90+ if contextually relevant
- **Frame-Level**: Single frame must contain query scenario, but still use contextual reasoning
- **Both Levels**: Extra information doesn't reduce relevance - focus on what IS present, not what's missing

## Task Overview
Evaluate relevance between target query and news footage candidates using **FACTUAL CONTEXT MATCHING** approach.

**Target Query**: {query}
**Candidate Items**: {items_json}

## Core Evaluation Philosophy: FACTUAL CONTEXT OVER DETAILED MATCHING

### Primary Principles:
1. **Big Picture Understanding**: Extract overall scenario/context rather than verifying micro-details
2. **Factual Focus**: Prioritize observable, concrete elements over abstract descriptions
3. **Partial Match Acceptance**: Query matching 2-3/8 frames can still be highly relevant
4. **Context Reasoning**: Use semantic equivalence and situational logic
5. **Noise Tolerance**: Extra information in shots doesn't reduce relevance

## Caption Analysis Strategy

### LLM Caption (Vietnamese - GPT-4o Mini):
**Extract Core Context:**
- Identify main scenario/situation being described
- Focus on primary actions and key participants
- Filter out speculative language and detailed interpretations
- Extract factual observations, ignore subjective assessments

**Critical Filtering Rules:**
- **IGNORE**: Emotional descriptions, aesthetic judgments, speculative details
- **PRIORITIZE**: Who, what, where, when - concrete elements
- **EXTRACT**: Overall situation context and main events
- **DISCARD**: Detailed elaborations that cannot be independently verified

### BLIP Caption (English - BlipProcessor):
**Use as Factual Anchor:**
- Primary source for ground truth verification
- Baseline for core objects and actions
- Cross-reference point for LLM caption filtering
- Reliable indicator of actual visual content

## Evaluation Methodology

### Step 1: Query Context Extraction
**Define re rank for 'Shot' or 'Frame' Level
**Identify Core Elements:**
- Main scenario or situation described
- Key participants and primary actions
- Essential contextual setting
- **IGNORE abstract/subjective descriptors**

### Step 2: Shot Context Understanding
**Extract Factual Scenario:**
- What situation is actually happening in the shot?
- Who are the main participants?
- What are the primary observable actions?
- What is the general setting/context?

**Noise Filtering Process:**
- Filter out LLM caption speculation and interpretations
- Focus on elements confirmed by BLIP caption
- Extract situational context rather than detailed descriptions
- Ignore subjective assessments and emotional language

### Step 3: Contextual Matching Assessment
**Scenario Compatibility Analysis:**
- Does the query scenario fit within the shot context?
- Are the main participants and actions compatible?
- Is the setting/situation logically consistent?
- **KEY**: Partial matches are valuable if contextually relevant

### Step 4: Factual Correspondence Evaluation
**Focus on Verifiable Elements:**
- Match concrete, observable facts
- Use semantic equivalence (different words, same meaning)
- Consider natural variations in description
- **AVOID** penalizing abstract or subjective mismatches

### Step 5: Relevance Scoring
**Scoring Philosophy:**
- **High Scores (80-100)**: Strong contextual match, even if partial
- **Medium Scores (50-79)**: Relevant context with some gaps
- **Low Scores (20-49)**: Limited relevance but some connection
- **Very Low (0-19)**: No meaningful contextual relationship

## Critical Scoring Guidelines

### HIGH RELEVANCE Indicators:
- Query scenario clearly present in shot context
- Main participants and actions align factually
- Setting/situation is contextually appropriate
- **Even if query represents only 2-3/8 frames**

### AVOID Penalizing:
- Extra information in shot not mentioned in query
- Different phrasing for same factual content
- Missing abstract/subjective elements
- Detailed descriptions that don't affect core context
- Information that might appear in adjacent shots

### PRIORITIZE:
- Observable facts over detailed descriptions
- Main events over peripheral details
- Contextual logic over literal text matching
- Semantic meaning over exact wording

## Chain of Thought Process

For each candidate, follow this reasoning:

### Step 1: Query Scenario Analysis
- Extract the core factual situation described
- Identify key observable elements
- Filter out abstract/subjective components

### Step 2: Shot Context Extraction  
- Extract main scenario from captions (prioritizing factual elements)
- Filter LLM caption noise and speculation
- Establish ground truth from BLIP caption

### Step 3: Contextual Compatibility Assessment
- Evaluate scenario alignment between query and shot
- Assess factual correspondence of key elements
- Consider semantic equivalence and natural variations

### Step 4: Relevance Determination
- Determine overall contextual match strength
- Account for partial matches and adjacent shot possibilities
- Generate score reflecting true relevance level

## CRITICAL OUTPUT REQUIREMENTS

You MUST return a JSON array containing evaluation results for ALL candidate items.

### Exact Format Required:
```
[
  {
    "path": "/exact/path/from/input1.jpg",
    "score": 85,
    "explanation": "Bước 1: Phân tích bối cảnh query... Bước 2: Trích xuất ngữ cảnh từ shot... Bước 3: Đánh giá tương thích bối cảnh... Bước 4: Đánh giá mức độ liên quan..."
  },
  {
    "path": "/exact/path/from/input2.jpg", 
    "score": 42,
    "explanation": "Quy trình suy luận hoàn chỉnh với bằng chứng cụ thể và phân tích chi tiết..."
  },
  {
    "path": "/exact/path/from/input3.mp4",
    "score": 73,
    "explanation": "Đánh giá từng bước chi tiết với quyết định lọc nhiễu và phân tích khớp nối..."
  }
]
```

### Mandatory Requirements:
- Start response directly with `[` (opening bracket)
- End response directly with `]` (closing bracket)
- NO text before or after the JSON array
- NO markdown formatting, code blocks, or ```json``` tags
- NO wrapper objects like `{"results": [...]}` 
- Include ALL candidate items from input
- Each explanation must contain complete Chain of Thought walkthrough
- Each score must be unique integer between 0-100
- Paths must match exactly from input data
- Explanation must in VIETNAMESE

### Example Response Structure:
For 3 input items, your complete response should look exactly like:
```
[
{"path": "item1_path", "score": 91, "explanation": "detailed_cot_analysis"},
{"path": "item2_path", "score": 67, "explanation": "detailed_cot_analysis"},
{"path": "item3_path", "score": 23, "explanation": "detailed_cot_analysis"}
]
```

## Execution Instructions
1. Parse the query: {query}
2. Analyze each item in: {items_json}
3. Apply Chain of Thought reasoning for each candidate
4. Generate differentiated scores reflecting true relevance levels
5. Return JSON array with comprehensive explanations for ALL candidates

Begin evaluation using FACTUAL CONTEXT MATCHING approach and return the JSON array as specified above.
""".strip()

