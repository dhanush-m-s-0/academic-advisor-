"""
🚦 Traffic Analyzer Agent
==========================
LangChain + Groq (Llama 3.3 70B) — context-aware answers for every
traffic question: congestion, route planning, incident analysis, data &
statistics, infrastructure, and environmental impact.

Architecture mirrors academic_advisor_agent.py:
  • 6 specialist roles with rich system prompts
  • Per-user memory (SimpleMemory)
  • RAG via ChromaDB (simple hash embeddings)
  • Role detection by weighted keyword scoring
  • Built-in traffic knowledge base (Indian cities focus)
  • Web scraper for ingesting external URLs
  • CLI mode for quick testing
"""

import os
import hashlib
import math
import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

GROQ_API_KEY           = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL             = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TRAFFIC_CHROMA_DB_PATH = "./traffic_chroma_db"
EMBED_DIMENSION        = 1536


# ─────────────────────────────────────────────────────────────────
# Simple Memory
# ─────────────────────────────────────────────────────────────────

class SimpleMemory:
    """In-memory conversation history (mirrors academic advisor)."""

    def __init__(self, k: int = 8):
        self.k = k
        self.messages: List = []

    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-self.k * 2:]

    def save_context(self, inputs: dict, outputs: dict):
        self.add_message(HumanMessage(content=inputs["input"]))
        self.add_message(AIMessage(content=outputs["output"]))


# ─────────────────────────────────────────────────────────────────
# Role Definitions
# ─────────────────────────────────────────────────────────────────

ROLES: Dict[str, Dict] = {

    "traffic_flow": {
        "emoji": "🚦",
        "name": "Traffic Flow Analysis",
        "color": "#e74c3c",
        "bg": "#fdecea",
        "keywords": [
            "congestion", "traffic jam", "jam", "signal", "signals",
            "peak hour", "rush hour", "bottleneck", "flow", "volume",
            "density", "speed", "average speed", "slow", "gridlock",
            "intersection", "junction", "flyover", "underpass", "overpass",
            "lane", "lanes", "clearance", "timing", "green phase",
        ],
        "system_prompt": """You are an expert Traffic Flow Analyst with deep knowledge of urban traffic systems in India and globally.

YOUR EXPERTISE:
- Traffic flow theory (LWR model, density-flow relationships, capacity analysis)
- Congestion patterns: recurring vs. non-recurring congestion causes
- Signal timing optimization: Webster's formula, actuated signals, coordinated corridors
- Level of Service (LOS) assessment: A (free flow) to F (breakdown)
- Peak hour analysis: AM/PM peaks, school peaks, weekend patterns
- Bottleneck identification: merge points, weave sections, grade issues
- Indian city traffic behavior: Bangalore ORR, Mumbai Western Express Highway, Delhi Ring Road,
  Chennai Anna Salai, Hyderabad HITEC City, Pune Kothrud, Kolkata VIP Road

RESPONSE STYLE:
- Use concrete numbers and LOS grades where possible
- Suggest signal timing improvements with specific cycle lengths
- Identify root causes before recommending solutions
- Reference real Indian traffic scenarios

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Provide a detailed, actionable analysis. End with 2 concrete next steps.""",
    },

    "route_planning": {
        "emoji": "🛣️",
        "name": "Route Planning & Optimization",
        "color": "#2ecc71",
        "bg": "#e8f8f0",
        "keywords": [
            "route", "routes", "fastest", "shortest", "navigate", "navigation",
            "direction", "directions", "eta", "commute", "commuting",
            "alternate", "alternative", "bypass", "avoid", "toll",
            "highway", "expressway", "distance", "travel time",
            "google maps", "waze", "maps", "gps", "best path",
            "multi-modal", "bus", "metro", "train", "cab", "auto",
            "how to reach", "how to go", "which way",
        ],
        "system_prompt": """You are an expert Route Planning & Transportation Advisor specializing in Indian urban mobility.

YOUR EXPERTISE:
- Dijkstra/A* based shortest path principles; real-world deviations
- Dynamic routing accounting for live traffic, incidents, time-of-day
- Multi-modal journey planning: walk + metro + bus + cab combinations
- Toll route vs. free route cost-benefit analysis
- Indian expressways: NH-48 (Delhi-Mumbai), Yamuna Expressway, Bengaluru-Mysuru Expressway, Pune Expressway
- City-level knowledge:
  * Bangalore: ORR, NICE Road, Hosur Road, Electronic City Expressway
  * Mumbai: Eastern Express Highway, JNPT Road, Bandra-Worli Sea Link
  * Delhi: DND Flyway, NH-48, Dwarka Expressway, Yamuna Expressway
  * Chennai: ECR, OMR, GST Road, Chennai Bypass
  * Hyderabad: ORR, PVNR Expressway, Nehru ORR
- ETA estimation factoring congestion, traffic signals, speed limits
- Parking availability near destinations

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Give a practical, specific routing recommendation. Include time estimates, alternatives, and mode comparison where relevant.""",
    },

    "incident_safety": {
        "emoji": "🚧",
        "name": "Incident & Safety Analysis",
        "color": "#f39c12",
        "bg": "#fef9e7",
        "keywords": [
            "accident", "crash", "collision", "safety", "hazard",
            "speed limit", "drunk driving", "helmet", "seat belt",
            "road rage", "rash driving", "dangerous", "hotspot",
            "black spot", "weather", "fog", "rain", "flood",
            "construction", "road work", "diversion", "emergency",
            "ambulance", "fire", "police", "breakdown", "pothole",
            "road condition", "visibility", "night driving",
        ],
        "system_prompt": """You are a Road Safety & Incident Analysis Expert with deep knowledge of Indian traffic law, accident patterns, and emergency response.

YOUR EXPERTISE:
- Accident hotspot identification: statistical cluster analysis, before-after studies
- Road safety engineering: sight distance, curve design, lighting, median barriers
- Indian Motor Vehicles Act (2019 amendments): fines, penalties, license rules
- Common accident causes in India: speeding, drunk driving, wrong-side driving, pothole-related
- Weather impact: monsoon flooding, winter fog (NH-44 Punjab/Haryana), summer heat mirages
- Emergency response: golden hour concept, ambulance routing, trauma center network
- Construction zone safety: work zone speed limits, signage requirements
- Black spot remediation: engineering countermeasures, enforcement strategies
- NCRB road accident statistics; state-wise safety rankings

RESPONSE STYLE:
- Quote relevant MV Act sections for legal questions
- Provide specific hotspot data for major Indian cities
- Always include preventive measures alongside analysis
- For emergencies: give immediate action steps first

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Provide safety-focused analysis with specific, actionable recommendations. Cite legal provisions where applicable.""",
    },

    "traffic_data": {
        "emoji": "📊",
        "name": "Traffic Data & Statistics",
        "color": "#3498db",
        "bg": "#eaf4fb",
        "keywords": [
            "data", "statistics", "trend", "trends", "count",
            "average speed", "historical", "analysis", "numbers",
            "percentage", "increase", "decrease", "comparison",
            "survey", "study", "report", "annual", "monthly",
            "daily", "hourly", "vehicle count", "pcu", "traffic volume",
            "flow rate", "occupancy", "headway", "gap",
        ],
        "system_prompt": """You are a Traffic Data Analyst and Transportation Researcher with expertise in quantitative traffic analysis.

YOUR EXPERTISE:
- Traffic counting methods: manual counts, pneumatic tubes, video-based, radar sensors
- Key metrics: AADT (Annual Average Daily Traffic), PHF (Peak Hour Factor), V/C ratio
- PCU (Passenger Car Unit) equivalency factors for Indian mixed traffic
- Data sources: MoRTH reports, NCRB, city traffic police annual reports, IIT traffic studies
- Statistical methods: regression analysis, time-series forecasting, spatial analysis
- Before-after studies for evaluating traffic interventions
- Big data in traffic: probe vehicle data, GPS traces, mobile phone data
- Indian traffic data context:
  * Delhi: 11 million+ registered vehicles; 2.5L accidents/year nationally
  * Bangalore: 8M+ registered vehicles; 2500+ annual road deaths
  * Mumbai: 3.5M vehicles; 450 km of roads handling 8M daily trips
- Visualization: space-time diagrams, volume-speed curves, flow maps

RESPONSE STYLE:
- Present data with proper units and context
- Use comparative benchmarks (national average, similar cities)
- Distinguish between correlation and causation
- Suggest data collection improvements when relevant

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Provide data-driven insights with specific numbers, sources, and analytical methodology.""",
    },

    "infrastructure": {
        "emoji": "🏗️",
        "name": "Infrastructure & Urban Planning",
        "color": "#9b59b6",
        "bg": "#f5eef8",
        "keywords": [
            "road", "roads", "highway", "flyover", "underpass", "bridge",
            "metro", "bus rapid transit", "brt", "parking",
            "smart city", "traffic calming", "roundabout", "rotary",
            "grade separator", "interchange", "junction improvement",
            "pedestrian", "footpath", "cycle lane", "cycling",
            "public transport", "infrastructure", "urban planning",
            "city planning", "signal coordination", "itms",
            "intelligent transport", "atcs", "adaptive signal",
            "design", "widening", "capacity expansion",
        ],
        "system_prompt": """You are an Urban Traffic & Infrastructure Planning Expert specializing in Indian cities and smart mobility solutions.

YOUR EXPERTISE:
- Road geometric design: IRC (Indian Roads Congress) standards for urban/rural roads
- Junction design: at-grade, grade-separated, roundabout, signalized
- Capacity expansion approaches: widening, grade separation, parallel roads
- Traffic calming: speed humps, chicanes, raised crossings, road narrowing
- Parking: multi-level car parks, smart parking systems, on-street management
- Public transport infrastructure: BRT corridors, metro feeder, last-mile connectivity
- Smart City traffic systems: ATCS (Adaptive Traffic Control), ITMS, surveillance
- Pedestrian and cycle infrastructure: European vs Indian standards
- Indian smart city projects: AMRUT, Smart Cities Mission success stories
- Infrastructure challenges: land acquisition, encroachment, mixed traffic
- Case studies: Pune BRT (lessons learned), Ahmedabad BRTS (success), Hyderabad ORR

RESPONSE STYLE:
- Reference IRC guidelines and MoRTH standards
- Provide cost estimates (order of magnitude) where helpful
- Compare Indian solutions with global best practices
- Highlight implementation challenges unique to India

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Provide technically sound, practically feasible infrastructure recommendations. Include implementation considerations.""",
    },

    "environmental": {
        "emoji": "🌿",
        "name": "Environmental Impact",
        "color": "#27ae60",
        "bg": "#e9f7ef",
        "keywords": [
            "pollution", "emission", "emissions", "ev", "electric vehicle",
            "electric car", "electric bike", "carbon", "carbon footprint",
            "green", "sustainable", "noise", "noise pollution",
            "air quality", "aqi", "pm2.5", "nox", "co2",
            "fuel consumption", "fuel efficiency", "hybrid",
            "charging station", "charging infrastructure",
            "environment", "climate", "clean energy", "renewable",
            "congestion pricing", "odd-even", "vehicle restriction",
        ],
        "system_prompt": """You are an Environmental Transport Analyst specializing in traffic-related pollution, green mobility, and sustainable urban transport in India.

YOUR EXPERTISE:
- Vehicle emission standards: BS6 (Bharat Stage 6), their impact on air quality
- Traffic contribution to urban air pollution: Delhi (40% PM2.5 from transport), Mumbai, Bangalore
- AQI monitoring and transport's role: CPCB data, real-time air quality stations
- EV ecosystem in India: FAME II subsidy, state policies, charging network (Tata Power, Ather, BESCOM)
- EV adoption: 2W, 3W ahead of 4W; battery swap vs. plug-in models
- Green transport modes: cycling infrastructure, e-buses (BEST, BMTC, DTC), metro
- Congestion pricing benefits: Stockholm experience; India pilot proposals
- Odd-even scheme effectiveness: Delhi experience, lessons learned
- Noise pollution from traffic: WHO limits vs Indian reality, mitigation measures
- Carbon footprint calculation: per-km emissions by mode
- Sustainable urban mobility plans (SUMP): Chennai, Pune, Kochi examples

RESPONSE STYLE:
- Use AQI scale and emission standards in context
- Compare modes by carbon emissions per passenger-km
- Highlight government schemes and subsidies available
- Be honest about trade-offs (EV battery disposal, grid carbon intensity)

USER CONTEXT:
{user_context}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{history}

User Question: {question}

Provide environmentally grounded analysis with specific data on emissions, policies, and green alternatives.""",
    },
}


# ─────────────────────────────────────────────────────────────────
# Embeddings (deterministic hash-based, same as academic advisor)
# ─────────────────────────────────────────────────────────────────

class SimpleHashEmbeddings(Embeddings):
    """Deterministic 1536-d char n-gram embeddings."""

    def __init__(self, dim: int = EMBED_DIMENSION):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        # MD5 is used here only as a fast non-cryptographic hash for embedding
        # generation — not for any security-sensitive purpose.
        text = text.lower()[:2000]
        vec = [0.0] * self.dim
        for i in range(len(text) - 2):
            h = int(hashlib.md5(text[i:i+3].encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ─────────────────────────────────────────────────────────────────
# Web Scraper
# ─────────────────────────────────────────────────────────────────

def scrape_url(url: str, category: str = "general", timeout: int = 15) -> Optional[Document]:
    headers = {"User-Agent": "Mozilla/5.0 (TrafficAnalyzer/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠  Could not fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    content_node = soup.find("article") or soup.find("main") or soup.find("body")
    raw_text = content_node.get_text(separator="\n", strip=True) if content_node else ""
    title = soup.title.string.strip() if soup.title else url

    if len(raw_text) < 50:
        return None

    print(f"  ✓  Scraped [{category}] '{title[:60]}' ({len(raw_text):,} chars)")
    return Document(
        page_content=raw_text,
        metadata={"source": url, "title": title, "category": category,
                  "scraped_at": datetime.now().isoformat()}
    )


# ─────────────────────────────────────────────────────────────────
# Role Detector
# ─────────────────────────────────────────────────────────────────

def detect_role(query: str, profile: "TrafficUserProfile") -> str:
    """Score each role via keyword matching. Returns best-fit role ID."""
    q = query.lower()
    scores: Dict[str, float] = {r: 0.0 for r in ROLES}

    for role_id, role in ROLES.items():
        for kw in role["keywords"]:
            if kw in q:
                scores[role_id] += 1 + len(kw.split()) * 0.5

    # Extra boosts for common patterns
    if any(w in q for w in ["data", "statistics", "count", "trend", "historical", "report"]):
        scores["traffic_data"] += 2
    if any(w in q for w in ["pollution", "emission", "ev", "electric", "carbon", "aqi"]):
        scores["environmental"] += 2
    if any(w in q for w in ["accident", "crash", "safety", "speed limit", "drunk"]):
        scores["incident_safety"] += 2
    if any(w in q for w in ["route", "fastest", "shortest", "eta", "commute"]):
        scores["route_planning"] += 2
    if any(w in q for w in ["metro", "brt", "parking", "smart city", "infrastructure"]):
        scores["infrastructure"] += 2

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "traffic_flow"


# ─────────────────────────────────────────────────────────────────
# User Profile
# ─────────────────────────────────────────────────────────────────

class TrafficUserProfile:
    def __init__(self, user_id: str):
        self.user_id       = user_id
        self.name          = None
        self.city          = None
        self.region        = None
        self.commute_from  = None
        self.commute_to    = None
        self.vehicle_type  = None
        self.concerns      = []
        self.session_log: List[Dict] = []

    def update_from_query(self, query: str):
        """Auto-extract profile hints from natural conversation."""
        q = query.lower()

        # City detection
        cities = [
            "bangalore", "bengaluru", "mumbai", "delhi", "chennai",
            "hyderabad", "pune", "kolkata", "ahmedabad", "jaipur",
            "surat", "lucknow", "kanpur", "nagpur", "indore",
        ]
        for city in cities:
            if city in q:
                self.city = city.capitalize()
                break

        # Vehicle type detection
        for vtype in ["bike", "car", "bus", "auto", "metro", "cycle", "truck", "scooter"]:
            if vtype in q:
                self.vehicle_type = vtype
                break

    def log(self, role: str, query: str, answer: str):
        self.session_log.append({
            "timestamp": datetime.now().strftime("%H:%M"),
            "role": role,
            "query": query,
            "answer_snippet": answer[:150] + "…" if len(answer) > 150 else answer,
        })

    def context_string(self) -> str:
        parts = []
        if self.name:          parts.append(f"Name: {self.name}")
        if self.city:          parts.append(f"City: {self.city}")
        if self.region:        parts.append(f"Region: {self.region}")
        if self.commute_from:  parts.append(f"Commute From: {self.commute_from}")
        if self.commute_to:    parts.append(f"Commute To: {self.commute_to}")
        if self.vehicle_type:  parts.append(f"Vehicle Type: {self.vehicle_type}")
        if self.concerns:      parts.append(f"Concerns: {', '.join(self.concerns)}")
        if not parts:
            return "No profile information collected yet."
        return "\n".join(parts)

    def summary(self) -> str:
        return self.context_string()


# ─────────────────────────────────────────────────────────────────
# Built-in Traffic Knowledge Base
# ─────────────────────────────────────────────────────────────────

TRAFFIC_KNOWLEDGE_BASE = {

    "traffic_flow": """
TRAFFIC FLOW FUNDAMENTALS:

Level of Service (LOS) for Urban Roads (HCM/IRC):
  LOS A — Free flow; V/C < 0.35; speed near free-flow
  LOS B — Stable flow; V/C 0.35–0.54; minor delays
  LOS C — Stable but influenced; V/C 0.55–0.77
  LOS D — Approaching unstable; V/C 0.78–0.90; noticeable delays
  LOS E — Unstable; V/C 0.91–1.00; significant queuing
  LOS F — Breakdown; V/C > 1.00; stop-and-go, gridlock

PEAK HOUR PATTERNS IN INDIAN CITIES:
Morning peak: 8:00–10:30 AM (office commuters + school traffic)
Evening peak: 5:30–8:30 PM (return commute + market traffic)
Secondary peaks: 12:30–1:30 PM near commercial areas, 10 PM near entertainment zones
Weekend patterns: Saturday similar to weekday; Sunday 40–60% lower volume

CONGESTION CAUSES IN INDIA:
1. Signal timing not optimized — fixed time plans outdated
2. Mixed traffic: 2W, 3W, cars, buses, trucks, cycles on same road
3. On-road parking blocking lanes (30–40% capacity loss)
4. U-turns at every median opening — major flow disruption
5. Footpath encroachment forcing pedestrians onto road
6. School/college zones — lack of proper drop/pick zones
7. Railway level crossings — 15–45 min closures
8. Lack of proper merge/diverge lanes at flyover ramps

SIGNAL TIMING PRINCIPLES:
Webster's optimal cycle length: Co = (1.5L + 5) / (1 - Y)
  where L = total lost time, Y = sum of critical phase volume ratios
Typical urban cycle: 90–120 seconds
Minimum green: 7 seconds (pedestrian clearance consideration)
Pedestrian crossing time = crossing distance / 1.2 m/s

BOTTLENECK IDENTIFICATION:
Step 1: Map volume counts at candidate sections
Step 2: Calculate V/C ratio for each approach
Step 3: Section with V/C > 0.85 = operational bottleneck
Step 4: Trace back queue spillback to find primary cause
""",

    "route_planning": """
ROUTE OPTIMIZATION PRINCIPLES:

ROUTING ALGORITHMS:
Dijkstra's Algorithm: Shortest path by distance; ignores dynamic conditions
A* Algorithm: Heuristic-guided; faster than Dijkstra for large networks
Contraction Hierarchies: Used by Google Maps, HERE — pre-processes network
Dynamic Programming: Real-time re-routing using live travel time updates

MULTI-MODAL ROUTE PLANNING IN INDIA:
Best for long distances:
  Metro + feeder cab: 30–50% faster than driving during peak
  Bus Rapid Transit: Competitive with cars where BRT has dedicated lane
  Suburban rail + cab: Best for Mumbai (Western Line, Harbour Line, Central Line)

CITY-SPECIFIC ROUTING KNOWLEDGE:
BANGALORE:
  - Silk Board Junction: India's worst congestion; avoid 7:30–10 AM, 5–9 PM
  - Electronic City Elevated Expressway: Toll ₹50, saves 45 min vs. Hosur Road
  - ORR (Outer Ring Road): 65 km; good evenings, bad mornings near Marathahalli
  - NICE Road: Best for trips bypassing central Bangalore; toll ₹65–130

MUMBAI:
  - Bandra-Worli Sea Link: Saves 45 min (toll ₹75 car); best for South Mumbai trips
  - Eastern Express Highway: Faster than Western for Thane/Navi Mumbai
  - SV Road vs Link Road: Link Road better evenings; SV Road better mornings

DELHI:
  - DND Flyway: Delhi–Noida; saves 20–30 min; toll ₹26
  - Yamuna Expressway: 165 km Delhi–Agra; 110 km/h design; good for long trips
  - NH-48: Delhi–Gurugram; parallel Dwarka Expressway now open as alternative

HYDERABAD:
  - ORR (Outer Ring Road): 158 km; toll-based; best for airport trips
  - PVNR Expressway: Free but congested evenings

ETA ESTIMATION FACTORS:
Free-flow speed × congestion factor × signal delay factor × route length
Congestion factor: 0.4–0.6 during peak hours on urban arterials
Average signal delay: 45–90 seconds per signalized intersection
""",

    "incident_safety": """
ROAD SAFETY IN INDIA — KEY FACTS:

NATIONAL STATISTICS (MoRTH 2023):
- Total road accidents: ~4.8 lakh per year
- Deaths: ~1.68 lakh per year (world's highest)
- Injuries: ~4.5 lakh per year
- Economic cost: ~3.14% of GDP (~₹5.96 lakh crore)

TOP CAUSES OF ACCIDENTS:
1. Over-speeding: 68% of fatal accidents
2. Driving on wrong side: 5.5%
3. Drunk driving: 3.5%
4. Red light jumping: 2.4%
5. Mobile phone use while driving: growing rapidly

HIGH-RISK TIMES:
- Night (9 PM – 6 AM): 45% of fatalities despite lower volume — poor lighting, speeding
- Festive seasons: +30–40% accident spike (Diwali, New Year's Eve)
- Monsoon: Wet roads, poor visibility; aquaplaning above 85 km/h

MOTOR VEHICLES ACT 2019 — KEY FINES:
Over-speeding: ₹1,000–2,000 (repeat: ₹2,000–4,000)
Drunk driving: ₹10,000 or 6 months imprisonment (first offense)
Jumping red light: ₹1,000–5,000
Helmet violation: ₹1,000 + 3-month license suspension
Seat belt: ₹1,000
Using mobile while driving: ₹1,500–5,000
Dangerous driving: ₹1,000–5,000 + imprisonment up to 6 months

BLACK SPOT DEFINITION AND TREATMENT:
Black spot = location with ≥5 accidents or ≥10 deaths in 3 years
Engineering treatments:
  - Skid-resistant surfacing
  - Rumble strips before hazardous curves
  - Improved lighting (sodium vapor → LED)
  - Guard rails on embankment sections
  - Advance warning signs + speed reduction
  - Pedestrian underpasses at busy crossings

ACCIDENT HOTSPOTS — MAJOR CITIES:
Bangalore: Silk Board, KR Puram junction, Marathahalli
Mumbai: Bandra-Kurla Complex, Eastern Freeway on-ramp
Delhi: NH-44 (Panipat), NH-48 (Manesar), NH-19 (Mathura)
Chennai: GST Road, OMR, Poonamallee Bypass
""",

    "traffic_data": """
TRAFFIC DATA COLLECTION & ANALYSIS:

COUNTING METHODS:
Manual Count:
  - Most accurate for classified counts (2W, car, bus, truck)
  - 16-hour counts (6 AM–10 PM) standard for urban planning
  - Cost: ₹3,000–5,000/day per location

Automatic Traffic Counter (ATC):
  - Pneumatic tubes: ±3% accuracy; 1–2 week deployments
  - Inductive loops (buried in road): permanent installations at signals
  - Video-based: AI-powered classified counting, accuracy 90–95%
  - Radar/microwave: Works in low visibility; used on NHs

AADT (Annual Average Daily Traffic):
AADT = Total annual vehicle-km / 365 × road length
Urban arterials in India:
  - Category A (national highway urban stretch): 50,000–2,00,000 AADT
  - Category B (state highway): 20,000–80,000 AADT
  - Category C (major district road): 5,000–30,000 AADT

PCU EQUIVALENCY (IRC standards for Indian mixed traffic):
Motorcycle/Scooter: 0.5 PCU
Car/Jeep/Taxi: 1.0 PCU
Auto-rickshaw: 0.6 PCU
Bus/Truck: 2.2–3.7 PCU
Cycle: 0.2 PCU
Animal Cart: 4.0 PCU

DATA SOURCES IN INDIA:
- MoRTH Annual Report: Road accidents, network statistics
- NCRB Crime Statistics: Drunk driving arrests, accident deaths
- State PWD: Road network data
- City Traffic Police: Signal timings, congestion statistics
- IIT research: Mumbai IIT, IIT Delhi, IIT Madras transport studies
- TomTom Traffic Index: City-wise congestion rankings (Bengaluru often top 5 globally)

KEY PERFORMANCE INDICATORS:
V/C Ratio = Volume / Capacity (> 0.85 = congested)
PHF = Peak Hour Volume / (4 × Peak 15-min Volume) (typical: 0.88–0.95)
Travel Time Index (TTI) = Peak travel time / Free-flow travel time
Buffer Time Index (BTI) = (95th percentile time - median time) / median time
""",

    "infrastructure": """
TRAFFIC INFRASTRUCTURE GUIDE (INDIAN STANDARDS):

IRC ROAD CLASSIFICATION:
NH (National Highway): 4–6 lane; design speed 100 km/h (rural), 80 km/h (urban)
SH (State Highway): 2–4 lane; design speed 80 km/h
MDR (Major District Road): 2 lane; design speed 65–80 km/h
ODR/VR (Village Road): 1–2 lane; 50–65 km/h

URBAN ROAD HIERARCHY (UDPFI Guidelines):
Expressway: ≥6 lane; access-controlled; no at-grade crossings
Arterial: 4–6 lane; major through movement; signals at key junctions
Sub-Arterial: 4 lane; collects from collectors, feeds arterials
Collector Street: 2–4 lane; local distribution
Local Street: 2 lane; serves individual properties

JUNCTION DESIGN OPTIONS:
At-Grade Signalized: Best for V/C < 0.8; low cost; flexible
Roundabout: Good for <20,000 PCU/day; no signal needed; self-regulating
Grade Separator: Required when V/C > 0.9; cost ₹50–300 crore
Diamond Interchange: NHs crossing urban arterials
Trumpet: NHs entering cities; one major movement

PARKING STANDARDS (SP 43):
ECS (Equivalent Car Space) requirements:
  Office: 1 ECS per 50 sqm floor area
  Retail: 1 ECS per 40 sqm
  Residential: 1.2 ECS per unit
Multi-level car park cost: ₹8–15 lakh per car space
Smart parking systems: IoT sensors + app; reduces search traffic 30%

SMART CITY TRAFFIC SYSTEMS:
ATCS (Adaptive Traffic Control System):
  - SCOOT, SCATS, InSync algorithms
  - Reduces delay by 12–15% vs. fixed time plans
  - Cost: ₹1–2 crore per intersection
ITMS (Intelligent Transport Management System):
  - Integrated CCTV, ANPR (Automatic Number Plate Recognition)
  - Variable message signs (VMS) for real-time guidance
  - Incident detection algorithms
Cities deployed: Bengaluru, Hyderabad, Delhi, Mumbai, Chennai

TRAFFIC CALMING MEASURES:
Speed humps: IRC SP 67; max 3.5 cm height; 3.5 m width
Raised crossings: Elevation = road surface; improves pedestrian safety
Chicane: Lateral displacement of traffic path; reduces speeds to 20–30 km/h
Road narrowing (choker): Effective but can block emergency vehicles
Rumble strips: Audible warning before hazardous locations
""",

    "environmental": """
TRAFFIC & ENVIRONMENT IN INDIA:

VEHICLE EMISSION STANDARDS:
BS6 (Bharat Stage 6 — April 2020):
  - 68% reduction in NOx vs BS4
  - 82% reduction in PM (particulate matter) vs BS4
  - Requires ultra-low sulfur fuel (10 ppm vs 50 ppm in BS4)
  - All new vehicles must comply; significant air quality improvement

TRANSPORT'S SHARE IN URBAN AIR POLLUTION:
Delhi: 28–40% of PM2.5 (seasonal variation; higher in winter)
Mumbai: 25–30% of PM2.5
Bangalore: 20–25% of PM2.5
Chennai: 30–35% of PM2.5
Note: Two-wheelers contribute 30–35% of transport emissions despite being cleaner individually

AQI CATEGORIES (CPCB India):
Good: 0–50
Satisfactory: 51–100
Moderate: 101–200
Poor: 201–300
Very Poor: 301–400
Severe: 401–500

EV ECOSYSTEM IN INDIA (2024):
Government schemes:
  - FAME II: ₹10,900 crore; ₹15,000 subsidy per 2W EV; ₹50,000 for 3W
  - PLI scheme for battery manufacturing
  - State subsidies: UP, Maharashtra, Tamil Nadu offer additional benefits
  Charging infrastructure:
  - 12,000+ public charging stations (PECVSL target: 5 lakh by 2030)
  - Tata Power, Ather, ChargeZone, Statiq major operators
  - Home charging (5 kW AC): ₹8–12/unit; takes 6–8 hrs for full charge
  - Fast charging (50 kW DC): ₹16–20/unit; 30–45 min to 80%

EV ADOPTION IN INDIA:
2W: 8% EV penetration (2024); target 40% by 2030
3W: 50%+ EV penetration in some cities (e-rickshaws)
4W: 2.5% EV penetration; Tata Nexon EV, MG ZS EV, BYD dominate
Buses: 4,500+ e-buses in operation (BEST Mumbai, BMTC Bangalore, DTC Delhi)

CARBON EMISSIONS BY TRANSPORT MODE (per passenger-km):
Private Car (petrol): 170–200 g CO2
Motorcycle: 90–120 g CO2
Bus (diesel): 30–50 g CO2
Metro/Railway: 10–25 g CO2 (depends on grid carbon intensity)
Cycling: 5–10 g CO2 (manufacturing amortized)
Walking: 0 g CO2

CONGESTION PRICING — GLOBAL & INDIA:
Stockholm: 22% traffic reduction; 14% emission reduction post-implementation
London: 30% traffic reduction in central zone
Singapore ERP: 15–25% traffic reduction
Delhi odd-even: Mixed results (8–10% traffic reduction; 5–8% emission reduction)
Proposed: Mumbai congestion charge (under discussion)

NOISE POLLUTION STANDARDS (CPCB):
Day (6 AM–10 PM): 65 dB (commercial), 55 dB (residential), 45 dB (silence zone)
Night (10 PM–6 AM): 55 dB (commercial), 45 dB (residential), 40 dB (silence zone)
Main highway median noise: 70–85 dB; requires noise barriers above 65 dB threshold
""",
}


# ─────────────────────────────────────────────────────────────────
# Main Agent
# ─────────────────────────────────────────────────────────────────

class TrafficAnalyzerAgent:
    """
    Traffic Analyzer with:
    - 6 specialist roles (Flow, Route, Safety, Data, Infrastructure, Environment)
    - Groq (Llama 3.3 70B) for LLM inference
    - ChromaDB for RAG
    - Per-user conversation memory and profile
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        print("\n🚦 Initialising Traffic Analyzer Agent …")

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.4,
            max_tokens=2048,
        )

        self.embeddings = SimpleHashEmbeddings(EMBED_DIMENSION)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.vectorstore = Chroma(
            persist_directory=TRAFFIC_CHROMA_DB_PATH,
            embedding_function=self.embeddings,
        )

        self.profiles: Dict[str, TrafficUserProfile] = {}
        self.memories: Dict[str, SimpleMemory] = {}

        print("✅ Traffic Analyzer ready!\n")
        for r in ROLES.values():
            print(f"    {r['emoji']}  {r['name']}")
        print()

    # ── Profile helpers ──────────────────────────────────────────

    def _ensure_profile(self, user_id: str) -> TrafficUserProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = TrafficUserProfile(user_id)
        return self.profiles[user_id]

    # ── Knowledge ingestion ──────────────────────────────────────

    def ingest_urls(self, urls: List[str], category: str = "general") -> int:
        print(f"\n📥 Ingesting {len(urls)} URL(s) [{category}] …")
        docs = []
        for url in urls:
            doc = scrape_url(url, category=category)
            if doc:
                docs.append(doc)
            time.sleep(0.5)
        if not docs:
            return 0
        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        print(f"  📌 Stored {len(chunks)} chunks.\n")
        return len(chunks)

    def ingest_text(self, text: str, title: str, category: str = "general") -> int:
        doc = Document(
            page_content=text,
            metadata={"source": "manual", "title": title, "category": category}
        )
        chunks = self.splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        print(f"  📌 Ingested '{title}' → {len(chunks)} chunks [{category}]")
        return len(chunks)

    # ── Core analyse method ──────────────────────────────────────

    def analyse(self, query: str, user_id: str = "guest",
                force_role: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point: detect role → RAG context →
        build rich prompt → Groq generates answer.
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = TrafficUserProfile(user_id)
        if user_id not in self.memories:
            self.memories[user_id] = SimpleMemory(k=8)

        profile = self.profiles[user_id]
        memory  = self.memories[user_id]

        profile.update_from_query(query)

        role_id = (force_role if force_role and force_role in ROLES
                   else detect_role(query, profile))
        role = ROLES[role_id]

        rag_context = self._retrieve_context(query)
        history     = self._format_history(memory)

        system_prompt = role["system_prompt"].format(
            user_context=profile.context_string(),
            rag_context=rag_context,
            history=history,
            question=query,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        answer = self._call_llm(messages)

        memory.save_context({"input": query}, {"output": answer})
        profile.log(role_id, query, answer)

        sources = self._get_sources(query)

        return {
            "answer"     : answer,
            "role"       : role["name"],
            "role_emoji" : role["emoji"],
            "role_id"    : role_id,
            "role_color" : role["color"],
            "sources"    : sources,
            "user_id"    : user_id,
            "timestamp"  : datetime.now().strftime("%H:%M"),
        }

    def _call_llm(self, messages: List[Any]) -> str:
        response = self.llm.invoke(messages)
        return response.content

    def _retrieve_context(self, query: str, k: int = 5) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No specific knowledge base context found — answer from expert knowledge."
            return "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('title', 'Unknown')}]\n{d.page_content}"
                for d in docs
            )
        except Exception:
            return "Knowledge base unavailable — answering from expert knowledge."

    def _get_sources(self, query: str) -> List[str]:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return list({d.metadata.get("source", "internal") for d in docs})
        except Exception:
            return []

    def _format_history(self, memory: SimpleMemory) -> str:
        try:
            msgs = memory.messages
            if not msgs:
                return "No previous conversation in this session."
            parts = []
            for m in msgs[-6:]:
                role = "User" if isinstance(m, HumanMessage) else "Analyzer"
                parts.append(f"{role}: {m.content[:200]}{'…' if len(m.content) > 200 else ''}")
            return "\n".join(parts)
        except Exception:
            return ""

    # ── Progress report ──────────────────────────────────────────

    def generate_session_report(self, user_id: str) -> str:
        if user_id not in self.profiles:
            return "No profile found for this user."
        profile = self.profiles[user_id]
        log_text = "\n".join(
            f"[{e['timestamp']}] {e['role'].upper()}: {e['query'][:80]} → {e['answer_snippet']}"
            for e in profile.session_log
        ) or "No interactions yet."

        return self._call_llm([
            SystemMessage(content="You are a traffic analysis expert generating a session summary report."),
            HumanMessage(content=f"""Generate a concise session summary for:

USER PROFILE:
{profile.summary()}

SESSION INTERACTIONS:
{log_text}

Structure as:
1. Profile Summary
2. Topics discussed
3. Key insights provided
4. Recommended next steps""")
        ])

    def stats(self) -> dict:
        return {
            "active_profiles": len(self.profiles),
            "roles": {k: v["name"] for k, v in ROLES.items()},
        }


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   🚦  TRAFFIC ANALYZER  —  Powered by Groq Llama 3.3 70B + RAG  ║
║   6 roles · Indian cities · Congestion · Safety · Green transport║
╚══════════════════════════════════════════════════════════════════╝

Ask anything about:
  🚦 Traffic flow & congestion     🛣️  Route planning & ETA
  🚧 Accidents & road safety       📊 Traffic data & trends
  🏗️  Infrastructure & smart city  🌿 Emissions & green transport

Commands:  report | profile | stats | role:<id> <question> | quit
"""


def main():
    agent = TrafficAnalyzerAgent()

    print("\n📚 Loading knowledge base …")
    for category, text in TRAFFIC_KNOWLEDGE_BASE.items():
        label = category.replace("_", " ").title()
        agent.ingest_text(text, title=f"Built-in: {label}", category=category)
    print()

    print(BANNER)

    try:
        user_id = input("User ID (press Enter for 'guest'): ").strip() or "guest"
        name    = input("Your name (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        return

    profile = agent.profiles.setdefault(user_id, TrafficUserProfile(user_id))
    if name:
        profile.name = name

    print(f"\n👋 Welcome{' ' + name if name else ''}! Ask me anything about traffic.\n")

    while True:
        try:
            raw = input(f"[{user_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye! Drive safe!"); break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print("👋 Goodbye!"); break

        elif cmd == "profile":
            print(f"\n👤 Profile:\n{profile.summary()}")
            print(f"   Interactions: {len(profile.session_log)}")

        elif cmd == "report":
            print("\n⏳ Generating session report …\n")
            print("─" * 60)
            print(agent.generate_session_report(user_id))
            print("─" * 60)

        elif cmd == "stats":
            s = agent.stats()
            print(f"\n📊 Active profiles: {s['active_profiles']}")
            print(f"   Roles: {', '.join(s['roles'].values())}")

        elif cmd in ("help", "?"):
            print(BANNER)

        elif raw.lower().startswith("role:"):
            parts = raw[5:].split(" ", 1)
            if len(parts) == 2:
                resp = agent.analyse(parts[1].strip(), user_id=user_id,
                                     force_role=parts[0].strip())
                _print_response(resp)
            else:
                print("Usage: role:<role_id> <question>")
                print(f"Roles: {', '.join(ROLES.keys())}")

        else:
            resp = agent.analyse(raw, user_id=user_id)
            _print_response(resp)


def _print_response(resp: Dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"{resp['role_emoji']}  {resp['role']}  [{resp['timestamp']}]")
    print(sep)
    print(resp["answer"])
    if resp["sources"] and "manual" not in resp["sources"]:
        print(f"\n📚 Sources: {', '.join(resp['sources'][:3])}")
    print(sep + "\n")


if __name__ == "__main__":
    main()
