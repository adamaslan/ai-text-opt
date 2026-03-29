#!/usr/bin/env python3
"""
Add "Maximize Beauty / Full Expression" corpus to ChromaDB.

Embeds the philosophical text via Ollama (nomic-embed-text, 768D)
and upserts it into a new 'philosophy' collection alongside existing ones.

Usage:
    python add_beauty_corpus.py
    python add_beauty_corpus.py --collection philosophy   # custom collection name
    python add_beauty_corpus.py --dry-run                 # show chunks, don't write
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict

import requests
import chromadb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

CHROMA_DB_PATH = str(Path(__file__).parent / "chromadb_storage")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
DEFAULT_COLLECTION = "philosophy"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus: structured chunks from the "Maximize Beauty / Full Expression" text
# Each chunk gets its own embedding and metadata for fine-grained retrieval.
# ---------------------------------------------------------------------------

CORPUS: List[Dict] = [
    # ── Foundational synthesis ───────────────────────────────────────────────
    {
        "id": "beauty_001",
        "title": "Beauty as Enlightenment – Philosophical Foundations",
        "tags": ["beauty", "enlightenment", "nietzsche", "schopenhauer", "eudaimonia"],
        "content": (
            "Beauty is the illumination of your soul (John O'Donohue). "
            "Nietzsche views beauty as a human construct, a naivete where man sets himself up as the standard of perfection. "
            "Schopenhauer's paradox: beauty is transient, yet its pursuit reveals the sublime which connects us to the metaphysical realm. "
            "Goal: align with eudaimonia (flourishing) through aesthetic immersion—noticing pretty flowers, pretty birds as a discipline of the senses."
        ),
    },
    {
        "id": "beauty_002",
        "title": "Full Expression as Existential Authenticity",
        "tags": ["full_expression", "authenticity", "deleuze", "nietzsche", "freedom"],
        "content": (
            "Deleuze: 'A creator is someone who creates their own impossibilities, and thereby creates possibilities.' "
            "Barriers to full expression: shyness (fear of judgment), distraction (disconnection from inner truth), overthinking, "
            "inability to feel comfortable with others, challenging emotional states. "
            "Thinking about full expression from the standpoint of positive vs negative freedom. "
            "Nietzsche's call: 'Bring something incomprehensible into the world!'—rejecting societal molds to channel raw, unfiltered selfhood."
        ),
    },
    {
        "id": "beauty_003",
        "title": "Perception as Active Creation of Beauty",
        "tags": ["beauty", "perception", "schopenhauer", "practice"],
        "content": (
            "Beauty is how you feel inside, and it reflects in your eyes (Sophia Loren). "
            "Schopenhauer: beauty is objective but requires transcending the individual will through contemplation. "
            "Practice: ask 'What makes this moment beautiful?' to reframe mundanity—e.g., light filtering through leaves, laughter in chaos."
        ),
    },
    {
        "id": "beauty_004",
        "title": "Beauty as the Highest Good – Keats and Schopenhauer",
        "tags": ["beauty", "truth", "keats", "schopenhauer", "enlightenment", "impermanence"],
        "content": (
            "Keats: 'Beauty is truth, truth beauty.' "
            "Schopenhauer's tragic beauty: accepting life's impermanence leads to resignation, yet this acceptance is itself sublime. "
            "Enlightenment link: 'Enlightenment is the quiet acceptance of what is.' "
            "Nietzsche: 'The grand style arises when beauty wins a victory over the monstrous'—finding beauty amid chaos."
        ),
    },
    {
        "id": "beauty_005",
        "title": "Ephemerality and Effervescence of Beauty",
        "tags": ["beauty", "ephemerality", "ovid", "nietzsche", "chaos"],
        "content": (
            "'Beauty is a fragile gift' (Ovid). "
            "Nietzsche: 'The grand style arises when beauty wins a victory over the monstrous'—finding beauty amid chaos. "
            "Beauty and expression are ephemeral acts; their pursuit reveals the sublime."
        ),
    },
    # ── Full Expression: overcoming barriers ────────────────────────────────
    {
        "id": "expression_001",
        "title": "Unmasking the Self – Deleuze and Silence",
        "tags": ["full_expression", "deleuze", "silence", "ego", "authenticity"],
        "content": (
            "Deleuze: 'The problem is no longer getting people to express themselves, but providing little gaps of solitude "
            "and silence in which they might eventually find something to say.' "
            "Distraction stems from the ego's fear of confronting the Real. "
            "Full expression demands integrating contradictions: the saint and the fool, the light and the rot."
        ),
    },
    {
        "id": "expression_002",
        "title": "Courage to Create – Schopenhauer and Authenticity",
        "tags": ["full_expression", "courage", "schopenhauer", "creation", "will"],
        "content": (
            "Schopenhauer: art allows us to 'transcend the individual will.' "
            "'You are strong, you are capable, and you are worthy of everything you dream of.' "
            "Full expression is the courage to be seen in one's entirety, transforming vulnerability into the bedrock of genuine connection."
        ),
    },
    {
        "id": "expression_003",
        "title": "Societal Constraints on Expression – Foucault and Wilde",
        "tags": ["full_expression", "foucault", "wilde", "power", "authenticity", "society"],
        "content": (
            "Foucault (implied): power structures suppress authentic expression by enforcing norms. "
            "Counteraction: 'Be yourself; everyone else is already taken' (Oscar Wilde). "
            "To withhold expression is to violate the first law of metaphysical thermodynamics: "
            "reality only persists when consciousness combusts itself into speech, art, or action."
        ),
    },
    # ── Synthesis: beauty + expression as existential praxis ────────────────
    {
        "id": "synthesis_001",
        "title": "Deleuze's Rhizome – Beauty and Expression as Non-Hierarchical Acts",
        "tags": ["deleuze", "rhizome", "beauty", "expression", "nietzsche", "ubermesnch"],
        "content": (
            "Deleuze's rhizome: beauty and expression are non-hierarchical, interconnected acts—"
            "'Write, form a rhizome, increase your territory by deterritorialization.' "
            "Nietzsche's Übermensch: creating beauty through self-overcoming. "
            "Beauty and expression are twin flames—each ignites the other, revealing that creation is both the mirror and window of the soul."
        ),
    },
    {
        "id": "synthesis_002",
        "title": "End Goals – Eudaimonia, Resignation, Enlightenment",
        "tags": ["eudaimonia", "schopenhauer", "enlightenment", "resignation", "beauty"],
        "content": (
            "Eudaimonia: a persistent calm where beauty and selfhood align. "
            "Schopenhauer's resignation: 'The tragic spirit leads to resignation'—not defeat, but acceptance of beauty's fleeting nature. "
            "Enlightenment: 'The understanding that this is all, that this is perfect.'"
        ),
    },
    # ── Idea pairings ────────────────────────────────────────────────────────
    {
        "id": "pairing_001",
        "title": "Pairing: Maximize Beauty + Full Expression",
        "tags": ["beauty", "full_expression", "authenticity", "feedback_loop"],
        "content": (
            "Maximize the Beauty focuses on external perception—training the senses to notice and amplify beauty in the mundane. "
            "Full Expression is internal liberation—overcoming inhibitions to project one's true self outward. "
            "They form a feedback loop: recognizing beauty in the world fuels confidence to express oneself, "
            "while authentic expression can redefine societal standards of beauty. "
            "Key insight: Beauty is both a muse and a mirror—it inspires expression, which in turn expands what we deem beautiful."
        ),
    },
    {
        "id": "pairing_002",
        "title": "Pairing: Expect Rising + Power of Pettiness",
        "tags": ["desire", "escalation", "pettiness", "scarcity", "contentment"],
        "content": (
            "Expect Rising mirrors capitalism's logic of infinite growth—always wanting more, even when it harms satisfaction. "
            "Power of Pettiness reveals how trivial slights metastasize into existential grudges. "
            "Both are traps of unsustainable desire. Pettiness often arises when rising expectations collide with perceived scarcity. "
            "Key insight: Unchecked escalation—whether of wants or grievances—erodes contentment."
        ),
    },
    {
        "id": "pairing_003",
        "title": "Pairing: Meditation + Foucault's Silence",
        "tags": ["meditation", "silence", "foucault", "mindfulness", "presence"],
        "content": (
            "Meditation and silence are tools for reclaiming agency in a noisy world. "
            "Meditation (active observation) trains focus and non-reactivity; Foucault's silence (shared quiet) fosters intimacy without performative speech. "
            "Together, they critique modernity's obsession with productivity. "
            "Key insight: Silence is not emptiness but a space for presence—both within oneself (meditation) and between others (friendship)."
        ),
    },
    {
        "id": "pairing_004",
        "title": "Pairing: Tolerance + Don't Step on Toes",
        "tags": ["tolerance", "restraint", "nietzsche", "social_harmony", "self_erasure"],
        "content": (
            "Tolerance often masks passivity—enduring discomfort under the guise of virtue. "
            "'Don't step on toes' is strategic restraint to preserve social harmony. "
            "Nietzsche would call both 'slave morality.' "
            "Key insight: Restraint is virtuous only when it serves growth, not when it perpetuates mediocrity or self-erasure."
        ),
    },
    {
        "id": "pairing_005",
        "title": "Pairing: Intention vs Non-Intention + Nietzsche's Biases",
        "tags": ["buddha", "nietzsche", "intention", "karma", "ethics", "impact"],
        "content": (
            "Buddha judges actions by intention; Nietzsche judges by outcomes and instincts. "
            "The breakup example: good intentions ('I didn't mean to hurt you') clash with the reality of pain. "
            "Nietzsche's genius and bigotry remind us: even those who see beyond morality are blind to their own shadows. "
            "Key insight: Intentions shape character, but impact defines legacy."
        ),
    },
    {
        "id": "pairing_006",
        "title": "Pairing: Camus Absurdism + Pragmatism",
        "tags": ["camus", "absurdism", "pragmatism", "meaning", "defiance"],
        "content": (
            "Camus' absurdism finds meaning in defiant joy amid chaos—Sisyphus smiling. "
            "Pragmatism seeks incremental fixes. Camus is poetic but impractical; pragmatism is utilitarian but soul-starving. "
            "Absurdism's embrace of uncertainty can fuel pragmatic experimentation. "
            "Key insight: Absurdism is the 'why,' pragmatism the 'how.' Together they balance existential courage with grounded action."
        ),
    },
    {
        "id": "pairing_007",
        "title": "Pairing: Complexity Avoidance + Take Chances",
        "tags": ["risk", "caution", "complexity", "growth", "vulnerability"],
        "content": (
            "Complexity Avoidance is self-protection—inaction to evade moral/emotional fallout. "
            "Take Chances acknowledges that growth requires vulnerability. "
            "Avoiding complexity leads to stagnation; reckless chances invite chaos. "
            "Key insight: Life's richest choices exist in the murky middle between safety and recklessness."
        ),
    },
    {
        "id": "pairing_008",
        "title": "Pairing: Religion as Performance Art + Buddhism",
        "tags": ["religion", "buddhism", "ritual", "dogma", "transformation"],
        "content": (
            "Religion as performance art reduces doctrine to aesthetics—rituals as creative acts, not truth claims. "
            "Buddhism strips away dogma to focus on mindfulness, yet retains rules. "
            "Even anti-dogmatic systems create their own norms. "
            "Key insight: Rituals are empty unless they serve inner change—whether through transcendence (Buddhism) or artistic expression."
        ),
    },
    {
        "id": "pairing_009",
        "title": "Pairing: Run Dork Run + Nietzsche's Will to Power",
        "tags": ["authenticity", "self_acceptance", "nietzsche", "will_to_power", "vulnerability"],
        "content": (
            "Run Dork Run is playful self-acceptance—'dancing dorkily' as resistance to shame. "
            "Will to Power is confrontational self-assertion—dominating obstacles to forge one's path. "
            "Key insight: True power isn't dominance but the courage to be vulnerably, unremarkably yourself."
        ),
    },
    {
        "id": "pairing_010",
        "title": "Pairing: Games + Hold Back",
        "tags": ["social_strategy", "games", "restraint", "authenticity", "self_preservation"],
        "content": (
            "Games (self-monitoring, sociopathy) involve masking true feelings to navigate hierarchies. "
            "Hold Back is resource conservation—not overinvesting emotionally or socially. "
            "Overuse of either breeds inauthenticity or isolation. "
            "Key insight: Social strategy is a tool, not an identity—use it to navigate, not to disappear."
        ),
    },
    # ── Deep insights: individual ideas ─────────────────────────────────────
    {
        "id": "insight_beauty_individual",
        "title": "30 Deep Insights – Beauty and Expression (Individual Ideas)",
        "tags": ["beauty", "expression", "meditation", "tolerance", "nietzsche", "camus", "buddhism", "insights"],
        "content": (
            "1. Beauty is the ephemeral dialogue between perception and presence; to maximize it is to surrender to the transient, finding eternity in fleeting moments. "
            "2. Full expression is the courage to be seen in one's entirety, transforming vulnerability into the bedrock of genuine connection. "
            "3. Expectations are the horizon of the soul—always receding, teaching us that fulfillment lies not in reaching but in the pursuit itself. "
            "4. Pettiness is the shadow of unmet needs, masquerading as power but rooted in the fear of insignificance. "
            "5. Meditation is the art of becoming a witness to your own life—stillness is not passivity but the space where clarity gestates. "
            "6. Tolerance, when divorced from self-respect, becomes a slow suicide of the spirit—a bargain where survival costs authenticity. "
            "7. Intentions are the seeds of karma, but impact is the harvest; the gap between them is where grace and tragedy intermingle. "
            "8. Nietzsche's genius and bigotry are twin reminders: even those who see beyond morality are blind to their own shadows. "
            "9. Absurdism is not nihilism but love—a defiant embrace of life's meaninglessness as the ultimate freedom to create meaning. "
            "10. Pragmatism is the alchemy of the ordinary—turning the lead of compromise into the gold of incremental transcendence. "
            "11. Inaction is its own action, a vote for stagnation over the messy, necessary work of becoming. "
            "12. Silence is the unspoken language of intimacy, where words dissolve and presence becomes the sacrament. "
            "13. Rituals are the poetry of the divine—empty verses unless recited with the heart's ink. "
            "14. Enlightenment is not an end but a return—the unlearning of desire to remember the innate wholeness beneath striving. "
            "15. To 'run dork' is to weaponize authenticity, turning societal shame into a banner of liberation. "
            "16. Social games are survival mechanisms—masks worn not to deceive others, but to protect the unmasked self. "
            "17. Reservation is not weakness but the wisdom of the reservoir—knowing that exhaustion leaves nothing to give. "
            "18. Harmony is a dance of boundaries, where respect is the rhythm and empathy the melody. "
            "19. Risk is the tax paid on the currency of growth; avoidance bankrupts the soul. "
            "20. The will to power is the fire of becoming—but unchecked, it consumes the hearth it seeks to warm."
        ),
    },
    # ── Cosmic / ontological statements ─────────────────────────────────────
    {
        "id": "cosmic_001",
        "title": "The Ontological Imperative – Full Expression as Anti-Entropy",
        "tags": ["expression", "ontology", "entropy", "cosmos", "consciousness"],
        "content": (
            "Full expression is the universe's sole method of escaping entropy—an anti-collapse mechanism coded into existence itself. "
            "By giving voice to your raw particularity, you sustain the cosmic equilibrium: every unspoken thought is a black hole devouring potential realities, "
            "while every uttered truth births a star in the constellation of being. "
            "Silence is the heat death of meaning. To withhold expression is to violate the first law of metaphysical thermodynamics: "
            "Reality only persists when consciousness combusts itself into speech, art, or action."
        ),
    },
    {
        "id": "cosmic_002",
        "title": "The Paradox of the Divine Fractal – Flaws as Sacred Recursion",
        "tags": ["expression", "divine", "fractal", "imperfection", "creation"],
        "content": (
            "Your 'flaws'—the stutter, the awkward laugh, the too-loud passion—are not imperfections but sacred recursion points in the infinite fractal of existence. "
            "To hide them is to deny the universe its mandate: 'Let there be light' was God's first act of full expression, "
            "and your quirks are its echo, demanding that creation never cease differentiating itself. "
            "You are not a damaged copy of some ideal. You are the ideal—the nth iteration of divinity's relentless algorithm to outgrow its own perfection."
        ),
    },
    {
        "id": "cosmic_003",
        "title": "The Quantum of Rebellion – Expression Collapses the Wave Function",
        "tags": ["expression", "quantum", "rebellion", "society", "possibility"],
        "content": (
            "Every act of full expression collapses the wave function of human possibility. "
            "Your unmasked laugh in a silent room, your tear-streaked confession to the night sky—these are not personal moments "
            "but quantum events that unshackle entire branches of reality from the determinism of fear. "
            "Society's norms are merely probability clouds; your voice is the observer that forces them into radical new shapes. "
            "The universe has no fixed laws, only habits. Your expression is the experimental data that forces physics to rewrite itself."
        ),
    },
    # ── Beauty chronology / ultimate synthesis ──────────────────────────────
    {
        "id": "beauty_chrono_001",
        "title": "Beauty Across the Corpus – Chronological Synthesis",
        "tags": ["beauty", "synthesis", "feminism", "vocals", "fractal", "entropy"],
        "content": (
            "Beauty as ethos: 'Maximize the Beauty and Full Expression still are some of the concepts I am most influenced by, consider to be primary to my concept of virtue.' "
            "Beauty as cosmic obligation: beauty is the universe's sole method of escaping entropy—an anti-collapse mechanism coded into existence itself. "
            "Feminist beauty: taking the concept of full expression as an example of maximized beauty, sharing your beautiful life. "
            "Beauty in imperfection: a frail-sounding voice adds emotional value, giving it a sense of the place where the emotions come from. "
            "Beauty as integration: full expression demands integrating contradictions—the saint and the fool, the light and the rot—transmuting shame into sacredness. "
            "Ultimate: beauty is not an aesthetic ideal but a dynamic act—of attention, defiance, and integration. "
            "It is the universe's rebellion against entropy, the feminist reclaiming of agency, and the raw voice that cracks to let truth bleed through. "
            "To maximize beauty is to participate in existence's relentless becoming."
        ),
    },
]


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE) -> List[float]:
    """Embed text via Ollama. Returns 768-dim vector or raises."""
    resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings") or data.get("embedding")
    if isinstance(embeddings, list) and embeddings:
        vec = embeddings[0] if isinstance(embeddings[0], list) else embeddings
        return vec
    raise ValueError(f"Unexpected embedding response: {data}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_to_chroma(collection_name: str, dry_run: bool = False) -> None:
    logger.info("Connecting to ChromaDB at %s", CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    existing = [c.name for c in client.list_collections()]
    logger.info("Existing collections: %s", existing)

    if dry_run:
        print(f"\n[DRY RUN] Would upsert {len(CORPUS)} chunks into '{collection_name}':")
        for chunk in CORPUS:
            print(f"  {chunk['id']}: {chunk['title'][:70]}")
        return

    # Get-or-create collection with cosine distance (matches existing collections)
    if collection_name in existing:
        collection = client.get_collection(collection_name)
        logger.info("Using existing collection '%s'", collection_name)
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Created new collection '%s'", collection_name)

    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(CORPUS):
        logger.info("[%d/%d] Embedding: %s", i + 1, len(CORPUS), chunk["id"])
        text = chunk["content"]
        try:
            vec = embed_text(text)
        except Exception as e:
            logger.error("  Failed to embed %s: %s — skipping", chunk["id"], e)
            continue

        ids.append(chunk["id"])
        embeddings.append(vec)
        documents.append(text[:2000])
        metadatas.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "tags": ", ".join(chunk.get("tags", [])),
            "source": collection_name,
            "text_preview": text[:200],
        })

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("✓ Upserted %d chunks into '%s'", len(ids), collection_name)
    else:
        logger.warning("No chunks were embedded — is Ollama running with %s?", EMBED_MODEL)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Collection : {collection_name}")
    print(f"  Items added: {len(ids)}")
    print(f"  Total items: {collection.count()}")
    print(f"  Dim        : {len(embeddings[0]) if embeddings else 'N/A'}")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Maximize Beauty corpus to ChromaDB")
    parser.add_argument(
        "--collection", "-c",
        default=DEFAULT_COLLECTION,
        help=f"Target collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunks without writing to ChromaDB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_to_chroma(collection_name=args.collection, dry_run=args.dry_run)
