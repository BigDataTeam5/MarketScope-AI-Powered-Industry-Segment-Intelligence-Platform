"""
Fallback content for the MarketScope marketing book queries
Used when real content can't be retrieved from Pinecone/S3
"""

FALLBACK_CHUNKS = {
    "marketing_chunk_1": {
        "chunk_id": "marketing_chunk_1",
        "content": "Marketing segmentation is the process of dividing a market of potential customers into groups, or segments, based on different characteristics. The segments created are composed of consumers who will respond similarly to marketing strategies and who share traits such as interests, needs, or locations. In healthcare, segmentation is often based on demographics, psychographics, behaviors, and medical needs."
    },
    "marketing_chunk_2": {
        "chunk_id": "marketing_chunk_2",
        "content": "Product positioning refers to the process of establishing the image or identity of a product in the minds of consumers. The positioning of healthcare products should focus on key benefits and differentiators that address specific patient needs and pain points. Effective positioning clearly communicates how a product differs from competitors and why it's the best choice for the target market."
    },
    "marketing_chunk_3": {
        "chunk_id": "marketing_chunk_3",
        "content": "Marketing strategy in healthcare should include a clear value proposition. This involves identifying the key benefits that your product offers, ensuring they align with patient needs, and communicating them effectively. For healthcare products, value propositions often center around improved outcomes, better quality of life, convenience, or cost-effectiveness."
    },
    "marketing_chunk_4": {
        "chunk_id": "marketing_chunk_4",
        "content": "Market research is essential for healthcare product success. It involves gathering data about potential customers, competitors, and the market environment. In healthcare, this includes understanding patient demographics, disease prevalence, treatment patterns, healthcare provider preferences, reimbursement landscapes, and regulatory requirements."
    },
    "marketing_chunk_5": {
        "chunk_id": "marketing_chunk_5",
        "content": "The marketing mix (4Ps) in healthcare includes: Product (features, quality, packaging), Price (list price, discounts, reimbursement), Place (distribution channels, market coverage), and Promotion (advertising, sales force, public relations). For healthcare products, additional considerations include patient education, provider engagement, and regulatory compliance."
    }
}

def get_fallback_chunks(query: str, top_k: int = 3) -> list:
    """Return fallback chunks when real content is unavailable"""
    # Just return the top N chunks based on the requested amount
    return list(FALLBACK_CHUNKS.values())[:top_k]
