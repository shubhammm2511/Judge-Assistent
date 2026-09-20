from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "rag_workflow.png"
WIDTH, HEIGHT = 2000, 1480


def font(size: int, bold: bool = False):
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / filename, size)


TITLE = font(48, True)
SECTION = font(30, True)
NODE_TITLE = font(25, True)
BODY = font(21)
SMALL = font(18)


image = Image.new("RGB", (WIDTH, HEIGHT), "#F7F9FC")
draw = ImageDraw.Draw(image)


def rounded_box(x, y, w, h, fill, outline="#D6DEEA", radius=22, title="", body=""):
    draw.rounded_rectangle((x, y, x + w, y + h), radius, fill=fill, outline=outline, width=3)
    if title:
        draw.text((x + 24, y + 18), title, fill="#14213D", font=NODE_TITLE)
    if body:
        lines = []
        for paragraph in body.split("\n"):
            lines.extend(wrap(paragraph, width=max(22, int(w / 18))))
        draw.multiline_text(
            (x + 24, y + 62),
            "\n".join(lines),
            fill="#52627A",
            font=BODY,
            spacing=7,
        )


def arrow(x1, y1, x2, y2, color="#6B7CF6", width=6):
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    size = 15
    if abs(x2 - x1) >= abs(y2 - y1):
        points = [(x2, y2), (x2 - size, y2 - size // 2), (x2 - size, y2 + size // 2)]
    else:
        points = [(x2, y2), (x2 - size // 2, y2 - size), (x2 + size // 2, y2 - size)]
    draw.polygon(points, fill=color)


draw.text((70, 48), "Agent-as-a-Judge RAG Workflow", fill="#14213D", font=TITLE)
draw.text(
    (72, 112),
    "The PDF is indexed once, then each question retrieves only the most relevant evidence before generation.",
    fill="#66758C",
    font=BODY,
)

# Two clear lanes make the indexing and query-time paths easy to explain.
draw.rounded_rectangle((60, 190, 960, 1370), 28, fill="#EEF2FF", outline="#D8DEFF", width=3)
draw.rounded_rectangle((1040, 190, 1940, 1370), 28, fill="#F2F8F5", outline="#D5EADF", width=3)
draw.text((100, 225), "1. Indexing / ingestion", fill="#4053D6", font=SECTION)
draw.text((1080, 225), "2. Question answering", fill="#227A52", font=SECTION)

# Indexing lane.
ix_x, ix_w = 155, 710
ix_nodes = [
    (280, "PDF document", "Agent-as-Judge.pdf\n44 non-empty pages", "#FFFFFF"),
    (455, "Text extraction", "PyMuPDF extracts page text\nand preserves page metadata", "#FFFFFF"),
    (630, "Cleaning and chunking", "Recursive chunks: ~900 characters\nwith overlap and chunk IDs", "#FFFFFF"),
    (805, "Metadata creation", "source filename · page number\nchunk ID", "#FFFFFF"),
    (980, "Embeddings", "Sentence-Transformers\nall-MiniLM-L6-v2 · 384 dimensions", "#FFFFFF"),
    (1155, "Vector database", "Chroma stores vectors, text,\nand metadata for similarity search", "#FFFFFF"),
]
for y, title, body, fill in ix_nodes:
    rounded_box(ix_x, y, ix_w, 120, fill, title=title, body=body)
for y in [400, 575, 750, 925, 1100]:
    arrow(ix_x + ix_w // 2, y, ix_x + ix_w // 2, y + 50)

# Query lane.
qx, qw = 1135, 710
q_nodes = [
    (310, "User question", "Example: What is the DevAI dataset?", "#FFFFFF"),
    (485, "Query embedding", "Convert the question into the\nsame embedding space", "#FFFFFF"),
    (660, "Similarity search", "Compare query vector with\nindexed chunk vectors", "#FFFFFF"),
    (835, "Top-K retrieved chunks", "Relevant text + source/page\nmetadata + similarity score", "#FFFFFF"),
    (1010, "Context construction", "Combine retrieved chunks\nwith the grounded prompt", "#FFFFFF"),
    (1185, "LLM answer + sources", "Concise answer, retrieved chunks,\npage numbers, and scores", "#FFFFFF"),
]
for y, title, body, fill in q_nodes:
    rounded_box(qx, y, qw, 120, fill, title=title, body=body)
for y in [430, 605, 780, 955, 1130]:
    arrow(qx + qw // 2, y, qx + qw // 2, y + 50, color="#35A36C")

# Cross-lane relationship.
arrow(ix_x + ix_w + 25, 1215, qx - 25, 1215, color="#8A95AC", width=5)
draw.text((870, 1170), "indexed evidence", fill="#66758C", font=SMALL)

# Footer principle.
draw.rounded_rectangle((160, 1410, 1840, 1450), 16, fill="#14213D")
draw.text(
    (190, 1418),
    "Grounding rule: answer only from retrieved context; if evidence is missing, say so.",
    fill="#FFFFFF",
    font=SMALL,
)

image.save(OUTPUT, "PNG", optimize=True)
print(f"Created {OUTPUT}")
