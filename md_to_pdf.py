# md_to_pdf.py - Convert Markdown reports to PDF

import os
import re


def convert_md_to_pdf(markdown_text: str, topic: str = "report") -> str:
    """Convert markdown text to a PDF file using weasyprint."""
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)

    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()[:50]
    safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
    output_path = os.path.join(output_dir, f"{safe_topic}.pdf")

    try:
        import markdown
        from weasyprint import HTML

        # Convert markdown to HTML
        html_body = markdown.markdown(
            markdown_text,
            extensions=['extra', 'codehilite', 'tables']
        )

        # Wrap in full HTML with styling
        html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
h2 {{ color: #16213e; margin-top: 30px; }}
h3 {{ color: #0f3460; }}
p {{ margin: 10px 0; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
blockquote {{ border-left: 4px solid #e94560; margin-left: 0; padding-left: 20px; color: #666; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #16213e; color: white; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

        HTML(string=html_full).write_pdf(output_path)
        print(f"PDF saved: {output_path}")
        return output_path

    except ImportError as e:
        print(f"PDF conversion not available (install weasyprint): {e}")
        return ""
    except Exception as e:
        print(f"PDF generation error: {e}")
        return ""
