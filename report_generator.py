import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image as PILImage

# --- Native PDF Drawing Functions ---

def draw_health_score_bar(score):
    """Draws a horizontal progress bar for the Health Score using native vector shapes."""
    d = Drawing(450, 40)
    
    # Background Bar (Grey)
    d.add(Rect(0, 15, 300, 20, fillColor=colors.HexColor('#f0f0f0'), strokeColor=None))
    
    # Foreground Bar (Dark Blue)
    bar_width = 300 * (score / 100.0)
    d.add(Rect(0, 15, bar_width, 20, fillColor=colors.HexColor('#0e2a47'), strokeColor=None))
    
    # 0 and 100 Labels under the bar
    d.add(String(0, 2, "0", fontSize=9, fillColor=colors.gray))
    d.add(String(285, 2, "100", fontSize=9, fillColor=colors.gray))
    
    # Actual Score Text beside the bar
    d.add(String(bar_width + 10, 20, f"{score}", fontSize=12, fontName="Helvetica-Bold", fillColor=colors.black))
    
    return d

def draw_distribution_bars(stats):
    """Draws independent horizontal progress bars for each capillary type natively."""
    total = sum(stats.values()) if stats else 1
    categories = ['Normal', 'Blur', 'Abnormal', 'Hemo', 'Aggregation']
    
    # Note: Slightly darkened the Yellow ('#D4AC0D') for better readability on white background.
    color_map = {
        'Normal': '#00FF00',       
        'Blur': '#D4AC0D',         
        'Abnormal': '#800080',     
        'Hemo': '#00FFFF',         
        'Aggregation': '#FF0000'   
    }
    
    # Calculate drawing height based on number of categories
    row_height = 35
    total_height = len(categories) * row_height
    d = Drawing(450, total_height)
    
    y_pos = total_height - row_height # Start drawing from the top down
    
    for cat in categories:
        val = stats.get(cat, 0)
        pct = (val / total) * 100
        
        # Category Label Text
        d.add(String(0, y_pos + 5, cat, fontSize=10, fontName="Helvetica-Bold"))
        
        # Background Bar (Grey, max 250 pixels wide)
        d.add(Rect(80, y_pos, 250, 15, fillColor=colors.HexColor('#f0f0f0'), strokeColor=None))
        
        # Foreground Bar (Colored)
        bar_width = 250 * (pct / 100.0)
        d.add(Rect(80, y_pos, bar_width, 15, fillColor=colors.HexColor(color_map[cat]), strokeColor=None))
        
        # 0 and 100 Labels
        d.add(String(80, y_pos - 10, "0", fontSize=8, fillColor=colors.gray))
        d.add(String(315, y_pos - 10, "100", fontSize=8, fillColor=colors.gray))
        
        # Percentage Text beside the bar
        d.add(String(80 + bar_width + 10, y_pos + 4, f"{pct:.1f}%", fontSize=10, fontName="Helvetica-Bold"))
        
        y_pos -= row_height
        
    return d

def format_gemini_markdown(text):
    """Basic parser to convert Gemini's markdown (**) to ReportLab bold tags (<b>)."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text.split('\n')

# --- Main PDF Generator ---

def create_pdf(user_data, stats, health_score, density, ai_text, overlay_image_rgb):
    """Constructs the PDF document dynamically."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # 1. Document Title
    story.append(Paragraph("Clinical Pathology Report", title_style))
    story.append(Spacer(1, 20))
    
    # 2. Patient Meta Data Table
    patient_info = [
        ["Subject ID:", user_data['name'], "Date:", user_data['date']],
        ["Age:", str(user_data['age']), "Gender:", user_data['gender']],
        ["Field of View:", f"{user_data['fov']} mm", "Linear Density:", f"{density:.2f} loops/mm"]
    ]
    
    t = Table(patient_info, colWidths=[80, 150, 80, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9f9f9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 30))
    
    # 3. Health Score Section (Native Drawing)
    story.append(Paragraph("Overall Health Score", heading_style))
    story.append(Spacer(1, 10))
    story.append(draw_health_score_bar(health_score))
    story.append(Spacer(1, 20))
    
    # 4. Capillary Composition Breakdown (Native Drawing)
    story.append(Paragraph("Capillary Composition Breakdown", heading_style))
    story.append(Spacer(1, 10))
    story.append(draw_distribution_bars(stats))
    story.append(Spacer(1, 30))
    
    # 5. Segmented Image Overlay
    story.append(Paragraph("Semantic Segmentation Analysis", heading_style))
    story.append(Spacer(1, 10))
    
    # Convert numpy RGB array to PIL Image, then to ReportLab Image
    img_pil = PILImage.fromarray(overlay_image_rgb)
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    rl_img = RLImage(img_buffer, width=400, height=300)
    story.append(rl_img)
    story.append(Spacer(1, 30))
    
    # 6. Clinical Assessment Text (From Gemini)
    story.append(Paragraph("Clinical Assessment", heading_style))
    story.append(Spacer(1, 10))
    
    formatted_paragraphs = format_gemini_markdown(ai_text)
    for p_text in formatted_paragraphs:
        if p_text.strip():
            story.append(Paragraph(p_text, normal_style))
            story.append(Spacer(1, 8))
            
    # Build the document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes