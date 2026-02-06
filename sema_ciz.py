import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- GÜNCELLEME: KAYIT YERİNİ SABİTLEME ---
# Kod dosyasının olduğu klasörü bul
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Hedef Klasör: {script_dir}")

def draw_box(ax, x, y, w, h, text, color='#E0E0E0', ec='black'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=ec, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x+w/2, y 

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

# ---------------------------------------------------------
# 1. EARLY FUSION ŞEMASI
# ---------------------------------------------------------
def create_early_fusion():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
   # ax.set_title("Şekil 1: Early Fusion Mimarisi", fontsize=14, y=0.02)

    draw_box(ax, 1, 10, 3, 1.5, "Görüntü\n(Image)", color='#FFD700')
    draw_box(ax, 6, 10, 3, 1.5, "Metin\n(Text)", color='#87CEEB')
    draw_arrow(ax, 2.5, 10, 2.5, 8.5)
    draw_arrow(ax, 7.5, 10, 7.5, 8.5)

    draw_box(ax, 1, 7, 3, 1.5, "ResNet-18\n(Encoder)", color='#FFD700')
    draw_box(ax, 6, 7, 3, 1.5, "BERT\n(Encoder)", color='#87CEEB')
    draw_arrow(ax, 2.5, 7, 2.5, 5.5)
    draw_arrow(ax, 7.5, 7, 7.5, 5.5)

    draw_box(ax, 1.5, 4.5, 2, 1, "512-D\nVector", color='#FFFACD')
    draw_box(ax, 6.5, 4.5, 2, 1, "768-D\nVector", color='#E0FFFF')
    draw_arrow(ax, 2.5, 4.5, 5, 3.5)
    draw_arrow(ax, 7.5, 4.5, 5, 3.5)

    draw_box(ax, 3.5, 2.5, 3, 1, "Concatenate\n(1280-D)", color='#FFA07A')
    draw_arrow(ax, 5, 2.5, 5, 1.5)

    draw_box(ax, 3.5, 0.5, 3, 1, "Classifier\n(MLP)", color='#98FB98')

    # TAM YOL İLE KAYDET
    save_path = os.path.join(script_dir, "sema_early.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"KAYDEDİLDİ: {save_path}")

# ---------------------------------------------------------
# 2. INTERMEDIATE FUSION ŞEMASI
# ---------------------------------------------------------
def create_intermediate_fusion():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
   # ax.set_title("Şekil 2: Intermediate Fusion (Cross-Attention)", fontsize=14, y=0.02)

    draw_box(ax, 1, 10, 3, 1.5, "Görüntü", color='#FFD700')
    draw_box(ax, 6, 10, 3, 1.5, "Metin", color='#87CEEB')
    draw_arrow(ax, 2.5, 10, 2.5, 8.5)
    draw_arrow(ax, 7.5, 10, 7.5, 8.5)

    draw_box(ax, 1, 7, 3, 1.5, "ResNet-18", color='#FFD700')
    draw_box(ax, 6, 7, 3, 1.5, "BERT", color='#87CEEB')
    draw_arrow(ax, 2.5, 7, 2.5, 6)
    draw_arrow(ax, 7.5, 7, 7.5, 6)

    draw_box(ax, 1.5, 5, 2, 1, "Proj.\n(256-D)", color='#FFFACD')
    draw_box(ax, 6.5, 5, 2, 1, "Proj.\n(256-D)", color='#E0FFFF')

    rect = patches.Rectangle((2, 2), 6, 2, linewidth=2, edgecolor='purple', facecolor='#F5E6F5')
    ax.add_patch(rect)
    ax.text(5, 3, "CROSS-ATTENTION BLOCK", ha='center', va='center', fontweight='bold')
    
    ax.annotate("Query", xy=(3, 4), xytext=(2.5, 5), arrowprops=dict(arrowstyle="->", color='red'))
    ax.annotate("Key", xy=(6, 4), xytext=(7.5, 5), arrowprops=dict(arrowstyle="->", color='blue'))
    ax.annotate("Value", xy=(7, 4), xytext=(7.8, 5), arrowprops=dict(arrowstyle="->", color='blue'))

    draw_arrow(ax, 5, 2, 5, 1.2)
    draw_box(ax, 3.5, 0.2, 3, 1, "Classifier", color='#98FB98')

    save_path = os.path.join(script_dir, "sema_inter.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"KAYDEDİLDİ: {save_path}")

# ---------------------------------------------------------
# 3. LATE FUSION ŞEMASI
# ---------------------------------------------------------
def create_late_fusion():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
   # ax.set_title("Şekil 3: Late Fusion (Decision Level)", fontsize=14, y=0.02)

    # SOL
    draw_box(ax, 1, 10, 3, 1.2, "Görüntü", color='#FFD700')
    draw_arrow(ax, 2.5, 10, 2.5, 9)
    draw_box(ax, 1, 8, 3, 1, "ResNet-18", color='#FFD700')
    draw_arrow(ax, 2.5, 8, 2.5, 7)
    draw_box(ax, 1, 6, 3, 1, "Softmax\n(Olasılık 1)", color='#FFFACD')

    # SAĞ
    draw_box(ax, 6, 10, 3, 1.2, "Metin", color='#87CEEB')
    draw_arrow(ax, 7.5, 10, 7.5, 9)
    draw_box(ax, 6, 8, 3, 1, "BERT", color='#87CEEB')
    draw_arrow(ax, 7.5, 8, 7.5, 7)
    draw_box(ax, 6, 6, 3, 1, "Softmax\n(Olasılık 2)", color='#E0FFFF')

    # BİRLEŞİM
    draw_arrow(ax, 2.5, 6, 4, 4.5)
    draw_arrow(ax, 7.5, 6, 6, 4.5)
    draw_box(ax, 3, 3, 4, 1.5, "Weighted Average\n(Ortalama Alma)", color='#FFA07A')
    draw_arrow(ax, 5, 3, 5, 2)
    draw_box(ax, 3.5, 1, 3, 1, "Final Karar", color='#98FB98')

    save_path = os.path.join(script_dir, "sema_late.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"KAYDEDİLDİ: {save_path}")

if __name__ == "__main__":
    create_early_fusion()
    create_intermediate_fusion()
    create_late_fusion()
    print("\nGÖRSELLER PROJE KLASÖRÜNE BIRAKILDI!")