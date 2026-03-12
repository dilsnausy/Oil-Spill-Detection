
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
from pathlib import Path
import sys

# CONFIG
BASE_DIR = Path("d:/antigravity/sat")
IMAGE_DIR = BASE_DIR / "dataset_out/sent2_tiff/_preview_gallery_pairs"
CSV_PATH = BASE_DIR / "combined_pairs_with_bbox.csv"
SELECTED_IDS_FILE = "selected_ids.txt"
OUTPUT_CSV = "selected_pairs.csv"

def safe_name(s: str) -> str:
    s = str(s).strip().replace("\\", "_").replace("/", "_")
    return "".join([c for c in s if c.isalnum() or c in "._-"])[:220]

class ImageSelectorApp:
    def __init__(self, root, df):
        self.root = root
        self.root.title("Satellite Image Pair Selector")
        self.df = df
        self.current_idx = 0
        self.selected_ids = []
        self.resize_timer = None
        
        # Load previous selection
        self.load_previous_selection()
        
        # UI Setup
        self.setup_ui()
        
        # Initial load might happen before layout is ready, so we wait slightly or rely on Configure event
        self.root.after(100, self.load_current_pair)
        
        # Shortcuts
        self.root.bind('<Right>', lambda e: self.next_pair())
        self.root.bind('<Left>', lambda e: self.prev_pair())
        self.root.bind('<space>', lambda e: self.toggle_select())
        self.root.bind('<Return>', lambda e: self.save_and_exit())

    def load_previous_selection(self):
        if os.path.exists(SELECTED_IDS_FILE):
            try:
                with open(SELECTED_IDS_FILE, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]
                    self.selected_ids = [l for l in lines if l]
                print(f"Loaded {len(self.selected_ids)} previously selected IDs.")
            except Exception as e:
                print(f"Error loading previous selection: {e}")

    def setup_ui(self):
        # Top Frame
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_status = tk.Label(top_frame, text="", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side=tk.LEFT)
        
        self.lbl_selected = tk.Label(top_frame, text=f"Selected: {len(self.selected_ids)}", font=("Arial", 14))
        self.lbl_selected.pack(side=tk.RIGHT)

        # Image Frame (Expandable)
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.img_frame.grid_columnconfigure(0, weight=1)
        self.img_frame.grid_columnconfigure(1, weight=1)
        self.img_frame.grid_rowconfigure(1, weight=1)
        
        # Labels
        self.lbl_sar = tk.Label(self.img_frame, text="SAR Log-Scaled", font=("Arial", 12))
        self.lbl_sar.grid(row=0, column=0, sticky="ew")
        
        self.lbl_opt = tk.Label(self.img_frame, text="Optical RGB", font=("Arial", 12))
        self.lbl_opt.grid(row=0, column=1, sticky="ew")
        
        # Canvases (Black background for contrast)
        self.canvas_sar = tk.Canvas(self.img_frame, bg="#202020", highlightthickness=0)
        self.canvas_sar.grid(row=1, column=0, padx=2, sticky="nsew")
        
        self.canvas_opt = tk.Canvas(self.img_frame, bg="#202020", highlightthickness=0)
        self.canvas_opt.grid(row=1, column=1, padx=2, sticky="nsew")
        
        # Bottom Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btn_frame, text="< Prev (Left)", command=self.prev_pair, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.btn_select = tk.Button(btn_frame, text="Select (Space)", command=self.toggle_select, bg="lightgray", width=25, font=("Arial", 12, "bold"))
        self.btn_select.pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="Next > (Right)", command=self.next_pair, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save & Exit (Enter)", command=self.save_and_exit, bg="lightblue", font=("Arial", 12)).pack(side=tk.RIGHT, padx=5)

        # Handle Resizing
        self.img_frame.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        # Debounce resize events
        if self.resize_timer:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(100, self.load_current_pair_no_idx_change)

    def load_current_pair(self):
        self.load_pair()

    def load_current_pair_no_idx_change(self):
        self.load_pair()

    def load_pair(self):
        if self.current_idx < 0 or self.current_idx >= len(self.df):
            return

        row = self.df.iloc[self.current_idx]
        patch_name = str(row.get("patch_name", f"sample_{self.current_idx:03d}"))
        
        # Get label from first 2 chars of image_file
        label_text = "Unknown"
        if "image_file" in row:
            raw_file = str(row["image_file"])
            if len(raw_file) >= 2:
                label_text = raw_file[:2].upper()
        
        safe_id = safe_name(patch_name)
        
        sar_path = IMAGE_DIR / f"{safe_id}__SAR.png"
        rgb_path = IMAGE_DIR / f"{safe_id}__RGB.png"
        
        status = f"Image {self.current_idx + 1} / {len(self.df)}: {patch_name}"
        self.lbl_status.config(text=status)
        
        # Pass the label text to be drawn on the image
        self.show_image(self.canvas_sar, sar_path, label_text)
        self.show_image(self.canvas_opt, rgb_path, label_text)
        
        if patch_name in self.selected_ids:
            self.btn_select.config(text="SELECTED (Space)", bg="green", fg="white")
        else:
            self.btn_select.config(text="Select (Space)", bg="lightgray", fg="black")

    def show_image(self, canvas, path, overlay_text=""):
        canvas.delete("all")
        # Get available size
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            cw, ch = 800, 600 # Default if not yet mapped
            
        if path.exists():
            try:
                img = Image.open(path)
                
                # If we want to draw text ON the image itself before scaling (or after)
                # To ensure it scales well, let's draw on the resized image or PIL image.
                # Drawing on PIL image is easier for positioning.
                
                # Calculate scale to fit
                iw, ih = img.size
                scale = min(cw / iw, ch / ih)
                
                # Resize
                nw, nh = int(iw * scale), int(ih * scale)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                
                # Draw text
                if overlay_text:
                    draw = ImageDraw.Draw(img)
                    # Try to use a large font
                    try:
                        # Use default font or specific ttf if available. 
                        # Adjust size relative to image
                        font_size = max(20, int(nh * 0.1)) 
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Position: Top Left
                    # Add a small shadow/outline for visibility
                    text_x, text_y = 10, 10
                    fill_color = "red"
                    outline_color = "black"
                    
                    # Draw outline/shadow
                    draw.text((text_x-1, text_y), overlay_text, font=font, fill=outline_color)
                    draw.text((text_x+1, text_y), overlay_text, font=font, fill=outline_color)
                    draw.text((text_x, text_y-1), overlay_text, font=font, fill=outline_color)
                    draw.text((text_x, text_y+1), overlay_text, font=font, fill=outline_color)
                    
                    # Draw main text
                    draw.text((text_x, text_y), overlay_text, font=font, fill=fill_color)

                tk_img = ImageTk.PhotoImage(img)
                canvas.image = tk_img # Keep ref
                
                # Center
                canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=tk_img)
            except Exception as e:
                canvas.create_text(cw//2, ch//2, text=f"Error: {e}", fill="red")
        else:
            canvas.create_text(cw//2, ch//2, text=f"Not Found", fill="white")

    def toggle_select(self):
        patch_name = str(self.df.iloc[self.current_idx]["patch_name"])
        if patch_name in self.selected_ids:
            self.selected_ids.remove(patch_name)
        else:
            self.selected_ids.append(patch_name)
        
        self.lbl_selected.config(text=f"Selected: {len(self.selected_ids)}")
        self.load_pair()

    def next_pair(self):
        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1
            self.load_pair()

    def prev_pair(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_pair()

    def save_and_exit(self):
        try:
            selected_df = self.df[self.df["patch_name"].isin(self.selected_ids)]
            selected_df.to_csv(OUTPUT_CSV, index=False)
            with open(SELECTED_IDS_FILE, "w") as f:
                f.write("\n".join(self.selected_ids))
            messagebox.showinfo("Success", f"Saved {len(self.selected_ids)} pairs to {OUTPUT_CSV}")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
    else:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows.")
        
        root = tk.Tk()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        # Start maximized or 95% size
        root.geometry(f"{int(w*0.95)}x{int(h*0.95)}")
        try:
            root.state('zoomed') 
        except:
            pass
        
        app = ImageSelectorApp(root, df)
        root.mainloop()
