import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from datetime import datetime
import os

# --- Environment Configuration ---
# Speed up HF hub downloads if allowed (no effect in local-only mode)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

class DNAAnalysis:
    def __init__(self, root):
        self.root = root
        self.root.title("DNA Analysis Toll")
        self.root.geometry("850x700")

        # --- Core: Model Selection Logic ---
        # We prioritize NT-500M-Human-Ref as the default for "Human vs Non-Human" detection.
        
        # Check for custom model path overrides in environment variables
        env_model_path = os.getenv("MODEL_PATH")
        env_local_only = os.getenv("LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")

        # Define available model options
        # Format: (Display Label, HuggingFace Model ID)
        self.model_options = [
            ("NT-500M Human Ref (Best for Human Detection)", "InstaDeepAI/nucleotide-transformer-500m-human-ref"),
            ("NT-2.5B 1000G (Best Accuracy for Human Variants)", "InstaDeepAI/nucleotide-transformer-2.5b-1000g"),
            ("NT-2.5B Multi-species (Best for Repair/General)", "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"),
        ]

        # If a custom path is provided via env, prepend it as the first option
        if env_model_path:
            self.model_options.insert(0, ("Custom (MODEL_PATH)", env_model_path))
            # Set the custom path as the initial default model name
            self.model_name = env_model_path
        else:
            # Default to the first option (500m-human-ref)
            self.model_name = self.model_options[0][1]

        # Map labels back to IDs for retrieval
        self.model_label_to_value = {label: value for label, value in self.model_options}
        
        # Set the dropdown variable to match the current self.model_name
        # If custom path matches nothing in standard list, it defaults to the first item visually
        default_label = next((label for label, value in self.model_options if value == self.model_name), self.model_options[0][0])
        self.model_select_var = tk.StringVar(value=default_label)

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Local-only toggle (defaults to True if a custom local path is set)
        self.local_only_var = tk.BooleanVar(value=bool(env_model_path) or env_local_only)

        # Model Context Configuration
        # Conservative max_length to ensure stability across different hardware
        self.max_length = 1000 
        self.chunk_overlap = 50
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure UI Styles
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        
        # --- 1. Input Section ---
        file_frame = ttk.LabelFrame(self.root, text="1. Input Data", padding=10)
        file_frame.pack(fill="x", padx=15, pady=5)
        
        self.path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.path_var, width=60).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Browse CSV", command=self.browse_file).pack(side="left")
        
        # --- 2. Configuration Section ---
        conf_frame = ttk.LabelFrame(self.root, text="2. Analysis Settings", padding=10)
        conf_frame.pack(fill="x", padx=15, pady=5)
        
        # Grid layout for settings
        grid_frame = ttk.Frame(conf_frame)
        grid_frame.pack(fill="x")
        
        # Column Selection
        ttk.Label(grid_frame, text="Target Column:", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.col_name_var = tk.StringVar(value="sequence_text")
        ttk.Entry(grid_frame, textvariable=self.col_name_var, width=25).grid(row=0, column=1, sticky="w", padx=5)
        
        # Task Selection
        ttk.Label(grid_frame, text="Analysis Task:", style="Header.TLabel").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.mode_var = tk.StringVar(value="classify")
        
        radio_frame = ttk.Frame(grid_frame)
        radio_frame.grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(radio_frame, text="Classification (Detect Human DNA)", variable=self.mode_var, value="classify").pack(side="left", padx=5)
        ttk.Radiobutton(radio_frame, text="Deep Repair (Fill 'N')", variable=self.mode_var, value="repair").pack(side="left", padx=5)
        
        # Low Memory Mode Toggle
        self.fp16_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(conf_frame, text="Enable FP16 (Low Memory Mode) - Recommended", 
                       variable=self.fp16_var).pack(anchor="w", padx=5, pady=(10,0))

        # Local files only toggle
        ttk.Checkbutton(conf_frame, text="Use local files only (no downloads)", 
                        variable=self.local_only_var).pack(anchor="w", padx=5, pady=(5,0))

        # Threshold Input
        ttk.Label(grid_frame, text="PPL Threshold:", style="Header.TLabel").grid(row=1, column=2, sticky="e", padx=5)
        self.threshold_var = tk.DoubleVar(value=45.0)
        spin = ttk.Spinbox(grid_frame, from_=1.0, to=500.0, increment=1.0, textvariable=self.threshold_var, width=8)
        spin.grid(row=1, column=3, sticky="w", padx=5)
        ttk.Label(grid_frame, text="(Lower = Stricter)", font=("Segoe UI", 8), foreground="gray").grid(row=1, column=4, sticky="w")
        
        # Model Selection Dropdown
        ttk.Label(grid_frame, text="Model:", style="Header.TLabel").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        model_combo = ttk.Combobox(grid_frame, textvariable=self.model_select_var, width=65, state="readonly")
        model_combo["values"] = [label for label, _ in self.model_options]
        model_combo.set(self.model_select_var.get())
        model_combo.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Helper label for model choice
        ttk.Label(conf_frame, text="Note: Use 'NT-500M Human Ref' for detecting Human vs. Contamination.", 
                 foreground="gray", font=("Segoe UI", 8)).pack(anchor="w", padx=5)

        # --- 3. Engine Control Section ---
        run_frame = ttk.LabelFrame(self.root, text="3. AI Engine Control", padding=10)
        run_frame.pack(fill="x", padx=15, pady=5)

        self.status_var = tk.StringVar(value=f"System Status: Idle (Device: {self.device.upper()})")
        ttk.Label(run_frame, textvariable=self.status_var, font=("Consolas", 10), foreground="#006400").pack(pady=5)
        
        btn_box = ttk.Frame(run_frame)
        btn_box.pack(pady=5)
        
        self.load_btn = ttk.Button(btn_box, text="Initialize Model", command=self.start_load_model)
        self.load_btn.pack(side="left", padx=10)
        
        self.process_btn = ttk.Button(btn_box, text="Start Processing", command=self.start_processing, state="disabled")
        self.process_btn.pack(side="left", padx=10)
        
        # --- 4. Logs & Progress ---
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=750, mode="determinate")
        self.progress.pack(pady=10)
        
        self.log_text = tk.Text(self.root, height=12, state="disabled", font=("Consolas", 9), bg="#F5F5F5")
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def log(self, msg):
        """Appends message to the UI log window."""
        self.log_text.config(state="normal")
        self.log_text.insert("end", f">> {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def browse_file(self):
        """Opens file dialog for CSV selection."""
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if f:
            self.path_var.set(f)
            self.log(f"Selected file: {os.path.basename(f)}")

    def start_load_model(self):
        """Starts model loading in a background thread."""
        self.load_btn.config(state="disabled")
        # Update selected model name from dropdown before launching load
        selected_label = self.model_select_var.get()
        self.model_name = self.model_label_to_value.get(selected_label, self.model_name)
        
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    def load_model_thread(self):
        """Handles model download and initialization."""
        try:
            local_only = self.local_only_var.get()
            mode_msg = "Local files only" if local_only else "Online/Local auto"
            self.status_var.set(f"Status: Loading Model... ({mode_msg})")
            self.log(f"Target Model: {self.model_name}")
            self.log(f"Loading Mode: {mode_msg}")
            
            self.log("Initializing Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=local_only
            )
            
            self.log("Loading Model Weights...")
            # VRAM Optimization
            if self.fp16_var.get() and self.device == "cuda":
                self.log("Mode: FP16 (Half Precision) - Saving VRAM")
                torch_dtype = torch.float16
            else:
                self.log("Mode: FP32 (Standard Precision)")
                torch_dtype = torch.float32
                
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                local_files_only=local_only
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.status_var.set(f"Status: Model Ready ({os.path.basename(self.model_name)} | {self.device})")
            self.log("Model Loaded Successfully!")
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
        except Exception as e:
            self.status_var.set("Status: Load Failed")
            self.log(f"CRITICAL ERROR: {str(e)}")
            self.log("Hint: If using 'local files only', ensure MODEL_PATH is correct or model is cached.")
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

    def calculate_perplexity(self, sequence):
        """
        Calculates the perplexity of a DNA sequence.
        Lower perplexity = The model finds this sequence 'familiar'.
        """
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"][0]
        
        # Chunking Strategy for long sequences
        chunks = []
        if len(input_ids) <= self.max_length:
            chunks.append(input_ids)
        else:
            # Sliding window
            stride = self.max_length - self.chunk_overlap
            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i : i + self.max_length]
                if len(chunk) > 10: chunks.append(chunk)

        if not chunks: return None

        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0).to(self.device)
                # Use input_ids as labels (masked language modeling loss)
                outputs = self.model(chunk, labels=chunk)
                loss = outputs.loss
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    count += 1
        
        if count == 0: return None
        return torch.exp(torch.tensor(total_loss / count)).item()

    def repair_sequence(self, sequence):
        """
        Repairs DNA sequence by predicting masked 'N' tokens.
        Fix: Handles 6-mer expansion by truncating prediction to 1 base.
        """
        # Basic validation
        if not sequence or 'N' not in sequence.upper():
            return sequence, "No 'N' found"

        # 1. Identify all 'N' positions for precise repair.
        # For simplicity/efficiency, we maintain the masked replacement logic
        # but will handle length control during the decoding phase.
        
        mask_str = self.tokenizer.mask_token
        # Note: Simple replacement makes the tokenizer see "...ACGT[MASK]ACGT..."
        # The NT tokenizer splits surrounding ACGT into bases or fragments, 
        # while [MASK] is treated as a 6-mer to be predicted.
        masked_seq_str = sequence.upper().replace('N', mask_str)
        
        inputs = self.tokenizer(masked_seq_str, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Length check
        if input_ids.shape[1] > self.max_length:
            return sequence, "Skipped (Too Long)"

        # Model Inference
        with torch.no_grad():
            logits = self.model(input_ids).logits

        # Find all mask positions
        mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_indices[0]) == 0:
             return sequence, "Tokenizer Error"

        # Get predicted token IDs
        predicted_ids = logits[mask_indices].argmax(dim=-1)
        
        # --- Critical Fix Start ---
        # We cannot directly replace input_ids with predicted_ids and decode, 
        # as that would insert the entire 6-mer (causing sequence length expansion).
        
        # We need to manually construct the repaired string.
        # Due to complex tokenizer behavior, the most robust method is:
        # 1. Convert input_ids back to a token list.
        # 2. For each [MASK] position, decode its predicted ID and take only the first character.
        
        # Convert original input_ids to list (assuming batch size is 1 here)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Create a pointer to iterate through predicted_ids
        pred_idx = 0
        
        restored_tokens = []
        for token in tokens:
            if token == mask_str:
                # This is a masked position; retrieve the corresponding predicted ID
                pred_id = predicted_ids[pred_idx]
                pred_idx += 1
                
                # Decode this token the full predicted 6-mer (e.g., might decode to "ACGTGC")
                pred_str = self.tokenizer.decode(pred_id)
                
                # Take only the first base (assuming 'N' represents a single point deletion)
                # Note: If 'N' represents a larger segment, keeping the full 6-mer might be better.
                # But in this "Deep Repair" context, we enforce 1:1 replacement.
                fix_base = pred_str[0] if pred_str else "N" 
                restored_tokens.append(fix_base)
            else:
                restored_tokens.append(token)
        
        # Reassemble the string
        # Use convert_tokens_to_string to handle any subword prefixes (like ##) correctly
        repaired_seq = self.tokenizer.convert_tokens_to_string(restored_tokens)
        
        # Clean up any spaces or residual special tokens
        repaired_seq = repaired_seq.replace(" ", "").replace(mask_str, "")
        
        # --- Critical Fix End ---
        
        return repaired_seq, "Repaired"


    def start_processing(self):
        input_file = self.path_var.get()
        col_name = self.col_name_var.get()
        mode = self.mode_var.get()
        
        if not input_file: 
            messagebox.showwarning("Input Error", "Please select a CSV file.")
            return
            
        self.process_btn.config(state="disabled")
        self.progress.configure(value=0)
        thresh = self.threshold_var.get()
        threading.Thread(target=self.process_thread, args=(input_file, col_name, mode, thresh), daemon=True).start()

    def process_thread(self, input_file, col_name, mode, threshold):
        """
        Main processing logic.
        Adapts interpretation based on which model is currently loaded.
        """
        try:
            # Determine context based on model name
            current_model_name = self.model_name.lower()
            is_human_ref = "human-ref" in current_model_name or "1000g" in current_model_name
            
            self.log(f"Starting Analysis... Mode: {mode.upper()}")
            self.log(f"Model Context: {'Human Reference' if is_human_ref else 'Multi-Species'}")

            df = pd.read_csv(input_file)
            if col_name not in df.columns: 
                raise ValueError(f"Column '{col_name}' missing in CSV.")
            
            results, details = [], []
            total = len(df)
            
            # Threshold tuning:
            # For NT-500M-Human-Ref, human DNA usually has perplexity < 20-40.
            # Random sequences or bacteria often have PPL > 100.
            #human_ppl_threshold = 45.0 

            for i, row in df.iterrows():
                seq = str(row[col_name]).strip()
                
                if not seq or seq.lower() == 'nan':
                    results.append(""); details.append("Empty")
                    continue
                
                if mode == "classify":
                    ppl = self.calculate_perplexity(seq)
                    if ppl is None:
                        results.append("Error"); details.append(-1)
                    else:
                        score = round(ppl, 2)
                        
                        # --- INTERPRETATION LOGIC ---
                        if is_human_ref:
                            # Low PPL = Matches Human Reference
                            # High PPL = Out of distribution (Contamination/Virus/Bacteria)
                            if ppl < threshold:    #human_ppl_threshold:
                                res_str = "Likely Human DNA"
                            else:
                                res_str = "Likely Contamination/Non-Human"
                        else:
                            # Multi-species model knows many species.
                            # Low PPL = Valid DNA of SOME species (could be bacteria).
                            # High PPL = Likely noise or very alien sequence.
                            if ppl < threshold:    #human_ppl_threshold:
                                res_str = "Valid DNA (Species Unknown)"
                            else:
                                res_str = "High Perplexity / Noise"

                        results.append(res_str)
                        details.append(score)
                else:
                    # Repair mode
                    fixed, status = self.repair_sequence(seq)
                    results.append(fixed); details.append(status)

                # UI Update
                if i % 5 == 0: # Update every 5 rows to reduce overhead
                    p = ((i+1)/total)*100
                    self.root.after(0, lambda v=p: self.progress.configure(value=v))
                    self.root.after(0, lambda idx=i: self.log(f"Processed {idx+1}/{total}"))

            # Save Results
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = f"processed_{mode}_{base_name}_{timestamp}.csv"
            
            if mode == "classify":
                df['classification_result'] = results
                df['perplexity_score'] = details
                df['model_used'] = self.model_name
            else:
                df['repaired_sequence'] = results
                df['repair_status'] = details
                
            df.to_csv(out_file, index=False)
            
            self.log(f"Done! Output saved to: {out_file}")
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Analysis Complete.\nSaved: {out_file}"))
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.progress.configure(value=0))

if __name__ == "__main__":
    root = tk.Tk()
    app = DNAAnalysis(root)
    root.mainloop()
