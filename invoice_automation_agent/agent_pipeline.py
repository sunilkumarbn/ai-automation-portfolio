import os
import shutil
import time
from pathlib import Path
from typing import TypedDict, List, Optional
import fitz  # PyMuPDF
import pandas as pd
from pydantic import BaseModel, Field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# --- 1. Folders & Config ---
BASE_DIR = Path(__file__).parent
INBOX_DIR = BASE_DIR / "inbox"
PROCESSED_DIR = BASE_DIR / "processed"
FAILED_DIR = BASE_DIR / "failed"
LEDGER_FILE = BASE_DIR / "ledger.xlsx"

for d in (INBOX_DIR, PROCESSED_DIR, FAILED_DIR):
    d.mkdir(exist_ok=True)

llm = ChatOllama(model="llama3.2", temperature=0)

# --- 2. Pydantic Output Schema ---
class LineItem(BaseModel):
    description: str
    amount: float

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the issuing company or vendor")
    invoice_number: str = Field(description="Unique invoice ID or number")
    invoice_date: str = Field(description="Date of invoice in YYYY-MM-DD format")
    subtotal: float = Field(description="Amount before tax")
    tax_amount: float = Field(default=0.0, description="Total tax/GST amount")
    total_amount: float = Field(description="Final payable total amount")
    line_items: List[LineItem] = Field(default_factory=list)

# --- 3. LangGraph State ---
class InvoiceState(TypedDict):
    file_path: str
    raw_text: str
    extracted_data: Optional[dict]
    validation_error: Optional[str]
    attempts: int
    status: str

# --- 4. Graph Nodes ---
def extract_text_node(state: InvoiceState) -> InvoiceState:
    doc = fitz.open(state["file_path"])
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return {**state, "raw_text": text, "attempts": 0}

def llm_extractor_node(state: InvoiceState) -> InvoiceState:
    structured_llm = llm.with_structured_output(InvoiceData)
    error_context = ""
    if state.get("validation_error"):
        error_context = (
            f"\nPREVIOUS ATTEMPT FAILED VALIDATION: {state['validation_error']}. "
            "Carefully re-check subtotal, tax_amount, and total_amount from the text."
        )

    prompt = (
        "Extract structured invoice details from the following document text. "
        "Ensure numbers are accurate floats.\n"
        f"{error_context}\n\nDOCUMENT TEXT:\n{state['raw_text']}"
    )
    result: InvoiceData = structured_llm.invoke(prompt)
    return {
        **state,
        "extracted_data": result.model_dump(),
        "attempts": state["attempts"] + 1,
    }

def validate_math_node(state: InvoiceState) -> InvoiceState:
    data = state["extracted_data"]
    expected_total = round(data["subtotal"] + data["tax_amount"], 2)
    actual_total = round(data["total_amount"], 2)

    if abs(expected_total - actual_total) > 0.05:
        err = f"Math mismatch: subtotal ({data['subtotal']}) + tax ({data['tax_amount']}) = {expected_total}, but total_amount = {actual_total}"
        return {**state, "validation_error": err, "status": "invalid"}

    return {**state, "validation_error": None, "status": "valid"}

def save_to_ledger_node(state: InvoiceState) -> InvoiceState:
    data = state["extracted_data"]
    row = {
        "Invoice Number": data["invoice_number"],
        "Vendor": data["vendor_name"],
        "Date": data["invoice_date"],
        "Subtotal": data["subtotal"],
        "Tax": data["tax_amount"],
        "Total": data["total_amount"],
        "Source File": Path(state["file_path"]).name,
    }
    df_new = pd.DataFrame([row])

    if LEDGER_FILE.exists():
        df_existing = pd.read_excel(LEDGER_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_excel(LEDGER_FILE, index=False)
    shutil.move(state["file_path"], PROCESSED_DIR / Path(state["file_path"]).name)
    print(f"[SUCCESS] Logged {data['invoice_number']} from {data['vendor_name']} -> {LEDGER_FILE.name}")
    return {**state, "status": "completed"}

def quarantine_node(state: InvoiceState) -> InvoiceState:
    shutil.move(state["file_path"], FAILED_DIR / Path(state["file_path"]).name)
    print(f"[FAILED] Moved {Path(state['file_path']).name} to failed/: {state['validation_error']}")
    return {**state, "status": "quarantined"}

# --- 5. Routing & Graph Compilation ---
def route_after_validation(state: InvoiceState) -> str:
    if state["status"] == "valid":
        return "save_to_ledger"
    if state["attempts"] < 2:
        return "llm_extractor"
    return "quarantine"

builder = StateGraph(InvoiceState)
builder.add_node("extract_text", extract_text_node)
builder.add_node("llm_extractor", llm_extractor_node)
builder.add_node("validate_math", validate_math_node)
builder.add_node("save_to_ledger", save_to_ledger_node)
builder.add_node("quarantine", quarantine_node)

builder.add_edge(START, "extract_text")
builder.add_edge("extract_text", "llm_extractor")
builder.add_edge("llm_extractor", "validate_math")
builder.add_conditional_edges(
    "validate_math",
    route_after_validation,
    {
        "save_to_ledger": "save_to_ledger",
        "llm_extractor": "llm_extractor",
        "quarantine": "quarantine",
    },
)
builder.add_edge("save_to_ledger", END)
builder.add_edge("quarantine", END)

invoice_graph = builder.compile()

# --- 6. Folder Watchdog Automation ---
class InboxHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            time.sleep(1)  # Wait for file write to complete
            print(f"\n[DETECTED] New PDF: {event.src_path}")
            invoice_graph.invoke({"file_path": event.src_path})

if __name__ == "__main__":
    # Process any existing PDFs first
    for pdf in INBOX_DIR.glob("*.pdf"):
        invoice_graph.invoke({"file_path": str(pdf)})

    observer = Observer()
    observer.schedule(InboxHandler(), str(INBOX_DIR), recursive=False)
    observer.start()
    print(f"Watching {INBOX_DIR} for incoming PDF invoices... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()